#!/usr/bin/env python3
"""
test_expert_similarity.py — Check if MoCE-IR experts converged to similar functions.

Strategy:
  1. Run model with torch.no_grad() on random inputs
  2. Monkey-patch each AdapterLayer to capture (input, shared_features)
  3. Then feed the SAME input to ALL 4 experts manually
  4. Compute pairwise output cosine similarity per layer

Usage:
  python test_expert_similarity.py --weights /path/to/checkpoint.ckpt [--device cuda]
"""

import os, sys, argparse, json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from net.moce_ir import MoCEIR, AdapterLayer


def load_model(ckpt_path, device='cuda'):
    raw = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in raw:
        sd = {k.replace('net.', ''): v for k, v in raw['state_dict'].items() if k.startswith('net.')}
    else:
        sd = raw

    pe_key = [k for k in sd if 'patch_embed.proj.weight' in k]
    dim = sd[pe_key[0]].shape[0] if pe_key else 48
    has_cls = any('task_proj' in k for k in sd)
    cls_dim = 128 if has_cls else 0

    model = MoCEIR(
        dim=dim, num_blocks=[4, 6, 6, 8], num_dec_blocks=[2, 4, 4],
        levels=4, heads=[1, 2, 4, 8], num_refinement_blocks=4,
        topk=1, num_experts=4, rank=2,
        with_complexity=has_cls, complexity_scale='max',
        rank_type='spread', depth_type='constant', stage_depth=[1, 1, 1],
        cls_dim=cls_dim,
    )

    new_sd = model.state_dict()
    for k in new_sd:
        if k in sd and new_sd[k].shape == sd[k].shape:
            new_sd[k] = sd[k]
        elif k in sd and 'freq_gate.weight' in k and new_sd[k].shape != sd[k].shape:
            d_old = sd[k].shape[1]
            new_sd[k][:, :d_old] = sd[k][:, :d_old]
    model.load_state_dict(new_sd, strict=False)
    return model.to(device).eval()


def find_adapters(module, prefix=''):
    for name, child in module._modules.items():
        if child is None:
            continue
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, AdapterLayer):
            yield full, child
        else:
            yield from find_adapters(child, full)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-samples', type=int, default=32)
    parser.add_argument('--input-size', type=int, default=128)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    print(f"[*] Loading {args.weights}")
    model = load_model(args.weights, device)
    adapters = list(find_adapters(model))
    print(f"[*] Found {len(adapters)} AdapterLayers")

    # ── Strategy: Monkey-patch each AdapterLayer to save (input, shared) ──
    adapter_inputs = {}  # {(dec_idx, layer_idx): (input_tensor, shared_tensor)}

    orig_forward = AdapterLayer.forward

    def patched_forward(self, x, freq_emb, shared):
        # Save inputs before routing
        key = (id(self),)  # unique id
        adapter_inputs[id(self)] = (x.detach().cpu(), shared.detach().cpu())
        return orig_forward(self, x, freq_emb, shared)

    AdapterLayer.forward = patched_forward

    # Run inference
    x = torch.randn(args.n_samples, 3, args.input_size, args.input_size, device=device)
    _ = model(x)

    # Restore
    AdapterLayer.forward = orig_forward

    # ── Now manually feed saved inputs to each expert ──
    expert_colors = ['🟦', '🟩', '🟨', '🟥']
    metrics = {}

    for name, adapter in adapters:
        key = id(adapter)
        if key not in adapter_inputs:
            print(f"  [!] No input captured for {name}")
            continue

        inp_x, inp_shared = adapter_inputs[key]
        inp_x = inp_x.to(device)
        inp_shared = inp_shared.to(device)

        # Run ALL 4 experts on the SAME input
        expert_outs = {}
        for ei in range(4):
            out = adapter.experts[ei](inp_x, inp_shared)
            expert_outs[ei] = out.detach().cpu()

        # Pairwise cosine similarity (mean-pool spatial → per-sample vectors)
        cos_mat = torch.zeros((4, 4))
        for i in range(4):
            vi = expert_outs[i].flatten(1)  # (N, C*H*W)
            for j in range(4):
                vj = expert_outs[j].flatten(1)
                # Cosine sim per sample, then average
                cos_per_sample = torch.nn.functional.cosine_similarity(vi, vj, dim=1)
                cos_mat[i, j] = cos_per_sample.mean().item()

        metrics[name] = cos_mat.tolist()

    # ── Print ──
    print("\n" + "=" * 72)
    print("  Expert Output Cosine Similarity (same input → different expert)")
    print("  🟢 >0.999 = nearly identical    🟡 >0.99 = very similar")
    print("  🟠 >0.9 = somewhat similar      🔴 <0.9 = genuinely different")
    print("=" * 72)

    all_off_diag = []
    for name in sorted(metrics.keys()):
        cs = metrics[name]
        print(f"\n  [{name}]")
        print(f"  {'':>8} ", end="")
        for ej in range(4):
            print(f"  {expert_colors[ej]}E{ej}  ", end="")
        print()
        for ei in range(4):
            print(f"  {expert_colors[ei]}E{ei:<5} ", end="")
            for ej in range(4):
                v = cs[ei][ej]
                marker = '🟢' if v >= 0.999 else ('🟡' if v >= 0.99 else ('🟠' if v >= 0.9 else '🔴'))
                print(f"  {marker}{v:.4f}", end="")
            print()

        off_diag = [cs[i][j] for i in range(4) for j in range(4) if i != j]
        all_off_diag.extend(off_diag)
        print(f"  {''.rjust(8)} Mean off-diag: {np.mean(off_diag):.4f}")

    print(f"\n  {'─' * 40}")
    print(f"  Overall off-diag mean: {np.mean(all_off_diag):.4f}")

    mean_val = np.mean(all_off_diag)
    if mean_val > 0.999:
        print(f"  🔴 VERDICT: Experts are NEARLY IDENTICAL → routing doesn't matter")
    elif mean_val > 0.99:
        print(f"  🟡 VERDICT: Experts are very similar → routing has marginal effect")
    elif mean_val > 0.9:
        print(f"  🟠 VERDICT: Moderate similarity → routing matters somewhat")
    else:
        print(f"  🟢 VERDICT: Experts are genuinely different → routing SHOULD matter")

    # ── Note: weight comparison skipped (experts have different rank shapes) ──
    print("[*] Done.\n")




















if __name__ == '__main__':
    main()
