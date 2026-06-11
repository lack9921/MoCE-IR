#!/usr/bin/env python3
"""
test_moe_utilization.py — Per-image MoE utilization visualization.

For a single input image, runs through the model and shows
which experts are used in each of the 10 AdapterLayers.

Usage:
  python test_moe_utilization.py \
      --image /path/to/image.png \
      --weights /path/to/checkpoint.ckpt \
      --device cuda

Output:
  Per-layer × expert gate heatmap table
  Overall expert utilization summary
"""

import os, sys, argparse
from PIL import Image

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
    parser.add_argument('--image', type=str, required=True, help='Path to a single image')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    print(f"[*] Loading model...")
    model = load_model(args.weights, device)
    adapters = list(find_adapters(model))
    E = 4  # num experts
    expert_colors = ['🟦', '🟩', '🟨', '🟥']

    # ── Load image ──
    img = Image.open(args.image).convert('RGB')
    w, h = img.size
    pw, ph = (16 - w % 16) % 16, (16 - h % 16) % 16
    from torchvision import transforms
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)  # (1,3,H,W)
    if pw or ph:
        tensor = torch.nn.functional.pad(tensor, (0, pw, 0, ph))
    print(f"[*] Image: {args.image} ({w}x{h})")

    # ── Monkey-patch to capture gates ──
    captured_gates = {}
    captured_expert_outs = {}

    orig_forward = AdapterLayer.forward
    def patched_forward(self, x, freq_emb, shared):
        name = None
        for n, a in adapters:
            if a is self:
                name = n
                break

        # Capture routing gates
        gates, top_k_indices, top_k_values, _ = self.routing(x, freq_emb)
        captured_gates[name] = gates[0].detach().cpu()  # (E,)

        # Also capture per-expert raw outputs
        expert_outs = {}
        for ei in range(E):
            expert_outs[ei] = self.experts[ei](x, shared)
        captured_expert_outs[name] = {
            ei: out.detach().cpu() for ei, out in expert_outs.items()
        }

        return orig_forward(self, x, freq_emb, shared)

    AdapterLayer.forward = patched_forward

    # Run
    _ = model(tensor)
    AdapterLayer.forward = orig_forward

    # ── Print results ──
    print("\n" + "=" * 72)
    print(f"  MoE Utilization — Per-Layer Gate Values")
    print(f"  (top_k={model.dec[0][2].layers[0].adapter.top_k if hasattr(model.dec[0][2].layers[0].adapter, 'top_k') else '?'})")
    print("=" * 72)

    header = f"  {'Layer':<12} "
    header += "  ".join([f"{expert_colors[e]} Expert{e:<5}" for e in range(E)])
    print(header)
    print(f"  {'─' * 12}─" + "─" * 14 * E)

    all_gates = []
    layer_gates = {}
    for name, adapter in adapters:
        gates = captured_gates[name]
        all_gates.append(gates)
        layer_gates[name] = gates

        gate_strs = []
        for e in range(E):
            v = gates[e].item()
            if v > 0.5:
                bar = '████'
            elif v > 0.3:
                bar = '██▓░'
            elif v > 0.1:
                bar = '██░░'
            elif v > 0.01:
                bar = '▓░░░'
            else:
                bar = '░░░░'
            gate_strs.append(f"{bar} {v:.3f}")

        print(f"  {name:<12} │ {'  '.join(gate_strs)}")

    # ── Summary per layer ──
    print("\n" + "=" * 72)
    print("  Dominant Expert per Layer")
    print("=" * 72)
    print(f"  {'Layer':<12} {'Dominant':<10} {'Confidence':<12} {'Active Experts'}")
    print(f"  {'─' * 12}─{'─' * 10}─{'─' * 12}─{'─' * 20}")

    for name, adapter in adapters:
        gates = captured_gates[name]
        de = gates.argmax().item()
        conf = gates[de].item()
        active = [e for e in range(E) if gates[e].item() > 0.01]
        active_str = " ".join([f"{expert_colors[e]}E{e}" for e in active])
        print(f"  {name:<12} {expert_colors[de]}E{de:<7} {conf:.3f} ({conf*100:.0f}%)   {active_str}")

    # ── Overall statistics ──
    print("\n" + "=" * 72)
    print("  Overall Expert Utilization (across all 10 layers)")
    print("=" * 72)

    all_gates_t = torch.stack(all_gates)  # (10, E)
    avg_gates = all_gates_t.mean(dim=0)

    for e in range(E):
        bar_len = int(avg_gates[e].item() * 40)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        print(f"  {expert_colors[e]} Expert{e}  {bar}  {avg_gates[e].item():.3f}")

    print(f"\n  Routing entropy (avg): {-(avg_gates * torch.log(avg_gates + 1e-8)).sum().item():.3f}")

    # ── Expert Output Contribution ──
    print("\n" + "=" * 72)
    print("  Expert Output Magnitude (L2 norm of each expert's output)")
    print("  (Higher = more influence on final output)")
    print("=" * 72)

    for name, adapter in adapters:
        outs = captured_expert_outs[name]
        norms = {}
        for ei in range(E):
            norms[ei] = outs[ei].norm().item()

        total = sum(norms.values())
        pcts = {ei: v/total*100 for ei, v in norms.items()}

        bar_str = []
        for ei in range(E):
            bar_len = int(pcts[ei] / 5)
            bar = '█' * bar_len
            bar_str.append(f"{expert_colors[ei]}{bar} {pcts[ei]:.0f}%")
        print(f"  {name:<12} │ {'  '.join(bar_str)}")

    print(f"\n[*] Done.")


if __name__ == '__main__':
    main()
