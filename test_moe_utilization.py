#!/usr/bin/env python3
"""
test_moe_utilization.py — Per-task average MoE utilization across full test set.

Usage:
  python test_moe_utilization.py \
      --weights /path/to/checkpoint.ckpt \
      --input /path/to/LoViF/test \
      --device cuda

Output:
  Per-task × per-layer expert gate heatmap
  Average expert utilization across all layers
"""

import os, sys, argparse
from PIL import Image
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from net.moce_ir import MoCEIR, AdapterLayer

TASKS = ['Blur', 'Haze', 'Lowlight', 'Rain', 'Snow']
EXPERT_COLORS = ['🟦', '🟩', '🟨', '🟥']


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
    parser.add_argument('--input', type=str, required=True, help='LoViF dataset root')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-per-task', type=int, default=100)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    print(f"[*] Loading model...")
    model = load_model(args.weights, device)
    adapters = list(find_adapters(model))
    print(f"[*] Found {len(adapters)} AdapterLayers")

    # Install gate-capture hook
    captured = defaultdict(list)  # {(name, task): list of gates tensors}

    orig_forward = AdapterLayer.forward
    def patched_forward(self, x, freq_emb, shared):
        name = None
        for n, a in adapters:
            if a is self:
                name = n
                break
        gates, *_ = self.routing(x, freq_emb)
        # We'll attach the task label later through the call chain
        captured[name].append(gates[0].detach().cpu())
        return orig_forward(self, x, freq_emb, shared)

    AdapterLayer.forward = patched_forward

    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    # Track per-task gates
    task_gates = {t: {n: [] for n, _ in adapters} for t in TASKS}

    for task in TASKS:
        lq_dir = os.path.join(args.input, task, 'LQ')
        if not os.path.isdir(lq_dir):
            print(f"  [!] No LQ dir: {lq_dir}")
            continue

        files = sorted([f for f in os.listdir(lq_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
        files = files[:args.max_per_task]

        for fname in files:
            img = Image.open(os.path.join(lq_dir, fname)).convert('RGB')
            w, h = img.size
            pw, ph = (16 - w % 16) % 16, (16 - h % 16) % 16
            tensor = to_tensor(img).unsqueeze(0).to(device)
            if pw or ph:
                tensor = torch.nn.functional.pad(tensor, (0, pw, 0, ph))

            captured.clear()
            _ = model(tensor)

            for name, gates_list in captured.items():
                task_gates[task][name].extend(gates_list)

        print(f"  [{task}] {len(files)} images ✓")

    AdapterLayer.forward = orig_forward

    # ── Print results ──
    E = 4

    # Per-layer × per-task heatmap
    print("\n" + "=" * 72)
    print("  Average Expert Gate Values per Layer per Task")
    print("=" * 72)
    print()

    for name, _ in adapters:
        print(f"  ── {name} ──")
        hdr = f"  {'Task':<10} "
        for e in range(E):
            hdr += f"  {EXPERT_COLORS[e]} E{e}     "
        print(hdr)
        print(f"  {'─' * 10}─" + "─" * 11 * E)

        for task in TASKS:
            gates_list = task_gates[task][name]
            if not gates_list:
                continue
            g = torch.stack(gates_list).mean(dim=0)  # (E,)

            bar_str = []
            for e in range(E):
                v = g[e].item()
                bar_len = int(v * 30)
                bar = '█' * bar_len + '░' * (30 - bar_len)
                bar_str.append(f"{bar} {v:.3f}")
            print(f"  {task:<10} │ {'  '.join(bar_str)}")
        print()

    # Cross-layer dominant expert matrix
    print("=" * 72)
    print("  Dominant Expert Matrix (most-used expert per layer per task)")
    print("=" * 72)
    print(f"  {'Layer':<20} ", end="")
    for task in TASKS:
        print(f"  {task:<10}", end="")
    print()
    print(f"  {'─' * 20}─" + "─" * 12 * len(TASKS))

    for name, _ in adapters:
        print(f"  {name:<20} ", end="")
        for task in TASKS:
            gates_list = task_gates[task][name]
            if not gates_list:
                print(f"  {'─':>10} ", end="")
                continue
            g = torch.stack(gates_list).mean(dim=0)
            de = g.argmax().item()
            cp = g[de].item()
            print(f"  {EXPERT_COLORS[de]}E{de} {cp*100:.0f}%", end="")
        print()

    # Overall average utilization per task
    print("\n" + "=" * 72)
    print("  Overall Expert Utilization (avg across all 10 layers)")
    print("=" * 72)

    for task in TASKS:
        # Average gates across all layers
        all_avg = []
        for name, _ in adapters:
            gates_list = task_gates[task][name]
            if gates_list:
                all_avg.append(torch.stack(gates_list).mean(dim=0))
        if not all_avg:
            continue
        avg_g = torch.stack(all_avg).mean(dim=0)

        print(f"\n  [{task}]")
        total = avg_g.sum().item()
        for e in range(E):
            pct = avg_g[e].item() / total * 100
            bar_len = int(pct / 3)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            print(f"  {EXPERT_COLORS[e]} E{e}: {bar} {pct:5.1f}%  (gate={avg_g[e].item():.3f})")

        # Entropy
        p = avg_g / avg_g.sum()
        H = -(p * torch.log(p + 1e-8)).sum().item()
        print(f"  {'':>2} Routing entropy: {H:.3f}  (1.39=random, <0.5=deterministic)")

    print("\n[*] Done.")


if __name__ == '__main__':
    main()
