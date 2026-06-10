#!/usr/bin/env python3
"""
test_routing.py — MoCE-IR Routing Expert Allocation Test (original main branch)

Hooks into every AdapterLayer's RoutingFunction to capture gates (gating scores)
for each input sample, then aggregates expert distribution per task.

Supports:
  - Real LoViF dataset (--input /path/to/LoViF)
  - Synthetic degradation simulation (no dataset needed)
  - Pretrained checkpoint loading (--weights)
  - Random-initialized model test (default)

Outputs:
  - Console table: Task → [DecoderBlock_1, Block_2, Block_3] → Expert distribution
  - JSON: detailed per-sample, per-block gate values
  - PNG heatmap: Task × Expert × DecoderBlock
"""

import os, sys, json, argparse, math
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from net.moce_ir import MoCEIR, AdapterLayer, RoutingFunction, FrequencyEmbedding


# ── Task definitions ────────────────────────────────────────────────
TASKS = ['Blur', 'Haze', 'Lowlight', 'Rain', 'Snow']
EXPERT_COLORS = ['🟦', '🟩', '🟨', '🟥']  # Expert 0-3


# ── Synthetic degradation generators (for testing without dataset) ──

def make_synthetic_blur(batch=4, size=256):
    """Gaussian blur kernel."""
    x = torch.rand(batch, 3, size, size)
    k = 15
    kernel = torch.zeros(1, 1, k, k)
    center = k // 2
    sigma = 3.0
    for i in range(k):
        for j in range(k):
            kernel[0, 0, i, j] = math.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.repeat(3, 1, 1, 1)
    return F.conv2d(x, kernel, padding=k//2, groups=3)

def make_synthetic_haze(batch=4, size=256):
    """Uniform mist scattering."""
    x = torch.rand(batch, 3, size, size)
    t = torch.rand(batch, 1, 1, 1) * 0.4 + 0.3  # transmission 0.3~0.7
    A = torch.rand(batch, 3, 1, 1) * 0.3 + 0.6   # airlight 0.6~0.9
    return x * t + A * (1 - t)

def make_synthetic_lowlight(batch=4, size=256):
    """Dark + amplified noise."""
    x = torch.rand(batch, 3, size, size) * 0.2  # dim
    noise = torch.randn_like(x) * 0.05
    return (x + noise).clamp(0, 1)

def make_synthetic_rain(batch=4, size=256):
    """Random streaks."""
    x = torch.rand(batch, 3, size, size)
    mask = torch.zeros(batch, 1, size, size)
    for _ in range(40):
        xs = np.random.randint(0, size-20)
        ys = np.random.randint(0, size-5)
        angle = np.random.uniform(-0.3, 0.3)
        for d in range(20):
            yi = ys + d
            xi = int(xs + d * angle)
            if 0 <= xi < size and 0 <= yi < size:
                mask[:, :, yi, xi] += 1.0
    mask = mask.clamp(0, 1).repeat(1, 3, 1, 1)
    return x * (1 - mask) + 0.7 * mask

def make_synthetic_snow(batch=4, size=256):
    """Random snowflakes."""
    x = torch.rand(batch, 3, size, size)
    mask = torch.zeros(batch, 1, size, size)
    for _ in range(200):
        xs = np.random.randint(0, size)
        ys = np.random.randint(0, size)
        r = np.random.randint(2, 6)
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    xi, yi = xs+dx, ys+dy
                    if 0 <= xi < size and 0 <= yi < size:
                        mask[:, :, yi, xi] += 1.0
    mask = mask.clamp(0, 1).repeat(1, 3, 1, 1)
    return x * (1 - mask) + 1.0 * mask

SYNTHETIC_MAKERS = {
    'Blur': make_synthetic_blur,
    'Haze': make_synthetic_haze,
    'Lowlight': make_synthetic_lowlight,
    'Rain': make_synthetic_rain,
    'Snow': make_synthetic_snow,
}


# ── Routing hook ────────────────────────────────────────────────────

class RoutingHook:
    """
    Hook manager: monkey-patches every RoutingFunction in the model to 
    capture gate values during forward.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.gates: Dict[int, List[torch.Tensor]] = defaultdict(list)
        # Track decoder block index from AdapterLayer nesting
        self._block_idx = 0
        self._layer_idx = 0
        self._orig_forwards = {}

    def _make_hook(self, block_idx: int, layer_idx: int):
        """Return a wrapper for RoutingFunction.forward that captures gates."""
        def hooked_forward(orig_fn, self_rf, x, freq_emb):
            gates, top_k_indices, top_k_values, aux_loss = orig_fn(self_rf, x, freq_emb)
            # gates: (B, num_experts) — soft gating scores (only top-k non-zero)
            # Store for our analysis
            hook_key = (block_idx, layer_idx)
            self.gates[hook_key].append(gates.detach().cpu())
            return gates, top_k_indices, top_k_values, aux_loss
        return hooked_forward

    def install(self):
        """Find all AdapterLayer → RoutingFunction and hook them."""
        self.gates.clear()
        self._block_idx = 0
        self._layer_idx = 0

        def _recurse(module, path=""):
            if isinstance(module, nn.ModuleList):
                for i, child in enumerate(module):
                    _recurse(child, f"{path}.{i}")
            elif isinstance(module, AdapterLayer):
                # Found an AdapterLayer with a routing sub-module
                rf = module.routing
                orig_fn = type(rf).forward
                block_idx = self._block_idx
                layer_idx = self._layer_idx
                self._layer_idx += 1

                # Monkey-patch
                def make_wrapper(orig_forward, bi, li):
                    def wrapper(x, freq_emb):
                        gates, top_k_indices, top_k_values, aux_loss = orig_forward(x, freq_emb)
                        hook_key = (bi, li)
                        self.gates[hook_key].append(gates.detach().cpu())
                        return gates, top_k_indices, top_k_values, aux_loss
                    return wrapper

                self._orig_forwards[(block_idx, layer_idx)] = (rf, orig_fn)
                rf.forward = make_wrapper(orig_fn.__get__(rf, type(rf)), block_idx, layer_idx)

            elif isinstance(module, (nn.Sequential, nn.Module)):
                for name, child in module._modules.items():
                    if child is not None:
                        _recurse(child, f"{path}.{name}")

        # Track block index across decoders
        for dec_idx, dec_group in enumerate(self.model.dec):
            # dec_group = [Upsample, FusionConv, DecoderResidualGroup]
            drg = dec_group[2]  # DecoderResidualGroup
            self._block_idx = dec_idx
            self._layer_idx = 0
            _recurse(drg, f"dec.{dec_idx}")

    def uninstall(self):
        """Restore original forward methods."""
        for key, (rf, orig_fn) in self._orig_forwards.items():
            rf.forward = orig_fn.__get__(rf, type(rf))
        self._orig_forwards.clear()

    def get_stats(self) -> Dict:
        """
        Aggregate gate statistics.

        Returns:
          {
            (block_idx, layer_idx): {
              'tasks': { task_name: { 'counts': [n0, n1, n2, n3], 'probs': [p0, p1, p2, p3] } },
              'overall': { 'counts': [...], 'probs': [...] },
            }
          }
        """
        return self.gates  # raw dict: (block, layer) → list of (B, E) tensors


# ── Helpers ─────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str):
    """Load checkpoint, auto-detect config. Simplified for routing test."""
    raw = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' in raw:
        sd = {k.replace('net.', ''): v for k, v in raw['state_dict'].items() if k.startswith('net.')}
    elif 'model_state_dict' in raw:
        sd = raw['model_state_dict']
    else:
        sd = raw

    # Auto-detect dim
    pe_key = [k for k in sd if 'patch_embed.proj.weight' in k]
    dim = sd[pe_key[0]].shape[0] if pe_key else 48

    # Detect levels
    enc_keys = [k for k in sd if k.startswith('enc.') and '.layers.' in k]
    max_level = 0
    for k in enc_keys:
        try:
            max_level = max(max_level, int(k.split('.')[1]) + 1)
        except (ValueError, IndexError):
            pass
    levels = max(4, max_level)

    # Heads (default for main branch MoCEIR-S)
    heads = [1, 2, 4, 8]

    # Blocks
    num_blocks = []
    for i in range(levels - 1):
        bk = [k for k in sd if k.startswith(f'enc.{i}.0.layers.')]
        nb = max((int(k.split('layers.')[1].split('.')[0]) for k in bk), default=4) + 1
        num_blocks.append(nb)

    latent_blocks = [k for k in sd if k.startswith('latent.layers.')]
    latent_n = max((int(k.split('layers.')[1].split('.')[0]) for k in latent_blocks), default=4) + 1
    num_blocks.append(latent_n)

    num_dec_blocks = []
    for i in range(levels - 1):
        dk = [k for k in sd if k.startswith(f'dec.{i}.2.layers.')]
        nd = max((int(k.split('layers.')[1].split('.')[0]) for k in dk), default=3) + 1
        num_dec_blocks.append(nd)
    num_dec_blocks = num_dec_blocks[::-1] if num_dec_blocks else [2, 4, 4]

    ref_blocks = [k for k in sd if k.startswith('refinement.layers.')]
    num_refinement_blocks = max((int(k.split('layers.')[1].split('.')[0]) for k in ref_blocks), default=3) + 1

    return sd, {
        'dim': dim,
        'num_blocks': num_blocks,
        'num_dec_blocks': num_dec_blocks,
        'levels': levels,
        'heads': heads,
        'num_refinement_blocks': num_refinement_blocks,
        'topk': 2,
        'num_experts': 4,
        'rank': 2,
        'with_complexity': False,
        'complexity_scale': 'max',
        'rank_type': 'spread',
        'depth_type': 'constant',
        'stage_depth': [1, 1, 1],
    }


# ── Static analysis: compute expert complexity ──────────────────────

def print_expert_complexity(model):
    """Print each expert's parameter count and architecture."""
    print("\n  ── Expert Complexity ──")
    for dec_idx, dec_group in enumerate(model.dec):
        drg = dec_group[2]  # DecoderResidualGroup
        for layer_idx, layer in enumerate(drg.layers):
            adapter = layer.adapter
            for e_idx, expert in enumerate(adapter.experts):
                n_params = sum(p.numel() for p in expert.parameters())
                # Get the FFTAttention inside
                for m in expert.modules():
                    if hasattr(m, 'patch_size'):
                        print(f"  dec.{dec_idx}.layer.{layer_idx} | Expert{e_idx} "
                              f"patch={m.patch_size} kernel={m.conv_kernel_size if hasattr(m, 'conv_kernel_size') else '?'} "
                              f"params={n_params:,}")


# ── Print routing heatmap ───────────────────────────────────────────

def print_routing_table(stats, tasks):
    """
    stats: dict (block_idx, layer_idx) → list of gates tensors (B, E)
           grouped by task in order of tasks list.
    
    We need to know which task each gates tensor belongs to.
    Since we run tasks sequentially, we track this externally.
    """
    pass  # We'll build this in main()


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MoCE-IR Routing Expert Allocation Test')
    parser.add_argument('--weights', type=str, default=None,
                        help='Checkpoint path (optional; random init if omitted)')
    parser.add_argument('--input', type=str, default=None,
                        help='LoViF dataset root with Blur/Haze/.../GT+LQ subdirs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch', type=int, default=8, help='Synthetic batch per task')
    parser.add_argument('--size', type=int, default=224, help='Image size for synthetics')
    parser.add_argument('--output', type=str, default='routing_analysis.json',
                        help='Save path for JSON analysis')

    # MoCEIR-S config (matches default in options.py)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[4, 6, 6, 8])
    parser.add_argument('--num_dec_blocks', type=int, nargs='+', default=[2, 4, 4])
    parser.add_argument('--levels', type=int, default=4)
    parser.add_argument('--heads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--rank_type', type=str, default='spread')
    parser.add_argument('--depth_type', type=str, default='constant')
    parser.add_argument('--stage_depth', type=int, nargs='+', default=[1, 1, 1])

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    print(f"[*] Device: {device}")

    # ── Build model ──
    print("[*] Building MoCEIR model...")
    model = MoCEIR(
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_dec_blocks=args.num_dec_blocks,
        levels=args.levels,
        heads=args.heads,
        num_refinement_blocks=args.num_refinement_blocks,
        topk=args.topk,
        num_experts=args.num_experts,
        rank=args.rank,
        with_complexity=False,   # Original main branch — no complexity bias by default
        complexity_scale='max',
        rank_type=args.rank_type,
        depth_type=args.depth_type,
        stage_depth=args.stage_depth,
    ).to(device).eval()

    # Load checkpoint if provided
    if args.weights:
        print(f"[*] Loading checkpoint: {args.weights}")
        sd, cfg = load_checkpoint(args.weights)
        if args.dim != cfg['dim']:
            print(f"  [*] Using auto-detected dim={cfg['dim']} from checkpoint")
            # Rebuild with correct dim
            model = MoCEIR(
                dim=cfg['dim'],
                num_blocks=cfg['num_blocks'],
                num_dec_blocks=cfg['num_dec_blocks'],
                levels=cfg['levels'],
                heads=cfg['heads'],
                num_refinement_blocks=cfg['num_refinement_blocks'],
                topk=cfg['topk'],
                num_experts=cfg['num_experts'],
                rank=cfg['rank'],
                with_complexity=cfg['with_complexity'],
                complexity_scale=cfg['complexity_scale'],
                rank_type=cfg['rank_type'],
                depth_type=cfg['depth_type'],
                stage_depth=cfg['stage_depth'],
            ).to(device).eval()
            model.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(sd, strict=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model params: {n_params:,} ({n_params/1e6:.2f}M)")

    # Print expert complexity
    print_expert_complexity(model)

    # Count AdapterLayers
    n_adapters = 0
    for dec_group in model.dec:
        drg = dec_group[2]
        n_adapters += len(drg.layers)
    print(f"[*] AdapterLayers found: {n_adapters}")
    print(f"[*] Experts per layer: {args.num_experts}, top_k={args.topk}")

    # ── Install routing hooks ──
    hook = RoutingHook(model)
    hook.install()
    print("[*] Routing hooks installed ✓")

    # ── Prepare data ──
    if args.input:
        # Real dataset mode
        print(f"[*] Using real dataset: {args.input}")
        data_sources = {}
        for task in TASKS:
            lq_dir = os.path.join(args.input, task, 'LQ')
            if os.path.isdir(lq_dir):
                files = sorted([
                    f for f in os.listdir(lq_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
                ])
                if files:
                    data_sources[task] = [(lq_dir, files)]
                    print(f"  {task}: {len(files)} images")
                else:
                    print(f"  {task}: 0 images found, using synthetic")
                    data_sources[task] = None
            else:
                print(f"  {task}: no LQ dir, using synthetic")
                data_sources[task] = None
    else:
        print("[*] No dataset provided — using synthetic degradation images")
        data_sources = {task: None for task in TASKS}

    # ── Run inference per task, capture gates ──
    # gates_by_task: { task_name: [(block, layer), gates_tensor] }
    # We store as list of (hook_key, gates) tuples for each task
    task_gates = {task: defaultdict(list) for task in TASKS}

    @torch.no_grad()
    def run_task(task, batch_input):
        """Run one forward pass for each sample (batch=1) and capture gates.
        
        Original MoCE-IR inference only supports batch=1 due to a squeezed
        index issue in AdapterLayer.forward() inference path.
        """
        for i in range(batch_input.shape[0]):
            single = batch_input[i:i+1].to(device)  # (1, C, H, W)
            hook.gates.clear()
            _ = model(single)
            for hook_key, gates_list in hook.gates.items():
                for g in gates_list:
                    task_gates[task][hook_key].append(g)

    with torch.no_grad():
        for task in TASKS:
            print(f"\n{'=' * 60}")
            print(f"[{task}] Generating samples and capturing routing...")

            if data_sources[task] is not None:
                # Real images
                lq_dir, files = data_sources[task]
                batch_size = args.batch
                all_images = []
                for fname in files[:100]:  # cap at 100 per task
                    img = Image.open(os.path.join(lq_dir, fname)).convert('RGB')
                    img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
                    all_images.append(img_t)
                    if len(all_images) >= batch_size:
                        batch_tensor = torch.stack(all_images)  # (B, 3, H, W)
                        run_task(task, batch_tensor)
                        all_images = []
                if all_images:
                    batch_tensor = torch.stack(all_images)
                    run_task(task, batch_tensor)
                n_samples = sum(len(v) for v in task_gates[task].values())
                print(f"  → {n_samples} routing decisions captured from real images")
            else:
                # Synthetic images
                maker = SYNTHETIC_MAKERS[task]
                for _ in range(5):  # 5 batches = 5 × batch routing decisions
                    batch_img = maker(args.batch, args.size)
                    run_task(task, batch_img)
                n_samples = sum(len(v) for v in task_gates[task].values())
                print(f"  → {n_samples} routing decisions captured from synthetic images")

    # ── Uninstall hooks ──
    hook.uninstall()

    # ── Aggregate and display ──
    print("\n" + "=" * 72)
    print("  MoCE-IR Routing Expert Allocation Analysis")
    print("=" * 72)

    # Collect all hook keys (block, layer) across all tasks
    all_hook_keys = set()
    for task in TASKS:
        all_hook_keys.update(task_gates[task].keys())
    all_hook_keys = sorted(all_hook_keys, key=lambda k: (k[0], k[1]))

    analysis = {}

    for hk in all_hook_keys:
        block, layer = hk
        print(f"\n  ── Decoder Block {block} / Layer {layer} ──")

        block_data = {}

        for task in TASKS:
            gates_tensors = task_gates[task].get(hk, [])
            if not gates_tensors:
                print(f"    {task:<12} → (no data)")
                continue

            # Concatenate all gates: (N_samples, E)
            all_gates = torch.cat(gates_tensors, dim=0)  # (N, E)
            N = all_gates.shape[0]
            E = all_gates.shape[1]

            # For each sample, get the assigned expert (top-1 from gates)
            assigned = all_gates.argmax(dim=1)  # (N,)
            counts = torch.zeros(E, dtype=torch.long)
            for e in range(E):
                counts[e] = (assigned == e).sum()

            probs = counts.float() / N

            # Average gate value per expert
            avg_gates = all_gates.mean(dim=0)

            bar = " ".join([
                f"{EXPERT_COLORS[e] if e < len(EXPERT_COLORS) else '🔘'} "
                f"E{e}: {probs[e].item()*100:5.1f}% "
                f"(gate={avg_gates[e].item():.3f})"
                for e in range(E)
            ])
            print(f"    {task:<12} │ {bar}")

            block_data[task] = {
                'n_samples': N,
                'counts': counts.tolist(),
                'probs': [round(p.item(), 4) for p in probs],
                'avg_gates': [round(g.item(), 4) for g in avg_gates],
                'dominant_expert': int(assigned.mode().values.item()) if N > 0 else -1,
                'dominant_prob': float(probs[assigned.mode().values].item()) if N > 0 else 0.0,
            }

        analysis[f"decoder_{block}_layer_{layer}"] = block_data

        # Overall dominant expert per task for this block
        print(f"    {'─' * 60}")
        dominant_line = "    DOMINANT    │ "
        for task in TASKS:
            bd = block_data.get(task, {})
            de = bd.get('dominant_expert', -1)
            dp = bd.get('dominant_prob', 0)
            dominant_line += f"{task[:4]}→E{de}({dp*100:.0f}%)  "
        print(dominant_line)

    # ── Cross-layer analysis: which expert dominates for each task? ──
    print("\n" + "=" * 72)
    print("  Cross-Layer Dominant Expert Matrix")
    print("=" * 72)
    print(f"  {'Layer':<20} ", end="")
    for task in TASKS:
        print(f"{task:<10} ", end="")
    print()
    print(f"  {'─' * 20}─" + "─" * 10 * len(TASKS))

    for hk in all_hook_keys:
        label = f"dec.{hk[0]}.layer.{hk[1]}"
        print(f"  {label:<20} ", end="")
        for task in TASKS:
            bd = analysis.get(f"decoder_{hk[0]}_layer_{hk[1]}", {}).get(task, {})
            de = bd.get('dominant_expert', -1)
            dp = bd.get('dominant_prob', 0)
            if de >= 0:
                # Colored cell
                icon = EXPERT_COLORS[de] if de < len(EXPERT_COLORS) else '?'
                print(f"  {icon}E{de} {dp*100:3.0f}%  ", end="")
            else:
                print(f"  {'─':>8}  ", end="")
        print()

    # ── Entropy analysis: how deterministic is each task's routing? ──
    print("\n" + "=" * 72)
    print("  Routing Entropy per Task (lower = more deterministic)")
    print("=" * 72)
    print(f"  {'Task':<12} ", end="")
    for hk in all_hook_keys:
        print(f"dec{hk[0]}.l{hk[1]}  ", end="")
    print("  AVG")
    print(f"  {'─' * 12} " + "─" * 12 * len(all_hook_keys) + " ────")

    for task in TASKS:
        entropies = []
        print(f"  {task:<12} ", end="")
        for hk in all_hook_keys:
            bd = analysis.get(f"decoder_{hk[0]}_layer_{hk[1]}", {}).get(task, {})
            probs = bd.get('probs', [])
            if probs:
                # H = -sum(p * log(p))
                p = torch.tensor(probs)
                p = p[p > 0]
                H = -(p * p.log()).sum().item()
                entropies.append(H)
                print(f" {H:.3f}   ", end="")
            else:
                print(f" {'─':>6}  ", end="")
        avg_H = np.mean(entropies) if entropies else 0
        print(f"  {avg_H:.3f}")

    # ── Save JSON ──
    result = {
        'config': {
            'model': 'MoCEIR',
            'dim': args.dim,
            'levels': args.levels,
            'heads': args.heads,
            'num_experts': args.num_experts,
            'topk': args.topk,
            'dataset': args.input or 'synthetic',
            'weights': args.weights,
        },
        'num_adapters': n_adapters,
        'expert_columns': EXPERT_COLORS,
        'layers': analysis,
    }

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[✓] Analysis saved: {args.output}")


if __name__ == '__main__':
    main()
