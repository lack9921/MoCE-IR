#!/usr/bin/env python3
"""
extract_frequency.py — Visualize what MoCE-IR's FrequencyEmbedding "sees".

Extracts frequency response maps and frequency embedding vectors
for each image in the validation set, for qualitative analysis.

Usage:
  python extract_frequency.py \
      --weights /path/to/checkpoint.ckpt \
      --input /path/to/LoViF/Test \
      --output ./freq_viz \
      --device cuda

Output per image:
  - {task}_{name}_input.png          — original input image
  - {task}_{name}_freq_response.png  — Laplacian response as heatmap
  - {task}_{name}_freq_vector.png    — frequency embedding vector (128×3 strip)
  - {task}_{name}_freq_response_fp.png — per-pixel % of max response map
"""

import os, sys, argparse
from PIL import Image
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from net.moce_ir import MoCEIR, FrequencyEmbedding

TASKS = ['Blur', 'Haze', 'Lowlight', 'Rain', 'Snow']


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


@torch.no_grad()
def extract(model: MoCEIR, image_tensor: torch.Tensor, device: str):
    """
    Run forward pass and extract frequency-related intermediates.

    Returns:
      freq_response: (B, C, H, W)  — Laplacian response per channel (before GELU)
      freq_emb:      (B, freq_dim)  — final frequency embedding vector
      feat_before:   (B, C, H, W)  — bottleneck features fed into FrequencyEmbedding
    """
    # Run model forward up to bottleneck (replicate internal logic)
    feats = model.patch_embed(image_tensor.to(device))  # (B, dim, H, W)
    enc_feats = []
    for block, downsample in model.enc:
        feats = block(feats)
        enc_feats.append(feats)
        feats = downsample(feats)
    feats = model.latent(feats)  # bottleneck (B, C_bottleneck, H_b, W_b)

    # Now manually step through FrequencyEmbedding
    # high_conv: HighPassConv2d (depthwise Laplacian)
    raw_response = model.freq_embed.high_conv[0].conv(feats)  # (B, C, H, W) before GELU

    # After GELU and global average pooling
    gated = model.freq_embed.high_conv[1](raw_response)  # GELU
    pooled = gated.mean(dim=(-2, -1))  # (B, C)
    freq_emb = model.freq_embed.mlp(pooled)  # (B, freq_dim)

    return raw_response, freq_emb, feats


def save_frequency_viz(image_path, freq_response, freq_emb, task, fname_out, out_dir):
    """Save frequency visualization for a single image."""
    # Load original image
    img_orig = Image.open(image_path).convert('RGB')
    img_np = np.array(img_orig)

    # Frequency response: average across channels to get single heatmap
    # freq_response: (1, C, H, W) → take mean across C and first batch
    resp = freq_response[0]  # (C, H, W)
    resp_mag = resp.abs().mean(dim=0).cpu().numpy()  # (H, W) mean magnitude

    # Normalize to 0-255 for visualization
    resp_min, resp_max = resp_mag.min(), resp_mag.max()
    resp_norm = ((resp_mag - resp_min) / (resp_max - resp_min + 1e-8) * 255).astype(np.uint8)

    # Frequency embedding vector (freq_dim)
    emb = freq_emb[0].cpu().numpy()
    # Reshape into most-square rectangle for visualization
    n = emb.shape[0]
    best_h = 1
    for h in range(1, int(np.sqrt(n)) + 1):
        if n % h == 0:
            best_h = h
    strip_h, strip_w = best_h, n // best_h
    strip = emb.reshape(strip_h, strip_w)
    # Normalize strip
    s_min, s_max = strip.min(), strip.max()
    strip_norm = ((strip - s_min) / (s_max - s_min + 1e-8) * 255).astype(np.uint8)

    base = os.path.splitext(fname_out)[0]

    # ── Plot input + frequency response side by side ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_np)
    axes[0].set_title(f'{task} - {base}')
    axes[0].axis('off')

    im = axes[1].imshow(resp_norm, cmap='jet', aspect='auto')
    axes[1].set_title(f'Laplacian Response (mean |resp|)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{task}_{base}_freq_overview.png'), dpi=150)
    plt.close()

    # ── Per-pixel max response (which channel responds strongest) ──
    resp_max_ch = resp.abs().argmax(dim=0).cpu().numpy()  # (H, W) channel index
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(resp_max_ch, cmap='tab20', aspect='auto')
    ax.set_title('Dominant Frequency Channel (argmax |resp|)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Channel index')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{task}_{base}_freq_dominant_ch.png'), dpi=150)
    plt.close()

    # ── Frequency embedding vector strip ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    im = ax.imshow(strip_norm[np.newaxis, :], cmap='viridis', aspect='auto')
    ax.set_title(f'Frequency Embedding Vector ({emb.shape[0]}d)')
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, orientation='horizontal')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{task}_{base}_freq_vector.png'), dpi=150)
    plt.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='./freq_viz')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-per-task', type=int, default=10)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    os.makedirs(args.output, exist_ok=True)

    print(f"[*] Loading model...")
    model = load_model(args.weights, device)
    print(f"[*] Device: {device}")

    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    all_freq_embs = {t: [] for t in TASKS}

    for task in TASKS:
        lq_dir = os.path.join(args.input, task, 'LQ')
        if not os.path.isdir(lq_dir):
            print(f"  [!] No LQ dir: {lq_dir}")
            continue

        files = sorted([f for f in os.listdir(lq_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])[:args.max_per_task]

        print(f"\n  [{task}] {len(files)} images...")
        for fname in files:
            img_path = os.path.join(lq_dir, fname)
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            pw, ph = (16 - w % 16) % 16, (16 - h % 16) % 16
            tensor = to_tensor(img).unsqueeze(0).to(device)
            if pw or ph:
                tensor = torch.nn.functional.pad(tensor, (0, pw, 0, ph))

            freq_response, freq_emb, bottleneck = extract(model, tensor, device)
            all_freq_embs[task].append(freq_emb[0].cpu().numpy())

            # Save visualizations
            save_frequency_viz(img_path, freq_response, freq_emb, task, fname, args.output)

        print(f"  ✓ Saved {len(files)} frequency vizzes to {args.output}")

    # ── Task-level frequency embedding comparison ──
    print("\n" + "=" * 72)
    print("  Task-Level Frequency Embedding Similarity")
    print("  (cosine similarity between average freq_emb per task)")
    print("=" * 72)

    task_avg = {}
    for task in TASKS:
        if all_freq_embs[task]:
            task_avg[task] = np.mean(all_freq_embs[task], axis=0)

    # Pairwise cosine
    print(f"\n  {'Task':<12}", end="")
    for t2 in TASKS:
        print(f"  {t2:<10}", end="")
    print()
    print(f"  {'─' * 12}─" + "─" * 12 * len(TASKS))

    for t1 in TASKS:
        if t1 not in task_avg:
            continue
        print(f"  {t1:<12}", end="")
        for t2 in TASKS:
            if t2 not in task_avg:
                print(f"  {'─':>10}", end="")
                continue
            v1 = task_avg[t1]
            v2 = task_avg[t2]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            marker = '🟢' if cos > 0.98 else ('🟡' if cos > 0.95 else ('🟠' if cos > 0.9 else '🔴'))
            print(f"  {marker}{cos:.4f}", end="")
        print()

    print(f"\n  🟢 >0.98 = very similar (bad for routing)")
    print(f"  🔴 <0.9  = distinct (good for routing)")
    print(f"\n[*] Output: {args.output}")


if __name__ == '__main__':
    main()
