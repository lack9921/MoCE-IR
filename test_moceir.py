#!/usr/bin/env python3
"""
MoCE-IR — Full LoViF Track Test Script
Auto-tests Blur / Haze / Lowlight / Rain / Snow for PSNR (Y), SSIM (Y), LPIPS.

Usage:
  python test_moceir.py --weights ./checkpoints/last.ckpt \
      --input /datasets_host/LoViF\ 2026

Options:
  --weights PATH    Model checkpoint (.ckpt / .pth)        [required]
  --input  PATH     Dataset root (Blur/Haze/.../GT + LQ)   [required]
  --output DIR      Output directory                       [default: eval_results]
  --tasks LIST      Comma-separated, e.g. Blur,Haze        [default: all 5]
  --batch N         Batch size                             [default: 1]
  --crop N          Crop border pixels for metrics         [default: 0]
  --save-imgs N     Save SR comparison per track           [default: 3]
  --check-full      Test ALL images (default cap: 100/track)
  --dim N           Override base dim (auto-detect if omitted)
  --device DEV      Device string                          [default: cuda]
"""

import os, sys, json, argparse, time, glob
import torch
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from net.moce_ir import MoCEIR


# ── Constants ──────────────────────────────────────────────────────
TASKS = ['Blur', 'Haze', 'Lowlight', 'Rain', 'Snow']


# ── Helpers ────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str):
    """Load checkpoint, extract state_dict, detect model config."""
    raw = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' in raw:
        sd = {k.replace('net.', ''): v for k, v in raw['state_dict'].items() if k.startswith('net.')}
        print(f"  Lightning checkpoint — {len([k for k in raw['state_dict'] if k.startswith('net.')])} net keys")
    elif 'model_state_dict' in raw:
        sd = raw['model_state_dict']
        print("  model_state_dict checkpoint")
    else:
        sd = raw
        print("  Raw state_dict checkpoint")

    # Auto-detect config from state dict keys
    pe_key = [k for k in sd if 'patch_embed.proj.weight' in k]
    dim = sd[pe_key[0]].shape[0] if pe_key else 48
    print(f"  Detected dim={dim}")

    # ── Detect encoder levels ──
    enc_keys = [k for k in sd if k.startswith('enc.') and '.layers.' in k]
    max_level = 0
    for k in enc_keys:
        try:
            level = int(k.split('.')[1])
            max_level = max(max_level, level + 1)
        except (ValueError, IndexError):
            pass
    levels = max(4, max_level)

    # ── Detect heads from temperature/shared param shapes ──
    # MoCEIR heads = [h0, h1, ..., hn] where n = levels
    # Encoder uses h0..h(n-2), Latent uses h(n-1)
    # Decoder REVERSES heads: dec.0→h(n-1), dec.1→h(n-2), ...
    # Refinement uses reversed heads[0] = h(n-1)
    #
    # Temperature shape = [num_heads, 1, 1] — scan decoder keys for reliability
    def _scan_head(sd, key_substr, debug=False):
        for k in sd:
            if key_substr in k and ('temperature' in k or 'shared' in k):
                val = int(sd[k].shape[0])
                if debug:
                    print(f'      [debug] {k} → heads={val}')
                return val
        return None

    # Read heads from decoder (reliable — CrossAttention always has temperature)
    dec_nhs = {}
    for di in range(10):
        nh = _scan_head(sd, f'dec.{di}.', debug=True)
        if nh is not None:
            dec_nhs[di] = nh

    print(f'  [debug] Decoder heads found: {dec_nhs}')

    # Reconstruct original heads array from decoder mapping
    # dec[di] → reversed heads[di] → original heads[levels-1-di]
    h_vals = [6] * levels
    for di, nh in dec_nhs.items():
        orig = levels - 1 - di
        if 0 <= orig < levels:
            h_vals[orig] = nh
            print(f'  [debug] dec.{di} ({nh}h) → orig heads[{orig}] = {nh}')

    # Fill any still-default ones from encoder keys
    for i in range(levels):
        if h_vals[i] == 6:
            nh = _scan_head(sd, f'enc.{i}.', debug=True)
            if nh is not None:
                h_vals[i] = nh
                print(f'  [debug] enc.{i} ({nh}h) → heads[{i}] = {nh}')

    if h_vals[-1] == 6:  # try latent
        nh = _scan_head(sd, 'latent.', debug=True)
        if nh is not None:
            h_vals[-1] = nh
            print(f'  [debug] latent ({nh}h) → heads[{levels-1}] = {nh}')

    heads = [max(1, x) for x in h_vals]
    print(f'  [debug] Final heads: {heads}  (reversed for decoder: {heads[::-1]})')

    # ── Detect num_blocks from encoder groups ──
    num_blocks = []
    for i in range(levels - 1):
        bk = [k for k in sd if k.startswith(f'enc.{i}.0.layers.')]
        num_blocks.append(max((int(k.split('layers.')[1].split('.')[0]) for k in bk), default=4) + 1)
    # Latent block count
    latent_blocks = [k for k in sd if k.startswith('latent.layers.')]
    latent_n = max((int(k.split('layers.')[1].split('.')[0]) for k in latent_blocks), default=4) + 1
    num_blocks.append(latent_n)

    # ── Decoder blocks ──
    num_dec_blocks = []
    for i in range(levels - 1):
        dk = [k for k in sd if k.startswith(f'dec.{i}.2.layers.')]
        num_dec_blocks.append(max((int(k.split('layers.')[1].split('.')[0]) for k in dk), default=3) + 1)
    num_dec_blocks = num_dec_blocks[::-1] if num_dec_blocks else [1, 1, 1]

    # ── Refinement blocks ──
    ref_blocks = [k for k in sd if k.startswith('refinement.layers.')]
    num_refinement_blocks = max((int(k.split('layers.')[1].split('.')[0]) for k in ref_blocks), default=3) + 1

    # Detect classifier routing
    has_cls = any('classifier' in k for k in sd)

    # Detect depth_type and stage_depth from decoder routing keys
    has_routing = any('gate_network' in k or 'router' in k for k in sd)
    depth_type = 'lin'
    stage_depth = [1, 1, 1]
    rank_type = 'spread'
    rank = 2
    num_experts = 4

    # Detect num_experts from Linear layers in experts
    expert_w = [k for k in sd if 'expert' in k and 'weight' in k]
    if expert_w:
        # Rough heuristic
        exp_keys = set()
        for k in expert_w:
            parts = k.split('.')
            for i, p in enumerate(parts):
                if p == 'expert':
                    exp_keys.add(parts[i+1] if i+1 < len(parts) else '0')
        num_experts = len([e for e in exp_keys if e.isdigit()]) or 4

    return sd, {
        'dim': dim,
        'num_blocks': num_blocks,
        'num_dec_blocks': num_dec_blocks,
        'levels': levels,
        'heads': [max(1, h) for h in heads] if heads else [1, 1, 1, 1],
        'num_refinement_blocks': num_refinement_blocks,
        'topk': 2,
        'num_experts': num_experts,
        'rank': rank,
        'with_complexity': has_cls,
        'complexity_scale': 'max',
        'rank_type': rank_type,
        'depth_type': depth_type,
        'stage_depth': stage_depth,
        'has_cls': has_cls,
    }


def build_model(sd, cfg, device='cuda'):
    """Instantiate MoCEIR, load weights, return model."""
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
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Architecture: MoCEIR dim={cfg['dim']} levels={cfg['levels']} "
          f"blocks={cfg['num_blocks']} heads={cfg['heads']}")
    print(f"  Classifier: {'✅' if cfg['has_cls'] else '❌'} | "
          f"Routing: {'✅' if cfg['has_cls'] else '❌'}")

    if missing:
        print(f"  ⚠  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  ⚠  Unexpected keys: {len(unexpected)}")

    return model.to(device).eval()


def ycbcr2bgr(img):
    """ITU-R BT.601 YCbCr → BGR (for Y-channel extraction)."""
    return np.dot(img, [24.966, 128.553, 65.481]) + 16.0


def to_y_channel(img):
    """Convert RGB uint8 [0,255] to Y channel of YCbCr."""
    img_f = img.astype(np.float32) / 255.0
    # RGB → Y (BT.601)
    y = np.dot(img_f, [65.481, 128.553, 24.966]) + 16.0
    return y


def calc_psnr_y(sr, gt, crop=0):
    """PSNR on Y channel only."""
    if crop:
        sr = sr[crop:-crop, crop:-crop]
        gt = gt[crop:-crop, crop:-crop]
    sr_y = to_y_channel(sr)
    gt_y = to_y_channel(gt)
    return peak_signal_noise_ratio(gt_y, sr_y, data_range=255)


def calc_ssim_y(sr, gt, crop=0):
    """SSIM on Y channel only."""
    if crop:
        sr = sr[crop:-crop, crop:-crop]
        gt = gt[crop:-crop, crop:-crop]
    sr_y = to_y_channel(sr)
    gt_y = to_y_channel(gt)
    return structural_similarity(gt_y, sr_y, data_range=255, gaussian_weights=True)


def save_comparison(gt, lq, sr, path):
    """LQ | SR | GT side-by-side."""
    h = max(gt.shape[0], lq.shape[0], sr.shape[0])
    def pad(img):
        w = int(img.shape[1] * h / img.shape[0])
        return ImageOps.pad(Image.fromarray(img.astype('uint8')), (w, h))
    canvas = Image.new('RGB', (lq.shape[1] * 3 + 4, h), (255, 255, 255))
    for i, img in enumerate([lq, sr, gt]):
        canvas.paste(pad(img), (i * (lq.shape[1] + 2), 0))
    canvas.save(path)


@torch.no_grad()
def infer_one(model, lq_np, device='cuda'):
    """Single image inference, returns SR array [0,255]."""
    h, w = lq_np.shape[:2]
    pw = (16 - w % 16) % 16
    ph = (16 - h % 16) % 16
    lq_pad = np.pad(lq_np, ((0, ph), (0, pw), (0, 0)), mode='reflect')

    inp = torch.from_numpy(lq_pad).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    out = model(inp).clamp(0, 1)
    if isinstance(out, (list, tuple)):
        out = out[0]
    sr = out[:, :, :h, :w].squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr = (sr * 255).clip(0, 255).astype(np.uint8)
    return sr


def score(psnr, ssim, lpips):
    return psnr + 10.0 * ssim - 5.0 * lpips


def print_table(per_task, overall):
    sep = "+" + "-" * 16 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 14 + "+"
    print("\n" + "=" * 72)
    print("  MoCE-IR — LoViF Track Evaluation")
    print("=" * 72)
    print(sep)
    print(f"| {'Task':<14} | {'PSNR':>10} | {'SSIM':>10} | {'LPIPS':>10} | {'Score':>12} |")
    print(sep)
    for task in TASKS:
        if task in per_task:
            m = per_task[task]
            sc = score(m['psnr'], m['ssim'], m['lpips'])
            print(f"| {task:<14} | {m['psnr']:>10.2f} | {m['ssim']:>10.4f} | {m['lpips']:>10.4f} | {sc:>12.2f} |")
    print(sep)
    if overall:
        print(f"| {'OVERALL':<14} | {overall['psnr']:>10.2f} | {overall['ssim']:>10.4f} "
              f"| {overall['lpips']:>10.4f} | {overall['score']:>12.2f} |")
        print(sep)
    print()


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MoCE-IR LoViF Track Test')
    parser.add_argument('--weights', type=str, required=True, help='Checkpoint (.ckpt / .pth)')
    parser.add_argument('--input', type=str, required=True, help='Dataset root')
    parser.add_argument('--output', type=str, default='eval_results', help='Output directory')
    parser.add_argument('--tasks', type=str, default=None, help='Comma-separated tasks')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--crop', type=int, default=0, help='Crop border for PSNR/SSIM')
    parser.add_argument('--save-imgs', type=int, default=3, help='Save N comparison images per task')
    parser.add_argument('--check-full', action='store_true', help='Test ALL images')
    parser.add_argument('--dim', type=int, default=None, help='Override base dim')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--heads', type=str, default=None,
                        help='Override heads array: comma-separated, e.g. 1,2,4,8')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    tasks = TASKS if args.tasks is None else [t.strip() for t in args.tasks.split(',')]

    # ── Load model ──
    print(f"[*] Loading checkpoint: {args.weights}")
    sd, cfg = load_checkpoint(args.weights)
    if args.dim:
        cfg['dim'] = args.dim

    if args.heads:
        heads_override = [int(x.strip()) for x in args.heads.split(',')]
        print(f'  [--heads] Using manual heads: {heads_override}')
        cfg['heads'] = heads_override

    model = build_model(sd, cfg, device)

    # ── Setup LPIPS (once) ──
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    print(f"[*] LPIPS model ready")

    # ── Output ──
    os.makedirs(args.output, exist_ok=True)

    # ── Evaluate ──
    per_task = {}
    all_ps, all_ss, all_ls = [], [], []

    for task in tasks:
        gt_dir = os.path.join(args.input, task, 'GT')
        lq_dir = os.path.join(args.input, task, 'LQ')

        if not os.path.isdir(gt_dir):
            print(f"[!] Skipping {task}: GT not found at {gt_dir}")
            continue
        if not os.path.isdir(lq_dir):
            print(f"[!] Skipping {task}: LQ not found at {lq_dir}")
            continue

        image_files = sorted([
            f for f in os.listdir(gt_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ])
        n_total = len(image_files)
        if not args.check_full:
            image_files = image_files[:100]

        print(f"\n[{task}] Testing {len(image_files)}/{n_total} images ...")
        task_out = os.path.join(args.output, task)
        os.makedirs(task_out, exist_ok=True)
        t0 = time.time()

        ps_list, ss_list, ls_list = [], [], []
        for idx, fname in enumerate(tqdm(image_files, desc=f'  {task}')):
            gt = np.array(Image.open(os.path.join(gt_dir, fname)).convert('RGB'))
            lq = np.array(Image.open(os.path.join(lq_dir, fname)).convert('RGB'))

            sr = infer_one(model, lq, device)

            # Save comparison
            if idx < args.save_imgs:
                save_comparison(gt, lq, sr, os.path.join(task_out, fname.rsplit('.', 1)[0] + '_cmp.png'))

            # PSNR (Y) & SSIM (Y)
            ps_list.append(calc_psnr_y(sr, gt, args.crop))
            ss_list.append(calc_ssim_y(sr, gt, args.crop))

            # LPIPS (RGB, [0,1])
            sr_t = torch.from_numpy(sr).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            gt_t = torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            ls_list.append(lpips_fn(sr_t, gt_t).item())

        per_task[task] = {
            'psnr': float(np.mean(ps_list)),
            'ssim': float(np.mean(ss_list)),
            'lpips': float(np.mean(ls_list)),
            'n_images': len(image_files),
            'time_s': round(time.time() - t0, 1),
        }
        all_ps.extend(ps_list)
        all_ss.extend(ss_list)
        all_ls.extend(ls_list)

        m = per_task[task]
        print(f"  [{task}] ✓ PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  "
              f"LPIPS={m['lpips']:.4f}  ({m['time_s']:.1f}s)")

    # ── Overall ──
    overall = None
    if all_ps:
        overall = {
            'psnr': float(np.mean(all_ps)),
            'ssim': float(np.mean(all_ss)),
            'lpips': float(np.mean(all_ls)),
            'score': score(np.mean(all_ps), np.mean(all_ss), np.mean(all_ls)),
        }

    print_table(per_task, overall)

    # ── Save results ──
    result = {
        'model': {
            'arch': 'MoCEIR',
            'params_m': round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            'checkpoint': args.weights,
            'config': {k: v for k, v in cfg.items() if k != 'state_dict'},
        },
        'config': {
            'dataset_root': args.input,
            'crop_border': args.crop,
            'batch_size': args.batch,
            'tasks': tasks,
            'check_full': args.check_full,
        },
        'per_task': per_task,
        'overall': overall,
    }
    result_path = os.path.join(args.output, 'test_moceir_results.json')
    json.dump(result, open(result_path, 'w'), indent=2)
    print(f"[✓] Results: {result_path}")
    if overall:
        print(f"[✓] SUMMARY: Score={overall['score']:.2f}  "
              f"PSNR={overall['psnr']:.2f}  SSIM={overall['ssim']:.4f}  "
              f"LPIPS={overall['lpips']:.4f}")
    print("[✓] Done.")


if __name__ == '__main__':
    main()
