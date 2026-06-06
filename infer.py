"""
MoCE-IR inference on a flat folder of LQ images.
LQ = Low-Quality (not Low-Resolution) — image restoration task.
Works on both main and feat/classifier-routing branches.

Usage:
  python infer.py --input_dir ./folder of --weights ./checkpoints/xxx/last.ckpt
  python infer.py --input_dir ./folder of --weights ./model.pth --output_dir ./out --device cuda:1
"""

import os
import sys
import glob
import argparse
from PIL import Image

import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from net.moce_ir import MoCEIR


def load_model(ckpt_path: str, device: str = 'cuda'):
    raw = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' in raw:
        sd = {k.replace('net.', ''): v for k, v in raw['state_dict'].items() if k.startswith('net.')}
    elif 'model_state_dict' in raw:
        sd = raw['model_state_dict']
    else:
        sd = raw

    # Detect if this is a classifier checkpoint
    has_cls = any('classifier' in k for k in sd)

    model = MoCEIR(
        dim=48, num_blocks=[4, 6, 6, 8], num_dec_blocks=[2, 4, 4],
        levels=4, heads=[1, 2, 4, 8], num_refinement_blocks=4,
        topk=1, num_experts=4, rank=2, with_complexity=True,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'  Classifier: {"✅" if has_cls else "❌"}')
    if has_cls and len(unexpected) > 0:
        print(f'  ⚠️  Classifier keys not loaded — switch to feat/classifier-routing')
    if len(missing) > 10:
        print(f'  ⚠️  {len(missing)} missing keys — wrong checkpoint?')

    return model.to(device).eval()


@torch.no_grad()
def infer_folder(model, input_dir: str, output_dir: str, device: str = 'cuda'):
    os.makedirs(output_dir, exist_ok=True)
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    paths = sorted(p for p in glob.glob(os.path.join(input_dir, '*'))
                   if os.path.isfile(p) and p.lower().endswith(exts))

    if not paths:
        print('  No images found.')
        return

    print(f'  {len(paths)} images')
    for path in tqdm(paths, desc='Infer'):
        img = Image.open(path).convert('RGB')
        w, h = img.size
        pw, ph = (16 - w % 16) % 16, (16 - h % 16) % 16
        pimg = Image.new('RGB', (w + pw, h + ph), (0, 0, 0))
        pimg.paste(img, (0, 0))

        tensor = transforms.ToTensor()(pimg).unsqueeze(0).to(device)
        out = model(tensor).clamp(0, 1)[:, :, :h, :w]
        out = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        Image.fromarray(out).save(os.path.join(output_dir, os.path.basename(path)))


def main():
    p = argparse.ArgumentParser(description='MoCE-IR inference')
    p.add_argument('--input_dir', required=True, help='Folder of LQ images (flat)')
    p.add_argument('--weights', required=True, help='Checkpoint path (.ckpt / .pth)')
    p.add_argument('--output_dir', default=None, help='Output folder (default: ./restored)')
    p.add_argument('--device', default='cuda', help='Device')
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    out_dir = args.output_dir or './restored'

    print(f'Input:  {args.input_dir}')
    print(f'Weights: {args.weights}')
    print(f'Output: {out_dir}')
    print(f'Device: {device}')
    model = load_model(args.weights, device)
    infer_folder(model, args.input_dir, out_dir, device)
    print(f'✅ Done → {out_dir}')


if __name__ == '__main__':
    main()
