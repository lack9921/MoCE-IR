"""从 checkpoint 直接统计参数数量和 cls_dim 相关权重"""
import torch, sys

ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)

if 'state_dict' in ckpt:
    sd = {k.replace('net.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('net.')}
else:
    sd = ckpt.get('model_state_dict', ckpt)

total = sum(p.numel() for p in sd.values())
print(f"Checkpoint: {sys.argv[1]}")
print(f"Net keys: {len(sd)}")
print(f"Total params: {total:,} ({total/1e6:.2f}M)")
print()

# cls_dim 相关的 key
cls_keys = [k for k in sd if 'task_proj' in k or 'task_cls' in k]
print(f"cls_dim 相关 keys ({len(cls_keys)}):")
for k in sorted(cls_keys):
    print(f"  {k:<45} {list(sd[k].shape)} = {sd[k].numel():,} params")
if cls_keys:
    cls_params = sum(sd[k].numel() for k in cls_keys)
    print(f"  cls_dim 总参数: {cls_params:,}")

# freq_gate 权重（展示 cls_dim 对路由的影响）
fg_keys = [k for k in sorted(sd) if 'freq_gate.weight' in k]
if fg_keys:
    print(f"\nfreq_gate.weight 形状:")
    for k in fg_keys[:3]:
        print(f"  {k:<55} {list(sd[k].shape)}")
    if len(fg_keys) > 3:
        print(f"  ... 共 {len(fg_keys)} 层")
