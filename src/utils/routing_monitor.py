"""
routing_monitor.py — Training-time routing diagnostics.

Lightning Callback that hooks into the MoCEIR model during training
to collect and log routing metrics to TensorBoard.

Usage in train.py:
    from utils.routing_monitor import RoutingMonitor
    trainer = pl.Trainer(callbacks=[..., RoutingMonitor()], ...)
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from lightning.pytorch.callbacks import Callback

from net.moce_ir import AdapterLayer, RoutingFunction


class RoutingMonitor(Callback):
    """Monitors MoE routing health during training.

    Logs to TensorBoard:
      - routing_entropy/{layer}        — per-layer routing entropy
      - expert_occupancy/{layer}/E{i}  — per-expert gate share (%)
      - cls_accuracy/val               — task classification accuracy
      - grad/freq_gate                 — freq_gate gradient norm
      - expert_similarity/{layer}      — pairwise cosine similarity between experts
    """

    def __init__(self):
        super().__init__()
        self._val_gates = defaultdict(list)  # {layer_name: [gates_tensor, ...]}
        self._val_deids = []
        self._adapter_names = []
        self._orig_rf_forward = None
        self._hooks_installed = False

    def _install_hooks(self, pl_module):
        """Monkey-patch RoutingFunction.forward on all AdapterLayers to capture gates."""
        if self._hooks_installed:
            return
        net = pl_module.net
        self._adapter_names = list(self._find_adapters(net))

        self._orig_rf_forward = RoutingFunction.forward

        def patched_rf_forward(self_rf, x, freq_emb):
            # Call original
            gates, top_k_indices, top_k_values, aux_loss = self_rf._original_forward(x, freq_emb)
            # Find which layer this RoutingFunction belongs to
            # (quick lookup: store name in the routing function object)
            layer_name = getattr(self_rf, '_monitor_name', None)
            if layer_name is not None:
                # Store gates for validation collection
                if not hasattr(net, '_captured_val_gates'):
                    net._captured_val_gates = {}
                # Only keep the gates from the latest batch
                net._captured_val_gates[layer_name] = gates.detach().cpu()
            return gates, top_k_indices, top_k_values, aux_loss

        # Store original and tag each RoutingFunction
        for name, adapter in self._adapter_names:
            rf = adapter.routing
            rf._original_forward = RoutingFunction.forward.__get__(rf, RoutingFunction)
            rf._monitor_name = name
        RoutingFunction.forward = patched_rf_forward

        self._hooks_installed = True

    def _find_adapters(self, module, prefix=''):
        for name, child in module._modules.items():
            if child is None:
                continue
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, AdapterLayer):
                yield full, child
            else:
                yield from self._find_adapters(child, full)

    def _uninstall_hooks(self):
        if self._hooks_installed and self._orig_rf_forward is not None:
            RoutingFunction.forward = self._orig_rf_forward
            for name, adapter in self._adapter_names:
                rf = adapter.routing
                if hasattr(rf, '_original_forward'):
                    delattr(rf, '_original_forward')
                if hasattr(rf, '_monitor_name'):
                    delattr(rf, '_monitor_name')
            self._hooks_installed = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def on_fit_start(self, trainer, pl_module):
        """Install hooks before training begins."""
        self._install_hooks(pl_module)

    def on_fit_end(self, trainer, pl_module):
        """Restore original forward."""
        self._uninstall_hooks()

    # ── Step-level ────────────────────────────────────────────────────

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect gates and task labels for this validation batch."""
        # Batch format from LoViFValDataset: ([clean_name, de_id], lq, gt)
        meta, lq, gt = batch
        _, de_id = meta
        self._val_deids.append(de_id.detach().cpu())

        if hasattr(pl_module.net, '_captured_val_gates'):
            for name, gates in pl_module.net._captured_val_gates.items():
                self._val_gates[name].append(gates)

    def on_after_backward(self, trainer, pl_module):
        """Compute freq_gate gradient norm."""
        total_sq = 0.0
        for n, p in pl_module.net.named_parameters():
            if 'freq_gate' in n and p.grad is not None:
                total_sq += p.grad.norm().item() ** 2
        if total_sq > 0:
            pl_module.log('grad/freq_gate', total_sq ** 0.5,
                          prog_bar=False, sync_dist=True)

    # ── Epoch-level ───────────────────────────────────────────────────

    def on_validation_epoch_end(self, trainer, pl_module):
        """Aggregate and log all collected metrics."""
        if not self._val_gates and not self._val_deids:
            return

        de_ids = torch.cat(self._val_deids) if self._val_deids else torch.tensor([])

        # ── Process collected gates per layer ──
        for layer_name, gate_list in self._val_gates.items():
            if not gate_list:
                continue
            all_gates = torch.cat(gate_list, dim=0)  # (N, E)
            N, E = all_gates.shape

            if N == 0:
                continue

            # Average gate distribution
            avg_gates = all_gates.mean(dim=0)
            p = avg_gates / (avg_gates.sum() + 1e-8)

            # ① Routing Entropy
            entropy = -(p * torch.log(p + 1e-8)).sum().item()
            tag = f"routing_entropy/{layer_name.replace('.', '_')}"
            pl_module.log(tag, entropy, prog_bar=False, sync_dist=True)

            # ② Expert Occupancy (% of total gate mass)
            for e in range(E):
                occ_tag = f"expert_occupancy/{layer_name.replace('.', '_')}/E{e}"
                pl_module.log(occ_tag, p[e].item() * 100, prog_bar=False, sync_dist=True)

        # ③ Task Classification Accuracy
        if hasattr(pl_module.net, 'cls_logits') and pl_module.net.cls_logits is not None and len(de_ids) > 0:
            preds = pl_module.net.cls_logits.argmax(dim=1).detach().cpu()
            acc = (preds == de_ids).float().mean().item()
            pl_module.log('cls_accuracy/val', acc, prog_bar=False, sync_dist=True)

        # ⑤ Expert Output Cosine Similarity (compute once per epoch)
        try:
            device = next(pl_module.net.parameters()).device
            x = torch.randn(8, 3, 128, 128, device=device)
            with torch.no_grad():
                _ = pl_module.net(x)  # warmup

            expert_outs = {}

            def make_hook(layer_name, expert_idx):
                def hook(mod, inp, out):
                    expert_outs[(layer_name, expert_idx)] = out.detach().cpu()
                return hook

            handles = []
            for name, adapter in self._adapter_names:
                for ei in range(4):
                    handles.append(
                        adapter.experts[ei].register_forward_hook(
                            make_hook(name, ei)))

            with torch.no_grad():
                _ = pl_module.net(x)

            for h in handles:
                h.remove()

            # Per-layer pairwise cosine similarity
            layer_sims = defaultdict(list)
            keys = list(expert_outs.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    name_i, ei = keys[i]
                    name_j, ej = keys[j]
                    if name_i == name_j:
                        vi = expert_outs[keys[i]].flatten(1)
                        vj = expert_outs[keys[j]].flatten(1)
                        cos = F.cosine_similarity(
                            vi.mean(0, keepdim=True),
                            vj.mean(0, keepdim=True)
                        ).item()
                        layer_sims[name_i].append(cos)

            for name, sims in layer_sims.items():
                mean_sim = float(np.mean(sims))
                tag = f"expert_similarity/{name.replace('.', '_')}"
                pl_module.log(tag, mean_sim, prog_bar=False, sync_dist=True)

        except Exception as e:
            print(f"  [RoutingMonitor] expert_similarity skipped: {e}")

        # Clear buffers
        self._val_gates.clear()
        self._val_deids.clear()
        if hasattr(pl_module.net, '_captured_val_gates'):
            pl_module.net._captured_val_gates = {}
