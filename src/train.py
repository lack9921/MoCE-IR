from typing import List

import os
import pathlib
import numpy as np

from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from net.moce_ir import MoCEIR

from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11, LoViFDataset, LoViFValDataset
from utils.loss_utils import FFTLoss

# Track names for per-track logging
LOVIF_TRACK_NAMES = ['blur', 'haze', 'lowlight', 'rain', 'snow']


def _to_y_channel(img_np):
    """RGB [0,1] → Y luminance channel [0,1] (BT.601)."""
    return np.dot(img_np[..., :3], [0.299, 0.587, 0.114])


class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.balance_loss_weight = opt.balance_loss_weight

        self.net = MoCEIR(
            dim=opt.dim, 
            num_blocks=opt.num_blocks, 
            num_dec_blocks=opt.num_dec_blocks, 
            levels=len(opt.num_blocks),
            heads=opt.heads, 
            num_refinement_blocks=opt.num_refinement_blocks, 
            topk=opt.topk, 
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity, 
            depth_type=opt.depth_type, 
            stage_depth=opt.stage_depth, 
            rank_type=opt.rank_type, 
            complexity_scale=opt.complexity_scale,)

        self.calc_lpips = None
        if opt.lovif:
            self.calc_lpips = LearnedPerceptualImagePatchSimilarity(
                net_type='vgg', normalize=True).cuda()

        if opt.loss_type == "fft":
            self.loss_fn = nn.L1Loss()
            self.aux_fn = FFTLoss(loss_weight=self.opt.fft_loss_weight)
        else:
            self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch, de_id)
        balance_loss = self.net.total_loss

        if self.opt.loss_type == "fft":
            loss = self.loss_fn(restored, clean_patch)
            aux_loss = self.aux_fn(restored, clean_patch)
            loss += aux_loss
        else:
            loss = self.loss_fn(restored, clean_patch)

        loss += self.balance_loss_weight * balance_loss
        self.log("Train_Loss", loss, sync_dist=True)
        self.log("Balance", balance_loss, sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR Schedule", lr, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch, de_id)

        # L1 loss
        loss = self.loss_fn(restored, clean_patch)
        self.log("val_loss", loss, sync_dist=True, batch_size=1)

        # Per-track PSNR / SSIM (batch_size=1 in val)
        for i in range(len(de_id)):
            restored_np = restored[i].detach().cpu().numpy().transpose(1, 2, 0)
            clean_np = clean_patch[i].detach().cpu().numpy().transpose(1, 2, 0)

            # Y channel (BT.601) — matches LoViF competition metric
            clean_y = _to_y_channel(clean_np)
            restored_y = _to_y_channel(restored_np)

            psnr_val = peak_signal_noise_ratio(clean_y, restored_y, data_range=1)
            ssim_val = structural_similarity(
                clean_y, restored_y, data_range=1)

            track = LOVIF_TRACK_NAMES[de_id[i]]
            self.log(f"val_psnr/{track}", psnr_val, sync_dist=True, batch_size=1)
            self.log(f"val_ssim/{track}", ssim_val, sync_dist=True, batch_size=1)

        # LPIPS on RGB (batch=1, single sample)
        if self.calc_lpips is not None:
            clean_rgb = clean_patch[:1]  # shape (1, C, H, W)
            restored_rgb = restored[:1].clamp(0, 1)  # clamp for LPIPS
            lpips_val = self.calc_lpips(restored_rgb, clean_rgb).item()
            self.log("val_lpips", lpips_val, sync_dist=True, batch_size=1)

        return loss

    def val_dataloader(self):
        valset = LoViFValDataset(self.opt)
        return DataLoader(
            valset, batch_size=1, shuffle=False,
            num_workers=2, pin_memory=True)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer, warmup_epochs=1, max_epochs=self.opt.epochs)
        return [optimizer], [scheduler]


def main(opt):
    print("Options")
    print(opt)
    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    log_dir = os.path.join("logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    if opt.wblogger:
        name = opt.model + "_" + time_stamp
        logger = WandbLogger(name=name, save_dir=log_dir, config=opt)
    else:
        logger = TensorBoardLogger(save_dir=log_dir)

    # Create model
    if opt.fine_tune_from:
        model = PLTrainModel.load_from_checkpoint(
            os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), opt=opt)
    else:
        model = PLTrainModel(opt)

    print(model)
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, every_n_epochs=5, save_top_k=-1, save_last=True)

    # Create datasets and dataloaders
    if opt.lovif:
        trainset = LoViFDataset(opt)
    elif "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        trainset = CDD11(opt, split="train", subset=subset)
    else:
        trainset = AIOTrainDataset(opt)

    trainloader = DataLoader(
        trainset, batch_size=opt.batch_size, pin_memory=True,
        shuffle=True, drop_last=True, num_workers=opt.num_workers)

    # Validation dataloader (only for --lovif mode with val data)
    valloader = None
    if opt.lovif:
        try:
            valset = LoViFValDataset(opt)
            if len(valset) > 0:
                valloader = DataLoader(
                    valset, batch_size=1, shuffle=False,
                    num_workers=2, pin_memory=True)
                print(f"[Validation] {len(valset)} samples")
        except Exception as e:
            print(f"[WARN] Could not create validation dataset: {e}")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=opt.accum_grad,
        deterministic=True,
        check_val_every_n_epoch=5)

    # Optionally resume from a checkpoint
    if opt.resume_from:
        ckpt_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
    else:
        ckpt_path = None

    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
        ckpt_path=ckpt_path)


if __name__ == '__main__':
    train_opt = train_options()
    main(train_opt)
