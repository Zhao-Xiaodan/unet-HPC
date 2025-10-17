import os, json, math, random
from datetime import datetime
from pathlib import Path
from PIL import Image
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from twodecoder_skip_64 import ConsensusAttnUNetStudent, TwoSelfAttnFuse

import copy

class ConsensusAttnUNetTeacher(nn.Module):
    def __init__(self, student_model):
        super().__init__()
        self.init_conv_img = copy.deepcopy(student_model.init_conv_img)
        self.blocks_img    = copy.deepcopy(student_model.blocks_img)

        attn_dim_64 = self.blocks_img[-1].dense_block.out_channels   
        self.attn_gt = TwoSelfAttnFuse(dim=attn_dim_64) 
        
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, img, gt_mask):
        # stem
        img_x = self.init_conv_img(img)
        gt_x  = self.init_conv_img(gt_mask)

        last_d_img = None
        last_d_gt  = None
        for block in self.blocks_img:
            d_img, t_img = block(img_x)   
            d_gt,  t_gt  = block(gt_x)
            img_x, gt_x  = t_img, t_gt
            last_d_img, last_d_gt = d_img, d_gt  

        x_feat = last_d_img   # [B, C64, 64, 64]
        z_feat = last_d_gt    # [B, C64, 64, 64]

        assert x_feat.size(1) == self.attn_gt.qkv_x.in_channels, \
            f"{x_feat.size(1)} != {self.attn_gt.qkv_x.in_channels}"
        assert z_feat.size(1) == self.attn_gt.qk_z.in_channels, \
            f"{z_feat.size(1)} != {self.attn_gt.qk_z.in_channels}"

        fused_gt_target = self.attn_gt(x_feat, z_feat)  # 64×64
        return fused_gt_target
    
@torch.no_grad()
def update_teacher_weights(teacher_model: nn.Module, student_model: nn.Module, alpha: float = 0.999):
    t_sd = teacher_model.state_dict()
    s_sd = student_model.state_dict()

    for k, tv in t_sd.items():
        if k not in s_sd:
            continue
        sv = s_sd[k]
        if tv.shape != sv.shape:
            continue

        sv = sv.to(device=tv.device)

        if torch.is_floating_point(tv):
            sv = sv.to(dtype=tv.dtype)
            tv.mul_(alpha).add_(sv, alpha=1.0 - alpha)
        else:
            tv.copy_(sv)

# checkpoint
def save_checkpoint(student_model, teacher_model, optimizer, scaler, epoch, history, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    p = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    state = {
        'epoch': epoch,
        'student_state_dict': student_model.state_dict(),
        'teacher_state_dict': teacher_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(), # This line now works
        'history': history
    }
    torch.save(state, p)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved checkpoint: {p}")

def load_checkpoint(student_model, teacher_model, optimizer, checkpoint_path, device, scaler=None):
    
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] No checkpoint found at {checkpoint_path}. Starting fresh.")
        history = {'train_total': [], 'val_total': [], 'train_bottleneck': []}
       
        for key in ['train_main','train_bg_adapt','train_tv','train_dice',
                    'val_main','val_bg_adapt','val_tv','val_dice']:
            history.setdefault(key, [])
        return 0, history

    print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    student_model.load_state_dict(ckpt['student_state_dict'], strict=False)
    teacher_model.load_state_dict(ckpt['teacher_state_dict'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        
    optimizer.zero_grad(set_to_none=True) 
    
    start_epoch = int(ckpt.get('epoch', -1)) + 1
    history = ckpt.get('history', {})
    history = _ensure_history_keys(history)

    print(f"[INFO] Resumed successfully from epoch {start_epoch-1}.")
    return start_epoch, history

        
def _percentile_norm(arr: np.ndarray):
    lo, hi = np.percentile(arr, [0.5, 99.5]).astype(np.float32)
    if hi <= lo: hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr.astype(np.float32)

def resize_chw_tensor(x: torch.Tensor, size_hw):
    # x: [C,H,W] or [B,C,H,W]
    if x.dim() == 3:
        x = x.unsqueeze(0)
        out = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return out.squeeze(0)
    else:
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)


def add_bg_artifacts(img01: np.ndarray, max_grad=0.15, p=0.5):
    
    if np.random.rand() > p:
        return img01
    H, W = img01.shape
    x = np.linspace(-1, 1, W)[None, :]
    y = np.linspace(-1, 1, H)[:, None]
    grad = (np.random.uniform(-max_grad, max_grad) * x +
            np.random.uniform(-max_grad, max_grad) * y)
    band = 0.03 * np.sin(2 * np.pi * (np.arange(W) / np.random.uniform(40, 120)))[None, :]
    noise = np.random.normal(0, 0.02, (H, W))
    out = img01 + grad + band + noise
    return np.clip(out, 0, 1).astype(np.float32)

def box_blur_same(arr: np.ndarray, k: int) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    if k <= 1:
        return arr.copy()
    k = int(k) | 1
    pad = k // 2
    
    kernel = np.ones((k, k), np.float32) / (k * k)
    from scipy.signal import convolve2d
    return convolve2d(arr, kernel, mode='same', boundary='symm').astype(np.float32)

def add_bg_artifacts_with_fade(img01: np.ndarray,
                               p: float = 1.0,
                               max_grad: float = 0.01,
                               band_amp: float = 0.012,
                               band_period: tuple = (90, 160),
                               noise_sigma: float = 0.008,
                               bead_fade_strength: float = 0.9,
                               bead_thresh: float = 0.40,
                               use_local_bg: bool = True,
                               bg_kernel: int = 21) -> np.ndarray:
    if np.random.rand() > p:
        return img01.astype(np.float32, copy=True)

    H, W = img01.shape
    x = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, H, dtype=np.float32)[:, None]

    grad  = (np.random.uniform(-max_grad, max_grad) * x +
             np.random.uniform(-max_grad, max_grad) * y)
    band  = band_amp * np.sin(2*np.pi * (np.arange(W, dtype=np.float32) /
                                         np.random.uniform(*band_period)))[None, :]
    noise = np.random.normal(0, noise_sigma, (H, W)).astype(np.float32)

    out = np.clip(img01.astype(np.float32) + grad + band + noise, 0, 1)

    bead_mask = (img01 < bead_thresh).astype(np.float32)
    darkness  = np.clip((bead_thresh - img01) / max(bead_thresh, 1e-6), 0, 1)
    s_pix     = bead_fade_strength * darkness * bead_mask

    bg = box_blur_same(out, bg_kernel) if use_local_bg else np.ones_like(out)
    out = (1.0 - s_pix) * out + s_pix * bg
    return np.clip(out, 0, 1).astype(np.float32)

def tversky_loss(prob: torch.Tensor, target: torch.Tensor, alpha=0.3, beta=0.7, smooth=1e-6):
    # bg=1，fg=0
    prob = prob.clamp(1e-6, 1-1e-6) 
    TP = (prob * target).sum(dim=(1,2,3))
    FP = (prob * (1 - target)).sum(dim=(1,2,3))
    FN = ((1 - prob) * target).sum(dim=(1,2,3))
    T  = (TP + smooth) / (TP + alpha*FN + beta*FP + smooth)
    return 1 - T.mean()

class AdaptiveBGDiceLoss(nn.Module):

    def __init__(self,
                 loss_type: str = "focal",
                 focal_alpha: float = 0.6,      
                 focal_gamma: float = 2.0,
                 tv_weight: float = 1,      
                 dice_weight: float = 0.4,      
                 illum_kernel: int = 193,       
                 delta: float = 0.05,        
                 bg_adapt_weight: float = 0.05, 
                 emphasize: str = "fg",   
                 b_norm_mode: str = "none",  
                 dataset_p5: float = 0.0,   
                 dataset_p95: float = 1.0,   
                 tv_flatness_beta: float = 10.0 
                 ):
        super().__init__()
        self.delta = float(delta)
        self.illum_kernel = int(illum_kernel)
        self.tv_weight = float(tv_weight)
        self.dice_weight = float(dice_weight)
        self.bg_adapt_weight = float(bg_adapt_weight)
        self.emphasize = emphasize
        self.b_norm_mode = b_norm_mode
        self.dataset_p5 = float(dataset_p5)
        self.dataset_p95 = float(dataset_p95)
        self.tv_flatness_beta = float(tv_flatness_beta)

        if loss_type.lower() == "focal":
            self.main_loss_impl = FocalBCEWithLogitsLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="none")
            self.use_focal = True
        else:
            self.main_loss_impl = nn.BCEWithLogitsLoss(reduction="none")
            self.use_focal = False

    def _norm_b(self, b: torch.Tensor) -> torch.Tensor:
        if self.b_norm_mode == "none":
            
            return b.clamp(0, 1)
        elif self.b_norm_mode == "per_image":
            bf = b.flatten(1)
            p5  = torch.quantile(bf, 0.05, dim=1, keepdim=True).view(-1,1,1,1)
            p95 = torch.quantile(bf, 0.95, dim=1, keepdim=True).view(-1,1,1,1)
            return ((b - p5) / (p95 - p5 + 1e-6)).clamp(0, 1)
        elif self.b_norm_mode == "dataset":
            return ((b - self.dataset_p5) / (self.dataset_p95 - self.dataset_p5 + 1e-6)).clamp(0, 1)
        else:
            raise ValueError(f"Unknown b_norm_mode: {self.b_norm_mode}")

    def forward(self,
                logits: torch.Tensor,
                target_bg: torch.Tensor,
                images: torch.Tensor,
                ignore: torch.Tensor = None):

        dev, dtype = logits.device, logits.dtype
        assert logits.dim() == 4 and logits.size(1) == 1, "logits [B,1,H,W]"

        target_bg = target_bg.to(dev, dtype=dtype, non_blocking=True)
        images    = images.to(dev, dtype=dtype, non_blocking=True)
        ig = torch.ones_like(target_bg, device=dev, dtype=dtype) if ignore is None \
             else ignore.to(dev, dtype=target_bg.dtype, non_blocking=True)

        y_bg = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)   # bg
        y_fg = 1.0 - y_bg
        t_bg = target_bg
        t_fg = 1.0 - target_bg

        # ---- main ----
        if self.emphasize == "fg":
            per = self.main_loss_impl(-logits, t_fg) 
        else:
            per = self.main_loss_impl(logits, t_bg) 
        main = per.mean() if per.dim() > 0 else per

        # ---- bg ----
        b_raw = box_blur(images.detach(), k=self.illum_kernel) 
        b = self._norm_b(b_raw)                                 
        t = (b + self.delta).clamp(max=1.0)                   
        w = (1.0 - b)                                        
        mask_bg = t_bg * ig
        over = F.relu(y_bg - t) * w * mask_bg
        denom = (w * mask_bg).sum().clamp_min(1.0)
        L_bg_adapt = over.sum() / denom

        # ---- TV ----
        if self.tv_weight > 0:
            bx = (b[:, :, 1:, :] - b[:, :, :-1, :]).abs()
            by = (b[:, :, :, 1:] - b[:, :, :, :-1]).abs()
            flat_wx = torch.exp(-self.tv_flatness_beta * bx)
            flat_wy = torch.exp(-self.tv_flatness_beta * by)
            bg_x = (t_bg[:, :, 1:, :] * t_bg[:, :, :-1, :] * ig[:, :, 1:, :] * ig[:, :, :-1, :])
            bg_y = (t_bg[:, :, :, 1:] * t_bg[:, :, :, :-1] * ig[:, :, :, 1:] * ig[:, :, :, :-1])
            yx = (y_bg[:, :, 1:, :] - y_bg[:, :, :-1, :]).abs() * flat_wx * bg_x
            yy = (y_bg[:, :, :, 1:] - y_bg[:, :, :, :-1]).abs() * flat_wy * bg_y
            denom_tv = (flat_wx * bg_x).sum() + (flat_wy * bg_y).sum()
            L_tv = self.tv_weight * (yx.sum() + yy.sum()) / (denom_tv + 1e-6)
        else:
            L_tv = logits.new_tensor(0.0)

        # ---- Tversky ----
        if self.emphasize == "fg":
            p, tmask = y_fg * ig, t_fg * ig
        else:
            p, tmask = y_bg * ig, t_bg * ig
        L_tversky = tversky_loss(p, tmask, alpha=0.7, beta=0.3)

        total = main + self.bg_adapt_weight * L_bg_adapt + L_tv + self.dice_weight * L_tversky
        logs = {
            "main": float(main.detach()),
            "bg_adapt": float(L_bg_adapt.detach()),
            "tv": float(L_tv.detach()),
            "dice": float(L_tversky.detach()),
            "y_bg_mean": float(y_bg.mean().detach()),
            "b_mean": float(b.mean().detach()),
        }
        return total, logs
    
def _make_edge_ignore(images, ratio=0.08):
    B,_,H,W = images.shape
    m = int(ratio * min(H, W))
    ignore = images.new_ones((B,1,H,W))
    ignore[:,:, :m, :] = 0;  ignore[:,:, H-m:, :] = 0
    ignore[:,:,:, :m] = 0;   ignore[:,:,:, W-m:] = 0
    return ignore

class BeadsDataset(Dataset):
    def __init__(self, img_dir, mask_dir, resize_to=None, augment=False,no_art_ratio=0.4, old_art_ratio=0.3, new_art_ratio=0.3):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.resize_to = resize_to
        self.augment = bool(augment)
        self.no_art_ratio = no_art_ratio
        self.old_art_ratio = old_art_ratio
        self.new_art_ratio = new_art_ratio

        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
        def list_images(root: Path):
            files = []
            for p in sorted(root.iterdir()):
                if p.name.startswith("."):
                    continue
                if p.is_dir():
                    continue
                if p.suffix.lower() in exts:
                    files.append(p)
            return files

        img_files  = list_images(self.img_dir)
        mask_files = list_images(self.mask_dir)
        img_map  = {p.stem: p for p in img_files}
        mask_map = {p.stem: p for p in mask_files}
        common   = sorted(set(img_map.keys()) & set(mask_map.keys()))
        if not common:
            raise RuntimeError(f"No paired images/masks found under {self.img_dir} and {self.mask_dir}")

        self.pairs = [(img_map[k], mask_map[k]) for k in common]
        print(f"[dataset] paired samples = {len(self.pairs)} (from {self.img_dir} / {self.mask_dir})")

    def __len__(self):
        return len(self.pairs)

    def _load_gray01(self, p: Path):
        if not p.is_file():
            raise FileNotFoundError(f"Not a file: {p}")
        im = Image.open(str(p)).convert("L")
        arr = np.array(im, dtype=np.float32)
        arr = _percentile_norm(arr)
        return arr  # [H,W] float32 [0,1]

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img01  = self._load_gray01(img_path)
        mask01 = self._load_gray01(mask_path)
        tgt01  = 1.0 - mask01  

        if self.augment:
            choice = np.random.rand()
            if choice < self.no_art_ratio:
                pass  
            elif choice < self.no_art_ratio + self.old_art_ratio:
                img01 = add_bg_artifacts(
                    img01,
                    max_grad=np.random.uniform(0.05, 0.25),   
                    p=0.5)
            else:
                img01 = add_bg_artifacts_with_fade(
                    img01,
                    bead_fade_strength=np.random.uniform(0.7, 1.2),
                    bead_thresh=np.random.uniform(0.7, 1.0),
                    bg_kernel=np.random.choice([33, 49, 65, 81]),
                    max_grad=np.random.uniform(0.0, 0.05),
                    band_amp=np.random.uniform(0.001, 0.01),
                    noise_sigma=np.random.uniform(0.001, 0.01))

        image = torch.from_numpy(img01).unsqueeze(0)
        mask  = torch.from_numpy(mask01).unsqueeze(0)
        target= torch.from_numpy(tgt01).unsqueeze(0)

        if self.resize_to is not None:
            H, W = self.resize_to
            image  = resize_chw_tensor(image,  (H, W))
            mask   = resize_chw_tensor(mask,   (H, W))
            target = resize_chw_tensor(target, (H, W))
        return image.float(), mask.float(), target.float()

# Losses
def dice_loss(prob, target, smooth=1e-6):
    prob = prob.reshape(-1)
    target = target.reshape(-1)
    intersection = (prob * target).sum()
    dice_coeff = (2. * intersection + smooth) / (prob.sum() + target.sum() + smooth)
    return 1 - dice_coeff

def tv_on_background(prob, target_bg, lam=1e-3):
    if lam <= 0:
        return prob.new_tensor(0.0)
    w = target_bg
    dx = prob[:, :, 1:, :] - prob[:, :, :-1, :]
    dy = prob[:, :, :, 1:] - prob[:, :, :, :-1]
    w_dx = w[:, :, 1:, :] * w[:, :, :-1, :]
    w_dy = w[:, :, :, 1:] * w[:, :, :, :-1]
    num = (dx.abs() * w_dx).sum() + (dy.abs() * w_dy).sum()
    den = (w_dx.sum() + w_dy.sum()).clamp_min(1e-6)
    return lam * (num / den)

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        pt = p * target + (1 - p) * (1 - target)
        focal = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            at = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal = at * focal
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal

def _ensure_history_keys(history=None):
    if history is None:
        history = {}
    base = ['total', 'main', 'bg_adapt', 'tv', 'dice']
    for p in ('train_', 'val_'):
        for k in base:
            history.setdefault(p + k, [])
    history.setdefault('train_bottleneck', [])
    history.setdefault('train_total', [])
    history.setdefault('val_total', [])
    return history

class BottleneckLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_bottleneck, target_bottleneck):
        return self.mse_loss(predicted_bottleneck, target_bottleneck)

def box_blur(x: torch.Tensor, k: int = 63):
    k = int(k) | 1       
    pad = k // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"),
                        kernel_size=k, stride=1)

def soft_dice(prob: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
    inter = (prob * target).sum(dim=(1,2,3))
    denom = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return 1.0 - ((2.0 * inter + smooth) / (denom + smooth)).mean()

def make_boundary_ring(mask_fg01, r=2):
    m = (mask_fg01 > 0.5).float()
    dil = F.max_pool2d(m, kernel_size=2*r+1, stride=1, padding=r)
    ero = 1.0 - F.max_pool2d(1.0 - m, kernel_size=2*r+1, stride=1, padding=r)
    ring = (dil - ero).clamp(0, 1)  
    return ring

def att_mse_affine(student: torch.Tensor, teacher: torch.Tensor, eps: float = 1e-6):

    ms = student.mean(dim=(1,2,3), keepdim=True)
    ss = student.std(dim=(1,2,3), keepdim=True)
    mt = teacher.mean(dim=(1,2,3), keepdim=True)
    st = teacher.std(dim=(1,2,3), keepdim=True)

    student_aligned = (student - ms) / (ss + eps) * st + mt

    return F.mse_loss(student_aligned, teacher)

def train_one_epoch_distillation(student_model, teacher_model, dataloader, criterion, optimizer, device, scaler, accum_steps=1):
    student_model.train()
    teacher_model.eval()
    
    optimizer.zero_grad(set_to_none=True)
    ATTN_LOSS_WEIGHT = 10

    sums = {"total": 0.0, "main": 0.0, "bg_adapt": 0.0, "tv": 0.0, "dice": 0.0, "bottleneck": 0.0}
    n_batches = 0

    for step, (images, masks, targets) in enumerate(dataloader, start=1):
        
        ignore = _make_edge_ignore(images, ratio=0.0) 
        
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)  
        targets= targets.to(device, non_blocking=True) 

        with torch.no_grad():
            fused_gt_target = teacher_model(images, masks) 

        with autocast(enabled=True):
            
            final_output, fake_mask, fused_fake = student_model(images)
            
            att_S = fused_fake 
            logits = final_output 

            main_loss, logs = criterion(logits, targets, images, ignore=ignore)

            L_att = att_mse_affine(att_S, fused_gt_target)

            loss = main_loss + ATTN_LOSS_WEIGHT * L_att

        loss = loss / accum_steps
        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            update_teacher_weights(teacher_model, student_model, alpha=0.999) 

        n_batches += 1
        sums["total"]      += float(loss.detach()) * accum_steps
        sums["main"]       += logs["main"]
        sums["bg_adapt"]   += logs["bg_adapt"]
        sums["tv"]         += logs["tv"]
        sums["dice"]       += logs["dice"]
        sums["bottleneck"] += float(L_att.detach())

    for k in sums: sums[k] /= max(1, n_batches)
    return sums

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    sums = {"total": 0.0, "main": 0.0, "bg_adapt": 0.0, "tv": 0.0, "dice": 0.0}
    n_batches = 0
    for images, masks, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets= targets.to(device, non_blocking=True)
        with autocast(enabled=False):
            outputs, _, _ = model(images)  
            total_loss, logs = criterion(outputs, targets, images)

        n_batches += 1
        sums["total"]    += float(total_loss)
        sums["main"]     += logs["main"]
        sums["bg_adapt"] += logs["bg_adapt"]
        sums["tv"]       += logs["tv"]
        sums["dice"]     += logs["dice"]

    for k in sums: sums[k] /= max(1, n_batches)
    return sums


def main():
    # ---- paths & hparams ----
    IMG_DIR = '/home/jovyan/mbi-toast/dataset/train/images/'
    MASK_DIR = '/home/jovyan/mbi-toast/dataset/train/masks/'
    RESIZE = 512          
    BATCH_SIZE = 1
    ACCUM_STEPS = 8
    LEARNING_RATE = 5e-4
    EPOCHS = 400
    CHECKPOINT_DIR = './checkpoint_consensus_distill'
    CHECKPOINT_TO_RESUME = os.path.join(CHECKPOINT_DIR, 'None') 
    
    # ---- device / setup ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- model / opt / loss ----
    model_kwargs = {
        "in_channels": 1, "num_classes": 1, 
        "growth_rate": 32, "block_layers": [4, 4, 4, 4], "compression": 0.5 
    }

    student_model = ConsensusAttnUNetStudent(**model_kwargs).to(device)
    teacher_model = ConsensusAttnUNetTeacher(student_model=student_model).to(device)
        
    optimizer = torch.optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

    criterion = AdaptiveBGDiceLoss(
        loss_type="focal", focal_alpha=0.4, focal_gamma=2.0,
        tv_weight=1, dice_weight=0.8, illum_kernel=193, 
        delta=0.07, bg_adapt_weight=0.05, emphasize="fg", b_norm_mode="none"
    )
    
    scaler = GradScaler(enabled=torch.cuda.is_available()) 

    start_epoch = 0
    
    if CHECKPOINT_TO_RESUME:
        start_epoch, history = load_checkpoint(student_model, teacher_model, optimizer, CHECKPOINT_TO_RESUME, device, scaler=scaler)
        history = _ensure_history_keys(history)
    else:
        history = _ensure_history_keys(None)
        update_teacher_weights(teacher_model, student_model, alpha=0.0)

    # ---- data ----
    RESIZE = 512; AUGMENT_ARTIFACTS = True
    resize_to = (RESIZE, RESIZE)
    full_dataset = BeadsDataset(img_dir=IMG_DIR, mask_dir=MASK_DIR,
                                resize_to=resize_to, augment=True, 
                                no_art_ratio=0.4, old_art_ratio=0.3, new_art_ratio=0.3)
    
    total_size = len(full_dataset); train_size = int(0.8 * total_size); val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.augment = bool(AUGMENT_ARTIFACTS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    print(f"Training on {train_size} samples, validating on {val_size} samples.")

    # ---- loop ----
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n=== Starting epoch {epoch+1}/{EPOCHS} ===")
        
        train_sums = train_one_epoch_distillation(student_model, teacher_model, train_loader, criterion, optimizer, device, scaler, accum_steps=ACCUM_STEPS)
        val_sums   = validate(student_model, val_loader, criterion, device)

        LOG_KEYS = ['total', 'main', 'bg_adapt', 'tv', 'dice'] 

        for k in LOG_KEYS:
            
            history[f'train_{k}'].append(train_sums[k]) 

            history[f'val_{k}'].append(val_sums[k])

        history['train_bottleneck'].append(train_sums['bottleneck']) 

        print(f"[Train] total={train_sums['total']:.6f} | main={train_sums['main']:.6f} | bg={train_sums['bg_adapt']:.6f} | tv={train_sums['tv']:.6f} | dice={train_sums['dice']:.6f} | att={train_sums['bottleneck']:.6f}")
        print(f"[ Val ] total={val_sums['total']:.6f} | main={val_sums['main']:.6f} | bg={val_sums['bg_adapt']:.6f} | tv={val_sums['tv']:.6f} | dice={val_sums['dice']:.6f}")

        save_checkpoint(student_model, teacher_model, optimizer, scaler, epoch, history, CHECKPOINT_DIR)
        
        latest_path = os.path.join(CHECKPOINT_DIR, 'latest.pth')
        if os.path.exists(latest_path):
             os.remove(latest_path)
        os.link(os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'), latest_path)


    print("Training finished!")
    final_history_path = os.path.join(CHECKPOINT_DIR, 'training_history.json')
    with open(final_history_path, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()