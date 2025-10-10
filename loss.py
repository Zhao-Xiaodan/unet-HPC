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


def tversky_loss(prob: torch.Tensor, target: torch.Tensor, alpha=0.3, beta=0.7, smooth=1e-6):
    
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
        b = b.float()
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
        assert logits.dim() == 4 and logits.size(1) == 1, "logits should be [B,1,H,W]"

        target_bg = target_bg.to(dev, dtype=dtype, non_blocking=True)
        images    = images.to(dev, dtype=dtype, non_blocking=True)
        ig = torch.ones_like(target_bg, device=dev, dtype=dtype) if ignore is None \
             else ignore.to(dev, dtype=target_bg.dtype, non_blocking=True)

        y_bg = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)  
        y_fg = 1.0 - y_bg
        t_bg = target_bg
        t_fg = 1.0 - target_bg

        if self.emphasize == "fg":
            per = self.main_loss_impl(-logits, t_fg) 
        else:
            per = self.main_loss_impl(logits, t_bg) 
        main = per.mean() if per.dim() > 0 else per

        #----bg -----
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

        # ----  Tversky ----
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

        # Teacher model forward: teacher(img, gt_mask) -> fused_gt_target
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


##################################
criterion = AdaptiveBGDiceLoss(
        loss_type="focal", focal_alpha=0.7, focal_gamma=1.2,
        tv_weight=1, dice_weight=1.5, illum_kernel=40, 
        delta=0.05, bg_adapt_weight=0.01, emphasize="fg", b_norm_mode="per_image"
    )
#############################