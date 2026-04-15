#!/usr/bin/env python3

# Unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

# ---------------------------
# Building Blocks
# ---------------------------

class DWConv(nn.Module):
    """Depthwise separable: DW(kxk) + BN + ReLU + PW(1x1) + BN + ReLU"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(1, ch // r)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)

class DoubleConv(nn.Module):
    """Two conv blocks; optionally depthwise and SE."""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None,
                 use_depthwise: bool = True, use_se: bool = False):
        super().__init__()
        mid = mid_ch or out_ch
        def conv(ic, oc):
            return DWConv(ic, oc) if use_depthwise else nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True)
            )
        self.conv1 = conv(in_ch, mid)
        self.conv2 = conv(mid, out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.conv2(x); return self.se(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_depthwise=True, use_se=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, use_depthwise=use_depthwise, use_se=use_se)
        )
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    """Bilinear upsample + concat + DoubleConv (default)."""
    def __init__(self, in_ch: int, out_ch: int, bilinear=True,
                 use_depthwise=True, use_se=False):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2,
                                   use_depthwise=use_depthwise, use_se=use_se)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, use_depthwise=use_depthwise, use_se=use_se)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if size mismatch
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)

class ASPPLite(nn.Module):
    """Tiny ASPP for cheap context."""
    def __init__(self, in_ch: int, out_ch: int, rates=(1, 2, 3)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for r in rates
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * len(rates), in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.proj(torch.cat(feats, dim=1))

# ---------------------------
# U-Net
# ---------------------------

class UNet(nn.Module):
    """
    Lightweight U-Net for multi-class segmentation (logits out).
    """
    def __init__(self, n_channels=3, n_classes=4, bilinear=True,
                 base_channels: int = 32,
                 use_depthwise=True, use_se_decoder=True, use_aspp=False):
        super().__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        b = int(base_channels)

        self.inc   = DoubleConv(n_channels, b, use_depthwise=use_depthwise, use_se=False)
        self.down1 = Down(b, b * 2,  use_depthwise=use_depthwise, use_se=False)
        self.down2 = Down(b * 2, b * 4, use_depthwise=use_depthwise, use_se=False)
        self.down3 = Down(b * 4, b * 8, use_depthwise=use_depthwise, use_se=False)
        factor = 2 if bilinear else 1
        self.down4 = Down(b * 8, (b * 16) // factor, use_depthwise=use_depthwise, use_se=False)

        self.aspp = ASPPLite((b * 16) // factor, b * 4) if use_aspp else nn.Identity()

        self.up1 = Up(b * 16, (b * 8) // factor, bilinear, use_depthwise=use_depthwise, use_se=use_se_decoder)
        self.up2 = Up(b * 8, (b * 4) // factor, bilinear, use_depthwise=use_depthwise, use_se=use_se_decoder)
        self.up3 = Up(b * 4, (b * 2) // factor, bilinear, use_depthwise=use_depthwise, use_se=use_se_decoder)
        self.up4 = Up(b * 2, b,              bilinear, use_depthwise=use_depthwise, use_se=use_se_decoder)

        self.outc = OutConv(b, n_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x) -> Tuple[torch.Tensor, None]:
        x1 = self.inc(x);  x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x5 = self.aspp(x5)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, None

# ---------------------------
# Losses / Metrics
# ---------------------------

def one_hot(target_ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target_ids.long(), num_classes).permute(0, 3, 1, 2).float()

def dice_loss_multiclass(probs: torch.Tensor, target_1h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = (0, 2, 3)
    num = 2 * (probs * target_1h).sum(dims)
    den = (probs * probs).sum(dims) + (target_1h * target_1h).sum(dims) + eps
    return (1.0 - (num / den)).mean()


def focal_loss_multiclass(logits: torch.Tensor,
                          target_ids: torch.Tensor,
                          alpha: torch.Tensor,
                          gamma: float = 2.0) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    tgt = target_ids.long().unsqueeze(1)
    log_pt = log_probs.gather(1, tgt).squeeze(1)
    pt     = probs.gather(1, tgt).squeeze(1)
    alpha_t = alpha[tgt.squeeze(1)]
    loss = -alpha_t * (1 - pt).pow(gamma) * log_pt
    return loss.mean()

@torch.no_grad()
def per_class_iou(pred_ids: torch.Tensor, target_ids: torch.Tensor, num_classes: int) -> List[float]:
    ious = []
    for c in range(num_classes):
        p = (pred_ids == c); t = (target_ids == c)
        inter = (p & t).sum().item(); union = (p | t).sum().item()
        ious.append(inter / union if union > 0 else 1.0)
    return ious

# Boundary F1 for thin lane quality
def _binary_boundary(mask: torch.Tensor) -> torch.Tensor:
    lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=mask.dtype, device=mask.device).view(1,1,3,3)
    b = F.conv2d(mask, lap, padding=1).abs()
    return (b > 0).float()

def _dilate(binmap: torch.Tensor, r: int) -> torch.Tensor:
    k = 2*r + 1
    return F.max_pool2d(binmap, kernel_size=k, stride=1, padding=r)

@torch.no_grad()
def boundary_f1(pred_ids: torch.Tensor, target_ids: torch.Tensor, lane_class_ids: List[int], r: int = 2) -> float:
    pred_lane = torch.zeros_like(pred_ids, dtype=torch.float32)
    targ_lane = torch.zeros_like(target_ids, dtype=torch.float32)
    for c in lane_class_ids:
        pred_lane += (pred_ids == c).float()
        targ_lane += (target_ids == c).float()
    pred_lane = (pred_lane > 0).float().unsqueeze(1)
    targ_lane = (targ_lane > 0).float().unsqueeze(1)
    pb = _binary_boundary(pred_lane); tb = _binary_boundary(targ_lane)
    pb_d = _dilate(pb, r); tb_d = _dilate(tb, r)
    tp_p = (pb * tb_d).sum().item(); pp = pb.sum().item()
    precision = tp_p / pp if pp > 0 else 1.0
    tp_t = (tb * pb_d).sum().item(); tp = tb.sum().item()
    recall = tp_t / tp if tp > 0 else 1.0
    return 2*precision*recall/(precision+recall) if (precision+recall)>0 else 1.0
@torch.no_grad()
def per_class_prf1(pred_ids: torch.Tensor, target_ids: torch.Tensor, num_classes: int):
    """
    Returns per-class precision, recall, and F1 score.
    """
    precision, recall, f1 = [], [], []
    for c in range(num_classes):
        p = (pred_ids == c)
        t = (target_ids == c)
        tp = (p & t).sum().item()
        fp = (p & ~t).sum().item()
        fn = (~p & t).sum().item()

        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1_c = 2 * prec * rec / (prec + rec + 1e-8)
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_c)
    return precision, recall, f1

@torch.no_grad()
def lane_pixelwise_f1(pred_ids: torch.Tensor, target_ids: torch.Tensor, lane_classes: List[int]):
    """
    Computes F1 over all lane pixels (binary mask of all lane classes).
    """
    pred_lane = torch.zeros_like(pred_ids).bool()
    targ_lane = torch.zeros_like(target_ids).bool()
    for c in lane_classes:
        pred_lane |= (pred_ids == c)
        targ_lane |= (target_ids == c)

    tp = (pred_lane & targ_lane).sum().item()
    fp = (pred_lane & ~targ_lane).sum().item()
    fn = (~pred_lane & targ_lane).sum().item()

    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return f1

@torch.no_grad()
def estimate_class_weights(loader, num_classes: int, device: torch.device) -> torch.Tensor:
    hist = torch.zeros(num_classes, dtype=torch.float64)
    for _, target in loader:
        target = target.to(device)
        for c in range(num_classes):
            hist[c] += (target == c).sum().item()
    freq = hist / hist.sum().clamp(min=1.0)
    med = torch.median(freq[freq > 0])
    weights = med / freq.clamp(min=1e-8)
    return weights.float()

@dataclass
class TrainConfig:
    num_classes: int
    lane_class_ids: List[int]
    epochs: int = 100
    lr: float = 3e-3
    weight_decay: float = 1e-2
    loss_type: str = "focal+dice"   # or "dice+ce"
    use_estimated_class_weights: bool = True
