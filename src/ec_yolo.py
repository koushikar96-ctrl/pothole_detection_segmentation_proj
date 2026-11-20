"""EC-YOLO detector — architecture exactly matching best_ec_yolo.pth checkpoint.

Backbone : EfficientNet-B0 (torchvision MBConv blocks, stages 0-8)
Neck     : SPPF → C2PSA → PAN (C3K blocks, reduce/upsample/down-sample)
Head     : CoordAtt on P5 + YOLOv8-style decoupled DFL head (cv2 box + cv3 cls)
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """Conv → BN → SiLU used throughout neck and heads."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DFL(nn.Module):
    """Distribution Focal Loss convolution: maps reg channels → 4 coords."""

    def __init__(self, c1: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, c1*4, H, W) → (B, 4, H, W) via softmax-weighted sum
        b, c, h, w = x.shape
        c1 = c // 4
        x = x.view(b, 4, c1, h * w).softmax(2)
        return self.conv(x.view(b * 4, c1, h, w)).view(b, 4, h, w)


# ---------------------------------------------------------------------------
# EfficientNet-B0 backbone  (reuse torchvision's MBConv blocks)
# ---------------------------------------------------------------------------

class EfficientNetB0Backbone(nn.Module):
    """Wrap torchvision EfficientNet-B0 feature stages as backbone.

    Extracts feature maps at strides 8 (stage-3 output), 16 (stage-5), 32 (stage-8).
    Channel widths: P3=40, P4=112, P5=1280.
    """

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import efficientnet_b0
        base = efficientnet_b0(weights=None)
        # features[0]  : stem  (stride 2)  → 32 ch
        # features[1]  : stage1 (stride 1) → 16 ch
        # features[2]  : stage2 (stride 2) → 24 ch   cumulative stride 4
        # features[3]  : stage3 (stride 2) → 40 ch   cumulative stride 8   (P3)
        # features[4]  : stage4 (stride 1) → 80 ch
        # features[5]  : stage5 (stride 2) → 112 ch  cumulative stride 16  (P4)
        # features[6]  : stage6 (stride 1) → 192 ch
        # features[7]  : stage7 (stride 2) → 320 ch  cumulative stride 32
        # features[8]  : head conv          → 1280 ch cumulative stride 32  (P5)
        self.efficientnet = base.features  # nn.Sequential of 9 sub-modules

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = x
        for i, layer in enumerate(self.efficientnet):
            y = layer(y)
            if i == 3:
                p3 = y   # 40 ch, stride 8
            elif i == 5:
                p4 = y   # 112 ch, stride 16
        p5 = y           # 1280 ch, stride 32  (after layer 8)
        return {"p3": p3, "p4": p4, "p5": p5}


# ---------------------------------------------------------------------------
# Neck building blocks
# ---------------------------------------------------------------------------

class SPPF(nn.Module):
    """Spatial Pyramid Pooling – Fast (YOLOv8 style)."""

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNAct(c1, c_, 1)
        self.cv2 = ConvBNAct(c_ * 4, c2, 1)
        self.pool = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class PSABlock(nn.Module):
    """Positional Self-Attention block used inside C2PSA."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio, num_heads)
        self.ffn = nn.Sequential(
            ConvBNAct(c, c * 2, 1),
            ConvBNAct(c * 2, c, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class Attention(nn.Module):
    """Multi-head attention with relative position encoding."""

    def __init__(self, dim: int, attn_ratio: float = 0.5, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = ConvBNAct(dim, h, 1)
        self.proj = ConvBNAct(dim, dim, 1)
        self.pe = ConvBNAct(dim, dim, 3, g=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        nh_kd = self.key_dim * self.num_heads
        q, k, v = qkv.split([self.num_heads * self.head_dim, nh_kd, nh_kd], dim=1)
        q = q.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.key_dim, N).permute(0, 1, 2, 3)
        v = v.reshape(B, self.num_heads, self.key_dim, N).permute(0, 1, 3, 2)
        attn = (q @ k) * self.scale
        attn = attn.softmax(-1)
        x_out = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        x_out = x_out + self.pe(v.permute(0, 1, 3, 2).reshape(B, C, H, W))
        return self.proj(x_out)


class C2PSA(nn.Module):
    """C2 with Positional Self-Attention (YOLOv8 style)."""

    def __init__(self, c1: int, c2: int, n: int = 2, e: float = 0.5) -> None:
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, 1)
        self.cv2 = ConvBNAct(2 * self.c, c2, 1)
        self.m = nn.Sequential(*[PSABlock(self.c) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split(self.c, 1)
        b = self.m(b)
        return self.cv2(torch.cat([a, b], 1))


class Bottleneck(nn.Module):
    """Standard YOLOv8 bottleneck."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, k: tuple = (3, 3), e: float = 0.5) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, k[0])
        self.cv2 = ConvBNAct(c_, c2, k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3K(nn.Module):
    """C3 with configurable kernel bottleneck (C3K). Matches neck.c3k_* layers."""

    def __init__(self, c1: int, c2: int, n: int = 2, shortcut: bool = False, e: float = 0.5) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, 1)
        self.cv2 = ConvBNAct(c1, c_, 1)
        self.cv3 = ConvBNAct(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1))


# ---------------------------------------------------------------------------
# CoordAtt (applied to P5 feature map)
# ---------------------------------------------------------------------------

class CoordAtt(nn.Module):
    """Coordinate Attention module (applied on P5 before detection heads)."""

    def __init__(self, inp: int, reduction: int = 32) -> None:
        super().__init__()
        hidden = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv_h = nn.Conv2d(hidden, inp, 1)
        self.conv_w = nn.Conv2d(hidden, inp, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)                       # (N,C,H,1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (N,C,1,W) → (N,C,W,1)
        y = torch.cat([x_h, x_w], dim=2)                        # (N,C,H+W,1)
        y = F.relu(self.bn1(self.conv1(y)), inplace=True)
        x_h, x_w = y.split([h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_h * a_w


# ---------------------------------------------------------------------------
# Full EC-YOLO neck
# ---------------------------------------------------------------------------

class ECYOLONeck(nn.Module):
    """PAN/FPN neck matching best_ec_yolo.pth.

    In-channels (from EfficientNet-B0): P3=40, P4=112, P5=1280.
    Out-channels for detection:          P3=128, P4=256, P5=512.
    """

    def __init__(self) -> None:
        super().__init__()
        # Adapt backbone channels to uniform neck channels
        self.p3_adapt = ConvBNAct(40, 128, 1)    # 40  → 128
        self.p4_adapt = ConvBNAct(112, 256, 1)   # 112 → 256
        self.p5_adapt = ConvBNAct(1280, 512, 1)  # 1280→ 512

        # P5 processing
        self.sppf = SPPF(512, 512)
        self.c2psa = C2PSA(512, 512, n=2)

        # Top-down path
        self.reduce_p5 = ConvBNAct(512, 256, 1)
        self.c3k_n4 = C3K(512, 256, n=2)          # cat(reduce_p5↑, p4_adapt) = 512 → 256
        self.reduce_n4 = ConvBNAct(256, 128, 1)
        self.c3k_n3 = C3K(256, 128, n=2)          # cat(reduce_n4↑, p3_adapt) = 256 → 128

        # Bottom-up path
        self.down_n3 = ConvBNAct(128, 256, 3, 2)
        self.c3k_out4 = C3K(512, 256, n=2)        # cat(down_n3, c3k_n4) = 512 → 256
        self.down_n4 = ConvBNAct(256, 512, 3, 2)
        self.c3k_out5 = C3K(1024, 512, n=2)       # cat(down_n4, c2psa) = 1024 → 512

    def forward(self, backbone_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p3_raw = self.p3_adapt(backbone_out["p3"])   # (B,128,80,80)
        p4_raw = self.p4_adapt(backbone_out["p4"])   # (B,256,40,40)
        p5_raw = self.p5_adapt(backbone_out["p5"])   # (B,512,20,20)

        p5 = self.c2psa(self.sppf(p5_raw))          # (B,512,20,20)

        # Top-down
        n4 = self.c3k_n4(torch.cat([F.interpolate(self.reduce_p5(p5), scale_factor=2, mode="nearest"), p4_raw], 1))
        n3 = self.c3k_n3(torch.cat([F.interpolate(self.reduce_n4(n4), scale_factor=2, mode="nearest"), p3_raw], 1))

        # Bottom-up
        out4 = self.c3k_out4(torch.cat([self.down_n3(n3), n4], 1))
        out5 = self.c3k_out5(torch.cat([self.down_n4(out4), p5], 1))

        return {"p3": n3, "p4": out4, "p5": out5}


# ---------------------------------------------------------------------------
# Detection head (YOLOv8 decoupled DFL head)
# ---------------------------------------------------------------------------

class DetectHead(nn.Module):
    """YOLOv8-style decoupled detection head for one class.

    cv2 branches predict box distribution (reg_max=16 → 64 channels).
    cv3 branches predict class logit (1 class → 1 channel).
    """

    def __init__(self, nc: int = 1, reg_max: int = 16, ch: tuple = (128, 256, 512)) -> None:
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + reg_max * 4  # outputs per anchor
        self.dfl = DFL(reg_max)

        c2 = max(16, reg_max * 4)
        # Box regression branches
        self.cv2 = nn.ModuleList([
            nn.Sequential(ConvBNAct(x, c2, 3), ConvBNAct(c2, c2, 3), nn.Conv2d(c2, reg_max * 4, 1))
            for x in ch
        ])
        # Class prediction branches  (depthwise then pointwise pattern from checkpoint)
        c3 = max(nc, ch[0])
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                nn.Sequential(ConvBNAct(x, x, 3, g=x), ConvBNAct(x, c3, 1)),
                nn.Sequential(ConvBNAct(c3, c3, 3, g=c3), ConvBNAct(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [torch.cat([self.cv2[i](f), self.cv3[i](f)], 1) for i, f in enumerate(features)]


# ---------------------------------------------------------------------------
# Full EC-YOLO model
# ---------------------------------------------------------------------------

class ECYOLODetector(nn.Module):
    """Full EC-YOLO: EfficientNet-B0 backbone + PAN neck + DFL detection head."""

    def __init__(self, nc: int = 1) -> None:
        super().__init__()
        self.backbone = EfficientNetB0Backbone()
        self.neck = ECYOLONeck()
        self.ca = CoordAtt(512, reduction=32)  # applied to P5 after neck
        self.detect = DetectHead(nc=nc, reg_max=16, ch=(128, 256, 512))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        bb = self.backbone(x)
        neck_out = self.neck(bb)
        # Apply CoordAtt to P5
        neck_out["p5"] = self.ca(neck_out["p5"])
        feats = [neck_out["p3"], neck_out["p4"], neck_out["p5"]]
        return self.detect(feats)


# ---------------------------------------------------------------------------
# Decoding / NMS
# ---------------------------------------------------------------------------

def _decode_dfl(box_dist: torch.Tensor, reg_max: int = 16) -> torch.Tensor:
    """Decode DFL box distribution (B,4*reg_max,H,W) → (B,4,H,W) ltrb."""
    B, _, H, W = box_dist.shape
    box_dist = box_dist.view(B, 4, reg_max, H * W).softmax(2)
    proj = torch.arange(reg_max, dtype=box_dist.dtype, device=box_dist.device)
    ltrb = (box_dist * proj.view(1, 1, reg_max, 1)).sum(2)  # (B,4,HW)
    return ltrb.view(B, 4, H, W)


def decode_predictions(
    predictions: List[torch.Tensor],
    image_size: int = 640,
    threshold: float = 0.25,
    iou_threshold: float = 0.45,
    reg_max: int = 16,
) -> list[list[tuple[float, float, float, float, float]]]:
    """Decode a list of per-scale raw head outputs into NMS-filtered boxes.

    Parameters
    ----------
    predictions:
        List of tensors [P3, P4, P5] each shaped (B, reg_max*4+nc, H, W).
    image_size:
        Side length of the square input image (used only for scaling).
    threshold:
        Minimum objectness confidence.
    iou_threshold:
        NMS IoU threshold.
    reg_max:
        DFL regression max bins (default 16).

    Returns
    -------
    Per-image list of ``(x1, y1, x2, y2, confidence)`` tuples in absolute pixels.
    """
    strides = [8, 16, 32]
    batch_size = predictions[0].shape[0]
    detections: list[list[tuple]] = [[] for _ in range(batch_size)]

    for pred, stride in zip(predictions, strides):
        B, C, H, W = pred.shape
        nc = C - reg_max * 4
        box_logits = pred[:, :reg_max * 4]   # (B, 64, H, W)
        cls_logits = pred[:, reg_max * 4:]   # (B, nc, H, W)

        # Decode DFL → ltrb distances
        ltrb = _decode_dfl(box_logits, reg_max)  # (B,4,H,W)  in grid units
        cls_prob = cls_logits.sigmoid()           # (B,nc,H,W)

        ltrb_np = ltrb.detach().cpu().numpy()
        cls_np = cls_prob.detach().cpu().numpy()

        for b in range(B):
            for gy in range(H):
                for gx in range(W):
                    conf = float(cls_np[b, 0, gy, gx])
                    if conf < threshold:
                        continue
                    # Anchor point in pixels
                    ax = (gx + 0.5) * stride
                    ay = (gy + 0.5) * stride
                    l, t, r, bot = (float(ltrb_np[b, i, gy, gx]) * stride for i in range(4))
                    x1 = ax - l
                    y1 = ay - t
                    x2 = ax + r
                    y2 = ay + bot
                    detections[b].append((x1, y1, x2, y2, conf))

    results = []
    for candidates in detections:
        boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _ in candidates]
        scores = [s for *_, s in candidates]
        kept = [] if not boxes else cv2.dnn.NMSBoxes(boxes, scores, threshold, iou_threshold)
        results.append([candidates[int(i)] for i in kept] if len(kept) else [])

    return results