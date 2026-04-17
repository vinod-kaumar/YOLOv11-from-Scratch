# ── Imports ───────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

# ── Model Scales ──────────────────────────────────
# model: [depth(d), width(w), max_channels(mc)]
model_scales = {
  "yolo11n": [0.50, 0.25, 1024],
  "yolo11s": [0.50, 0.50, 1024],
  "yolo11m": [0.50, 1.00, 512],
  "yolo11l": [1.00, 1.00, 512],
  "yolo11x": [1.00, 1.50, 512]}

# ── Backbone Building Blocks ──────────────────────
def autopad(k, p = None, d = 1):
    if d > 1:
        d = d*(k-1) + 1 if isinstance(k, int) else [d*(x-1) + 1 for x in k]

    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()
    
    def __init__(self, c1, c2, k = 1, s = 1, p = None, g = 1, d = 1 , act = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups = g, dilation = d, bias = False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (self.default_act if act is True 
                   else act if isinstance(act, nn.Module) 
                   else nn.Identity()
        )    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut = True, g = 1, k = (3, 3), e = 0.5):
        super().__init__()
        c_  = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g = g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y
    
class C2f(nn.Module):
    def __init__(self, c1, c2, n = 1, shortcut = False, g = 1, e = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2*self.c, 1, 1)
        self.cv2 = Conv((2 + n)*self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k = ((3, 3), (3, 3)), e = 1.0) for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class C3(nn.Module):
    def __init__(self, c1, c2, n = 1, shortcut = True, g = 1, e = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1,)
        self.cv3 = Conv(2*c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k = ((3, 3), (3, 3)), e = 1.0) for _ in range(n))
        )
        
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y1 = self.m(y1)
        return self.cv3(torch.cat((y1, y2), dim=1))    

class C3k(C3):
    def __init__(self, c1, c2, n = 1, shortcut = True, g = 1, e = 0.5, k = 3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k = (k, k), e = 1.0) for _ in range(n))
        )
  
        
class C3k2(C2f):
    def __init__(self, c1, c2, n = 1, c3k = False, e = 0.5, g = 1, shortcut = True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c,2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
    
class SPPF(nn.Module):
    def __init__(self, c1, c2, k = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_*4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size = k, stride = 1, padding = k // 2)
        
    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
        

# ── Head Building Blocks ──────────────────────────
class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, attn_ratio = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act= False)
        self.proj = Conv(dim, dim, 1, act = False)
        self.pe = Conv(dim, dim, 3, 1, g = dim, act = False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim = 2
        )
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
    
class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio = 0.5, num_heads = 4, shortcut = True) -> None:
        super().__init__()
        
        self.attn = Attention(c, attn_ratio = attn_ratio, num_heads = num_heads)
        self.ffn = nn.Sequential(
            Conv(c, c*2, 1), Conv(c*2, c, 1, act = False)
        )
        self.add = shortcut
        
    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    
class C2PSA(nn.Module):
    def __init__(self, c1, c2, n = 1, e = 0.5):
        super().__init__()
        assert c1 == c2, "Input and output channels must be the same for C2PSA"
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2*self.c, 1)
        self.cv2 = Conv(2*self.c, c1, 1)
        
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio = 0.5, num_heads = self.c // 64) for _ in range(n))
        )
    
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim = 1)
        b = self.m(b)
        return self.cv2(torch.cat((a,b), dim = 1))
    
class Concat(nn.Module):
    def __init__(self, dimension = 1):
        super().__init__()
        self.d = dimension
        
    def forward(self, x):
        return torch.cat(x, dim = self.d)

class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = x.view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
    
def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device

    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh   = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

class Detect(nn.Module):
    end2end = False
    legacy = False

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.shape = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )

        self.cv3 = (
            nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3),
                    Conv(c3, c3, 3),
                    nn.Conv2d(c3, self.nc, 1)
                )
                for x in ch
            )
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1)
                )
                for x in ch
            )
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), dim=1)

        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], dim=2)

        if self.shape != shape:
            anchors, strides = make_anchors(x, self.stride, 0.5)
            self.anchors = anchors.transpose(0, 1)
            self.strides = strides.transpose(0, 1)
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), dim=1)

    def decode_bboxes(self, bboxes, anchors):
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
            
# ── Model Classes ─────────────────────────────────
def get_model_params(model_name):
    d, w, mc = model_scales[model_name]
    def ch(base):
        return min(int(base * w), mc)
    def dep(base):
        return max(round(base * d), 1)
    return ch, dep

class Backbone(nn.Module):
    def __init__(self, model_name="yolo11n"):
        super().__init__()
        ch, dep = get_model_params(model_name)
        self.layer0 = Conv(3,         ch(64),   k=3, s=2)
        self.layer1 = Conv(ch(64),    ch(128),  k=3, s=2)
        self.layer2 = C3k2(ch(128),   ch(256),  n=dep(2), c3k=False, e=0.25)
        self.layer3 = Conv(ch(256),   ch(256),  k=3, s=2)
        self.layer4 = C3k2(ch(256),   ch(512),  n=dep(2), c3k=False, e=0.25)
        self.layer5 = Conv(ch(512),   ch(512),  k=3, s=2)
        self.layer6 = C3k2(ch(512),   ch(512),  n=dep(2), c3k=True,  e=0.25)
        self.layer7 = Conv(ch(512),   ch(1024), k=3, s=2)
        self.layer8 = C3k2(ch(1024),  ch(1024), n=dep(2), c3k=True,  e=0.25)
        self.layer9 = SPPF(ch(1024),  ch(1024), k=5)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        p3 = self.layer4(x)   
        x  = self.layer5(p3)
        p4 = self.layer6(x)   
        x  = self.layer7(p4)
        x  = self.layer8(x)
        p5 = self.layer9(x)  
        return p3, p4, p5

class Neck(nn.Module):
    def __init__(self, model_name="yolo11n"):
        super().__init__()
        ch, dep = get_model_params(model_name)
        self.layer10 = C2PSA(ch(1024), ch(1024), n=dep(2))
        self.layer11 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer12 = Concat(dimension=1)
        self.layer13 = C3k2(ch(1024) + ch(512), ch(512), n=dep(2), c3k=False, e=0.5)
        self.layer14 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer15 = Concat(dimension=1)
        self.layer16 = C3k2(ch(512) + ch(512), ch(256), n=dep(2), c3k=False, e=0.5)
        self.layer17 = Conv(ch(256), ch(256), k=3, s=2)
        self.layer18 = Concat(dimension=1)
        self.layer19 = C3k2(ch(256) + ch(512), ch(512), n=dep(2), c3k=False, e=0.5)
        self.layer20 = Conv(ch(512), ch(512), k=3, s=2)
        self.layer21 = Concat(dimension=1)
        self.layer22 = C3k2(ch(512) + ch(1024), ch(1024), n=dep(2), c3k=True, e=0.5)

    def forward(self, p3, p4, p5):
        x10 = self.layer10(p5)
        x = self.layer11(x10)
        x = self.layer12([x, p4])
        x13 = self.layer13(x)
        x = self.layer14(x13)
        x = self.layer15([x, p3])
        n3 = self.layer16(x)
        x = self.layer17(n3)
        x = self.layer18([x, x13])
        n4 = self.layer19(x)
        x = self.layer20(n4)
        x = self.layer21([x, x10])
        n5 = self.layer22(x)
        return n3, n4, n5

class DetectHead(nn.Module):
    def __init__(self, model_name="yolo11n", nc=1):
        super().__init__()
        ch, dep = get_model_params(model_name)
        neck_channels = (ch(256), ch(512), ch(1024))
        self.detect = Detect(nc=nc, ch=neck_channels)
        self.detect.stride = torch.tensor([8.0, 16.0, 32.0])
        self.detect.bias_init()

    def forward(self, x):
        return self.detect(x)

class YOLOv11(nn.Module):
    def __init__(self, model_name="yolo11n", nc=1):
        super().__init__()
        self.model_name = model_name
        self.nc = nc
        self.backbone = Backbone(model_name)
        self.neck = Neck(model_name)
        self.head = DetectHead(model_name, nc)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        return self.head([n3, n4, n5])


# post-processing methods

def apply_nms(predictions, conf_thresh=0.25, iou_thresh=0.45):
    """
    Apply NMS to raw model output.

    predictions : [B, 8400, 5]  (cx, cy, w, h, score)  pixel space
                  model in eval mode already decoded boxes to xywh

    Returns list of length B, each element:
        np.ndarray [N, 6]  (x1, y1, x2, y2, score, class)  or empty
    """
    results = []
    for pred in predictions:          # pred: [8400, 5]
        scores = pred[:, 4]
        mask   = scores > conf_thresh
        pred   = pred[mask]

        if pred.shape[0] == 0:
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue

        # xywh -> xyxy
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = cx - w / 2;  y1 = cy - h / 2
        x2 = cx + w / 2;  y2 = cy + h / 2
        boxes  = torch.stack([x1, y1, x2, y2], dim=1)
        scores = pred[:, 4]

        keep   = nms(boxes, scores, iou_thresh)
        boxes  = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        labels = np.zeros(len(keep), dtype=np.float32)  # nc=1, class always 0

        results.append(
            np.concatenate([boxes, scores[:, None], labels[:, None]], axis=1)
        )
    return results

def box_iou_numpy(box1, box2):
    """
    box1 : [N, 4]  xyxy
    box2 : [M, 4]  xyxy
    Returns [N, M] IoU matrix
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]

    inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])  # [N, M]
    inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union

def compute_ap(recall, precision):
    """Compute area under PR curve using 101-point interpolation."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx  = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

def compute_map(all_preds, all_gts, iou_thresholds=None, img_size=640):
    """
    Compute mAP@0.5 and mAP@0.5:0.95.

    all_preds : list of np.ndarray [N, 6]  (x1,y1,x2,y2, score, cls)
    all_gts   : list of np.ndarray [M, 5]  (cls, cx,cy,w,h) normalized
                GT boxes are still in normalized format - we convert below

    Returns: map50 (float), map5095 (float), precision (float), recall (float)
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)

    ap_per_threshold = []

    for iou_thr in iou_thresholds:
        tp_list, conf_list, n_gt_total = [], [], 0

        for preds, gts in zip(all_preds, all_gts):
            # convert GT from normalized cxcywh -> pixel xyxy (IMG_SIZE)
            if gts.shape[0] > 0:
                gts_px = gts[gts[:, 1:].sum(-1) > 0]   # remove padding
                if gts_px.shape[0] > 0:
                    cx  = gts_px[:, 1] * img_size
                    cy  = gts_px[:, 2] * img_size
                    w   = gts_px[:, 3] * img_size
                    h   = gts_px[:, 4] * img_size
                    gt_boxes = np.stack(
                        [cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1
                    )
                else:
                    gt_boxes = np.zeros((0, 4))
            else:
                gt_boxes = np.zeros((0, 4))

            n_gt_total += len(gt_boxes)
            matched = np.zeros(len(gt_boxes), dtype=bool)

            if preds.shape[0] == 0:
                continue

            # sort preds by descending confidence
            order  = np.argsort(-preds[:, 4])
            preds  = preds[order]

            for pred in preds:
                conf_list.append(pred[4])
                if len(gt_boxes) == 0:
                    tp_list.append(0)
                    continue
                ious    = box_iou_numpy(pred[:4][None], gt_boxes)[0]  # [M]
                best_i  = np.argmax(ious)
                if ious[best_i] >= iou_thr and not matched[best_i]:
                    tp_list.append(1)
                    matched[best_i] = True
                else:
                    tp_list.append(0)

        if len(tp_list) == 0 or n_gt_total == 0:
            ap_per_threshold.append(0.0)
            continue

        tp_arr   = np.array(tp_list, dtype=np.float32)
        conf_arr = np.array(conf_list)
        order    = np.argsort(-conf_arr)
        tp_arr   = tp_arr[order]

        cum_tp  = np.cumsum(tp_arr)
        cum_fp  = np.cumsum(1 - tp_arr)
        recall  = cum_tp / (n_gt_total + 1e-7)
        precision = cum_tp / (cum_tp + cum_fp + 1e-7)

        ap_per_threshold.append(compute_ap(recall, precision))

    # precision/recall at iou=0.5 for logging
    final_prec = precision[-1] if len(tp_list) > 0 else 0.0
    final_rec  = recall[-1]    if len(tp_list) > 0 else 0.0

    map50   = ap_per_threshold[0]
    map5095 = float(np.mean(ap_per_threshold))
    return map50, map5095, float(final_prec), float(final_rec)


if __name__ == "__main__":
    pass
