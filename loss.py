import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import make_anchors

def bbox_iou(box1, box2, xywh=True, CIoU=False, eps=1e-7):
    """
    box1, box2 : [N, 4]  — must be same shape, element-wise comparison
    xywh       : True  → input is (cx, cy, w, h)
                 False → input is (x1, y1, x2, y2)
    """
    if xywh:
        (x1, y1, w1, h1) = box1.unbind(-1)
        (x2, y2, w2, h2) = box2.unbind(-1)
        b1_x1, b1_x2 = x1 - w1/2, x1 + w1/2
        b1_y1, b1_y2 = y1 - h1/2, y1 + h1/2
        b2_x1, b2_x2 = x2 - w2/2, x2 + w2/2
        b2_y1, b2_y2 = y2 - h2/2, y2 + h2/2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    union = w1*h1 + w2*h2 - inter + eps
    iou   = inter / union

    if CIoU:
        cw   = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch   = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2   = cw**2 + ch**2 + eps
        rho2 = ((b2_x1+b2_x2 - b1_x1-b1_x2)**2 +
                (b2_y1+b2_y2 - b1_y1-b1_y2)**2) / 4
        v    = (4/math.pi**2) * (torch.atan(w2/(h2+eps)) - torch.atan(w1/(h1+eps)))**2
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2/c2 + v*alpha)

    return iou


# ── DFLoss ─────────────────────────────────────────────────────────────
class DFLoss(nn.Module):
    """
    Distribution Focal Loss.

    pred_dist : [n_fg, 4*reg_max]   raw logits per anchor per direction
    target    : [n_fg, 4]           float distances (already in grid units)
    """
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        # clamp target to valid range
        target = target.clamp(0, self.reg_max - 1 - 1e-6)   # [n_fg, 4]
        tl = target.long()                                    # floor bin
        tr = (tl + 1).clamp(max=self.reg_max - 1)            # ceil bin
        wl = tr.float() - target                              # weight for floor
        wr = 1.0 - wl                                         # weight for ceil

        # pred_dist: [n_fg, 4*reg_max] → [n_fg*4, reg_max]
        pred = pred_dist.view(-1, self.reg_max)

        loss = (
            F.cross_entropy(pred, tl.view(-1), reduction='none') * wl.view(-1) +
            F.cross_entropy(pred, tr.view(-1), reduction='none') * wr.view(-1)
        )
        # [n_fg*4] → [n_fg, 4] → mean over 4 directions → [n_fg]
        return loss.view_as(target).mean(-1)


# ── TaskAlignedAssigner ────────────────────────────────────────────────
class TaskAlignedAssigner(nn.Module):
    """
    Assigns GT boxes to anchor points.

    Inputs are all in pixel space, boxes in xyxy format.

    alignment score = cls_score^alpha * iou^beta
    Top-k anchors per GT (that also lie inside the GT box) are positives.
    If one anchor is claimed by multiple GTs → assign to highest IoU GT.
    """
    def __init__(self, topk=13, nc=1, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk  = topk
        self.nc    = nc
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        pd_scores  : [B, 8400, 1]        predicted scores after sigmoid
        pd_bboxes  : [B, 8400, 4]        predicted boxes xyxy pixel space
        anc_points : [8400, 2]           anchor cx,cy pixel space
        gt_labels  : [B, max_gt, 1]      class ids (all 0 for polyp)
        gt_bboxes  : [B, max_gt, 4]      GT boxes xyxy pixel space
        mask_gt    : [B, max_gt, 1]      1=real GT, 0=padding

        Returns
        -------
        target_labels : [B, 8400]        class index for each anchor
        target_bboxes : [B, 8400, 4]     GT box assigned to each anchor (xyxy)
        target_scores : [B, 8400, 1]     alignment-normalized score
        fg_mask       : [B, 8400] bool   True = positive anchor
        """
        B, max_gt, _ = gt_bboxes.shape
        device       = gt_bboxes.device
        na           = pd_bboxes.shape[1]   # 8400

        # ── 1. which anchors lie inside each GT box? ───────────────────
        # [B, max_gt, 8400]
        mask_in_gts = self._candidates_in_gts(anc_points, gt_bboxes)

        # combined validity: inside GT AND GT is not padding
        # mask_gt: [B, max_gt, 1] → [B, max_gt, 8400]
        valid_mask = mask_in_gts * mask_gt.squeeze(-1).unsqueeze(-1)  # [B, max_gt, 8400]

        # ── 2. alignment score per (GT, anchor) pair ───────────────────
        align_metric, overlaps = self._box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, valid_mask
        )
        # align_metric : [B, max_gt, 8400]
        # overlaps     : [B, max_gt, 8400]

        # ── 3. top-k anchors per GT ────────────────────────────────────
        mask_topk = self._select_topk(align_metric, na)  # [B, max_gt, 8400]

        # positive = inside GT + top-k + real GT
        mask_pos = mask_topk * valid_mask                 # [B, max_gt, 8400]

        # ── 4. resolve conflicts ───────────────────────────────────────
        target_gt_idx, fg_mask, mask_pos = self._resolve_conflicts(
            mask_pos, overlaps
        )
        # target_gt_idx : [B, 8400]
        # fg_mask       : [B, 8400]

        # ── 5. gather targets ──────────────────────────────────────────
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask, B, max_gt, device
        )

        # ── 6. normalize scores by alignment ──────────────────────────
        # for each GT, find its best alignment score and best overlap
        align_metric  = align_metric * mask_pos                        # zero negatives
        pos_align_max = align_metric.amax(dim=-1, keepdim=True)        # [B, max_gt, 1]
        pos_iou_max   = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # [B, max_gt, 1]

        # normalized weight per (GT, anchor): [B, max_gt, 8400]
        # norm_weights  = (overlaps * mask_pos) / (pos_align_max + self.eps) * pos_iou_max
        norm_weights  = (align_metric * mask_pos) / (pos_align_max + self.eps) * pos_iou_max
        
        # per anchor: take the max across GTs → [B, 8400]
        norm_weights  = norm_weights.amax(dim=1)

        # scale target_scores: [B, 8400, 1]
        target_scores = target_scores * norm_weights.unsqueeze(-1)

        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    # ── internal helpers ───────────────────────────────────────────────

    def _candidates_in_gts(self, anc_points, gt_bboxes):
        """
        anc_points : [8400, 2]
        gt_bboxes  : [B, max_gt, 4]  xyxy pixel space
        Returns    : [B, max_gt, 8400]  True if anchor inside GT
        """
        ap = anc_points.unsqueeze(0).unsqueeze(0)   # [1, 1, 8400, 2]
        gb = gt_bboxes.unsqueeze(2)                  # [B, max_gt, 1, 4]
        lt = ap - gb[..., :2]                        # [B, max_gt, 8400, 2]
        rb = gb[..., 2:] - ap                        # [B, max_gt, 8400, 2]
        return torch.cat((lt, rb), dim=-1).amin(dim=-1) > 0  # [B, max_gt, 8400]

    def _box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, valid_mask):
        """
        Compute alignment score and IoU for every (GT, anchor) pair.

        pd_scores  : [B, 8400, 1]
        pd_bboxes  : [B, 8400, 4]   xyxy
        gt_labels  : [B, max_gt, 1]
        gt_bboxes  : [B, max_gt, 4] xyxy
        valid_mask : [B, max_gt, 8400]

        Returns
        -------
        align_metric : [B, max_gt, 8400]
        overlaps     : [B, max_gt, 8400]
        """
        B, na, _   = pd_bboxes.shape
        _, max_gt, _ = gt_bboxes.shape

        # nc=1 → cls score for all anchors is just pd_scores[..., 0]
        # expand to [B, max_gt, 8400]
        bbox_scores = pd_scores[..., 0].unsqueeze(1).expand(-1, max_gt, -1)

        # IoU: compare every GT with every predicted box
        # expand both to [B, max_gt, 8400, 4] then flatten for bbox_iou
        gt_exp = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)   # [B, max_gt, 8400, 4]
        pd_exp = pd_bboxes.unsqueeze(1).expand(-1, max_gt, -1, -1)  # [B, max_gt, 8400, 4]

        overlaps = bbox_iou(
            gt_exp.reshape(-1, 4),
            pd_exp.reshape(-1, 4),
            xywh=False,
            CIoU=False
        ).reshape(B, max_gt, na).clamp(0)

        align_metric = (bbox_scores.clamp(0).pow(self.alpha) *
                        overlaps.pow(self.beta)) * valid_mask
        return align_metric, overlaps

    def _select_topk(self, align_metric, na):
        """
        align_metric : [B, max_gt, 8400]
        Returns      : [B, max_gt, 8400]  float mask
        """
        k = min(self.topk, na)
        topk_vals, _ = align_metric.topk(k, dim=-1, largest=True)
        threshold     = topk_vals[..., -1].unsqueeze(-1)  # k-th best
        return (align_metric >= threshold).float()

    def _resolve_conflicts(self, mask_pos, overlaps):
        """
        mask_pos : [B, max_gt, 8400]
        overlaps : [B, max_gt, 8400]

        If anchor claimed by >1 GT → assign to GT with highest IoU.

        Returns
        -------
        target_gt_idx : [B, 8400]
        fg_mask       : [B, 8400]
        mask_pos      : [B, max_gt, 8400]  cleaned
        """
        fg_mask = mask_pos.sum(dim=1)   # [B, 8400]  number of GTs per anchor

        if fg_mask.max() > 1:
            disputed = (fg_mask.unsqueeze(1) > 1).expand_as(mask_pos)
            best_gt  = overlaps.argmax(dim=1, keepdim=True).expand_as(mask_pos)
            one_hot  = torch.zeros_like(mask_pos).scatter_(1, best_gt, 1)
            mask_pos = torch.where(disputed, one_hot, mask_pos)
            fg_mask  = mask_pos.sum(dim=1)

        target_gt_idx = mask_pos.argmax(dim=1)   # [B, 8400]
        return target_gt_idx, fg_mask, mask_pos

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask,
                     B, max_gt, device):
        """
        Gather label, box, score targets for every anchor.

        Returns
        -------
        target_labels : [B, 8400]
        target_bboxes : [B, 8400, 4]
        target_scores : [B, 8400, 1]
        """
        batch_idx = torch.arange(B, device=device).unsqueeze(-1)   # [B, 1]
        flat_idx  = target_gt_idx + batch_idx * max_gt              # [B, 8400]

        target_labels = gt_labels.long().view(-1)[flat_idx]         # [B, 8400]
        target_bboxes = gt_bboxes.view(-1, 4)[flat_idx]             # [B, 8400, 4]
        target_labels = target_labels.clamp(0)

        # one-hot scores [B, 8400, 1]  (nc=1)
        target_scores = torch.zeros(
            (B, target_labels.shape[1], self.nc),
            device=device, dtype=torch.float
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1.0)
        target_scores *= fg_mask.unsqueeze(-1)   # zero out negative anchors

        return target_labels, target_bboxes, target_scores


# ── BboxLoss ───────────────────────────────────────────────────────────
class BboxLoss(nn.Module):
    """
    Computes CIoU + DFL loss for positive anchors only.

    pred_dist  : [B, 8400, 4*reg_max]  raw distribution logits
    pred_bboxes: [B, 8400, 4]          decoded boxes xyxy pixel space
    anc_points : [8400, 2]             anchor cx,cy pixel space
    stride_tensor: [8400, 1]           stride per anchor
    target_bboxes: [B, 8400, 4]        GT boxes xyxy pixel space
    target_scores: [B, 8400, 1]        alignment-normalized weights
    fg_mask    : [B, 8400] bool        positive anchor mask
    """
    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl = DFLoss(reg_max)
        self.reg_max = reg_max

    def forward(self, pred_dist, pred_bboxes, anc_points, stride_tensor,
                target_bboxes, target_scores, fg_mask):

        weight             = target_scores.squeeze(-1)[fg_mask]   # [n_fg]
        target_scores_sum  = target_scores.sum().clamp(min=1e-4)

        # ── CIoU loss ─────────────────────────────────────────────────
        pb = pred_bboxes[fg_mask]    # [n_fg, 4]  xyxy
        tb = target_bboxes[fg_mask]  # [n_fg, 4]  xyxy

        ciou     = bbox_iou(pb, tb, xywh=False, CIoU=True)
        loss_box = ((1.0 - ciou) * weight).sum() / target_scores_sum

        # ── DFL loss ──────────────────────────────────────────────────
        # convert GT xyxy → distances from anchor in stride units
        ap_fg = anc_points.unsqueeze(0).expand(
            pred_bboxes.shape[0], -1, -1
        )[fg_mask]                                 # [n_fg, 2]
        st_fg = stride_tensor.unsqueeze(0).expand(
            pred_bboxes.shape[0], -1, -1
        )[fg_mask]                                 # [n_fg, 1]

        # GT xyxy → (lt, rb) distances in grid units
        tb_lt  = ap_fg - tb[..., :2]              # [n_fg, 2]
        tb_rb  = tb[..., 2:] - ap_fg              # [n_fg, 2]
        dist_tgt = torch.cat([tb_lt, tb_rb], dim=-1) / st_fg   # [n_fg, 4]
        dist_tgt = dist_tgt.clamp(0, self.reg_max - 1 - 1e-6)

        loss_dfl = (self.dfl(pred_dist[fg_mask], dist_tgt) * weight).sum() / target_scores_sum

        return loss_box, loss_dfl


# ── v8DetectionLoss (main loss) ────────────────────────────────────────
class v8DetectionLoss(nn.Module):
    """
    Full YOLO11 detection loss.

    nc      = 1   (polyp)
    reg_max = 16
    strides = [8, 16, 32]

    Expects
    -------
    raw_preds : list of 3 tensors from model in train mode
                each [B, nc + 4*reg_max, H, W]  i.e. [B, 65, H, W]
    targets   : [B, max_gt, 5]
                each row = [class_id, cx, cy, w, h]  normalized 0-1

    img_size  : 640
    """
    def __init__(self, nc=1, reg_max=16, strides=None):
        super().__init__()
        self.nc      = nc
        self.reg_max = reg_max
        self.strides = strides or [8., 16., 32.]

        self.assigner = TaskAlignedAssigner(topk=13, nc=nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(reg_max)

        # loss weights (same as official ultralytics)
        self.lambda_cls = 0.5
        self.lambda_box = 7.5
        self.lambda_dfl = 1.5

    def forward(self, raw_preds, targets, img_size=640):
        """
        raw_preds : list of 3 feature maps [B, 65, H, W]
        targets   : [B, max_gt, 5]  normalized [cls, cx, cy, w, h]
        """
        device = raw_preds[0].device
        B      = raw_preds[0].shape[0]

        # ── 1. build anchors ───────────────────────────────────────────
        # reuse make_anchors from utils.py
        # returns anchor_points [8400, 2] and stride_tensor [8400, 1]
        # anchor_points are in grid-cell units (not pixel) from make_anchors
        # we need pixel space → multiply by stride
        anc_pts_grid, stride_tensor = make_anchors(raw_preds, self.strides, 0.5)
        anc_pts_pixel = anc_pts_grid * stride_tensor   # [8400, 2] pixel space

        # ── 2. split raw predictions ───────────────────────────────────
        # each raw_pred: [B, 65, H, W]
        # cat all scales → [B, 8400, 65]
        pred_cat = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in raw_preds], dim=1
        )   # [B, 8400, 65]

        pred_dist   = pred_cat[..., :4 * self.reg_max]   # [B, 8400, 64]  reg logits
        pred_logits = pred_cat[..., 4 * self.reg_max:]   # [B, 8400, 1]   cls logits
        pred_scores = pred_logits.sigmoid()               # [B, 8400, 1]

        # ── 3. decode predicted boxes ──────────────────────────────────
        # softmax over reg_max bins → weighted sum → distances
        # then convert distances + anchors → xyxy pixel space
        proj     = torch.arange(self.reg_max, dtype=torch.float, device=device)
        dist     = (pred_dist
                    .view(B, -1, 4, self.reg_max)
                    .softmax(-1) * proj).sum(-1)          # [B, 8400, 4]  grid units
        dist_px  = dist * stride_tensor.unsqueeze(0)      # [B, 8400, 4]  pixel units

        # lt, rb → x1y1, x2y2
        lt, rb   = dist_px[..., :2], dist_px[..., 2:]
        pred_bboxes = torch.cat([
            anc_pts_pixel.unsqueeze(0) - lt,              # x1y1
            anc_pts_pixel.unsqueeze(0) + rb               # x2y2
        ], dim=-1)                                         # [B, 8400, 4] xyxy pixel

        # ── 4. prepare GT targets ──────────────────────────────────────
        # targets: [B, max_gt, 5]  normalized [cls, cx, cy, w, h]
        gt_labels = targets[..., 0:1].long()               # [B, max_gt, 1]
        gt_cxcywh = targets[..., 1:5] * img_size           # [B, max_gt, 4] pixel cxcywh

        # convert to xyxy pixel space
        gt_bboxes = torch.cat([
            gt_cxcywh[..., :2] - gt_cxcywh[..., 2:] / 2,  # x1y1
            gt_cxcywh[..., :2] + gt_cxcywh[..., 2:] / 2   # x2y2
        ], dim=-1)                                           # [B, max_gt, 4] xyxy pixel

        # mask_gt: 1 for real GTs, 0 for padding rows (all zeros)
        mask_gt = (gt_cxcywh.sum(-1, keepdim=True) > 0).float()  # [B, max_gt, 1]

        # ── 5. TAL assignment ──────────────────────────────────────────
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach(),
            pred_bboxes.detach(),
            anc_pts_pixel,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        # target_labels : [B, 8400]
        # target_bboxes : [B, 8400, 4]  xyxy pixel
        # target_scores : [B, 8400, 1]
        # fg_mask       : [B, 8400] bool

        target_scores_sum = target_scores.sum().clamp(min=1e-4)

        # Uncomment below guard line if u want val loss in the same scale of train_loss        
        # ── guard: no foreground anchors (batch has only background) ──
        # if not fg_mask.any():
        #     zero = pred_logits.sum() * 0.0  # keeps gradient graph alive
        #     return zero, {
        #         'total': 0.0, 'cls': 0.0, 'box': 0.0, 'dfl': 0.0, 'n_fg': 0
        #     }

        # ── 6. classification loss (BCE) ───────────────────────────────
        # target for cls: [B, 8400, 1]  (target_scores already 0 for negatives)
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_logits,
            target_scores,
            reduction='none'
        ).sum() / target_scores_sum

        # ── 7. box + dfl loss ──────────────────────────────────────────
        loss_box = pred_dist.sum() * 0.0   # keeps gradient graph alive
        loss_dfl = pred_dist.sum() * 0.0

        if fg_mask.any():
            loss_box, loss_dfl = self.bbox_loss(
                pred_dist,
                pred_bboxes,
                anc_pts_pixel,
                stride_tensor,
                target_bboxes,
                target_scores,
                fg_mask
            )

        # ── 8. combine ─────────────────────────────────────────────────
        total = (self.lambda_cls * loss_cls +
                 self.lambda_box * loss_box +
                 self.lambda_dfl * loss_dfl)

        return total, {
            'total' : total.item(),
            'cls'   : loss_cls.item(),
            'box'   : loss_box.item(),
            'dfl'   : loss_dfl.item(),
            'n_fg'  : int(fg_mask.sum())
        }
        
if __name__ == "__main__":
    pass