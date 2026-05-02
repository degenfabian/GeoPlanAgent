"""Shared binary-mask segmentation metrics for the SAM3 training + eval k-fold
scripts (torch tensors). The numpy pixel-IoU used by the segmentation ablations
lives separately in ablations/utils.py:iou_score (different array library)."""

import torch


def binary_mask_metrics(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> dict:
    """IoU + precision + recall + F1 + Dice from two binary masks of equal shape.

    Accepts float (0/1) or bool masks. All metrics come from one TP/FP/FN triple
    so they stay internally consistent (IoU and F1 can't disagree on a case).
    """
    pred = pred_bin.bool()
    gt = gt_bin.bool()
    tp = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    pred_sum = pred.sum().item()
    gt_sum = gt.sum().item()
    fp = pred_sum - tp
    fn = gt_sum - tp
    iou = tp / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    dice = 2 * tp / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1, "dice": dice}
