# src/loss_metrics.py
import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        preds = torch.sigmoid(preds)
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        intersection = (preds_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            preds_flat.sum() + targets_flat.sum() + self.smooth
        )
        return bce_loss + dice_loss

def dice_coef(preds, targets, smooth=1.0):
    preds = torch.sigmoid(preds) > 0.5
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def iou_score(preds, targets, smooth=1.0):
    preds = torch.sigmoid(preds) > 0.5
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)
