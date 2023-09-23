import torch
import torch.nn.functional as F
import numpy as np


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask_, mask_, smooth=1e-20, n_classes=None):
    batch_mean_iou = []
    for i in range(len(pred_mask_)):
        pred_mask = pred_mask_[i]
        mask = mask_[i]
        with torch.no_grad():
            pred_mask = F.softmax(pred_mask, dim=0)
            pred_mask = torch.argmax(pred_mask, dim=0)
            pred_mask = pred_mask.contiguous().view(-1)
            mask = mask.contiguous().view(-1)

            iou_per_class = []
            for clas in range(0, n_classes): #loop per pixel class
                true_class = pred_mask == clas
                true_label = mask == clas
                if true_label.long().sum().item() == 0: #no exist label in this loop
                    iou_per_class.append(np.nan)
                else:
                    intersect = torch.logical_and(true_class, true_label).sum().float().item()
                    union = torch.logical_or(true_class, true_label).sum().float().item()

                    iou = (intersect + smooth) / (union + smooth)
                    iou_per_class.append(iou)
        batch_mean_iou.append(np.nanmean(iou_per_class))
    return np.mean(batch_mean_iou)
