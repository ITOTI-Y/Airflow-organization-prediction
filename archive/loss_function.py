import torch.nn as nn
import torch
import torch.nn.functional as F


class Combinedloss(nn.Module):

    def __init__(self, weights: torch.Tensor, alpha=0.5):
        super().__init__()
        self.cross_loss = nn.CrossEntropyLoss(weight=weights)
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=1)

    def dice_loss(self, inputs, targets):
        smooth = 1.
        inputs = self.softmax(inputs)
        targets_one_hot = self.one_hot(targets, inputs.shape[1])
        intersect = torch.sum(inputs * targets_one_hot)
        denominator = torch.sum(inputs + targets_one_hot)
        dice_coeff = (2 * intersect) / (inputs.shape[1] * (denominator))
        dice_loss = 1 - dice_coeff
        return dice_loss

    def one_hot(self, targets, num_classes):
        targets_one_hot = F.one_hot(targets, num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        targets_one_hot = targets_one_hot.to(targets.device)
        return targets_one_hot

    def forward(self, inputs, targets):
        cross_loss = self.cross_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        loss = self.alpha * cross_loss + (1 - self.alpha) * dice_loss
        return loss


class Iouloss(nn.Module):

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, targets):
        inputs = self.softmax(inputs)
        targets = self.one_hot(targets, inputs.shape[1])
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + self.eps) / (union + self.eps)
        loss = 1 - IoU
        return loss

    def one_hot(self, targets, num_classes):
        targets_one_hot = F.one_hot(targets, num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        targets_one_hot = targets_one_hot.to(targets.device)
        return targets_one_hot

class HungarianLoss(nn.Module):
    def __init__(self, num_classes, weight_dict={'cls': 1, 'bbox': 5}, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.bbox_criterion = nn.L1Loss(reduction='none')
        self.cls_criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # [bs*num_queries, num_classes]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [bs*num_queries, 4]

        tgt_ids = torch.cat([t['labels'] for t in targets])  # [sum(num_objs)]
        tgt_bbox = torch.cat([t['boxes'] for t in targets])  # [sum(num_objs), 4]

        # Compute the cost matrix
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = self.bbox_criterion(out_bbox, tgt_bbox)
        cost_bbox = cost_bbox.sum(-1)
        cost_matrix = self.weight_dict['cls'] * cost_class + self.weight_dict['bbox'] * cost_bbox
        cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()

        # Compute the optimal assignment
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(1, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        indices = [(i.to(out_prob.device), j.to(out_prob.device)) for i, j in indices]

        # Compute the losses
        cls_loss = self.cls_criterion(out_prob, tgt_ids)
        bbox_loss = self.bbox_criterion(out_bbox, tgt_bbox)
        cls_loss = sum([cls_loss[i, j].mean() for (i, j) in indices])
        bbox_loss = sum([bbox_loss[i, j].mean() for (i, j) in indices])
        loss = cls_loss + bbox_loss

        return loss