import torch
import torch.nn as nn
import torch.nn.functional as F

class Combined_loss(nn.Module):
    def __init__(self, alpha=0.5,device='cpu',weight=torch.tensor([1.,1.,1.,1.])):
        super().__init__()
        self.alpha = alpha
        # self.weight = torch.tensor([0.1,2.0,0.1,0.1]).to(device)
        self.weight = weight.to(device)
        self.cross_loss = CrossEntropyLoss(weight=self.weight)
        self.dice_loss = Dice_loss(weight=self.weight)
        self.focal_loss = Focal_Loss(alpha=0.75,weight=self.weight)
        

    def forward(self, outputs, targets):
        # cross_loss = self.cross_loss(inputs, targets)
        losses = 0
        if not isinstance(outputs,list):
            outputs = [outputs]
        targets = targets.float()
        for output in outputs:
            target_resize = F.interpolate(targets, size=output.size()[2:], mode='nearest')
            target_resize = target_resize.int().long()
            dice_loss = self.dice_loss(output, target_resize)
            focal_loss = self.focal_loss(output, target_resize)
            loss = self.alpha*focal_loss + (1-self.alpha) * dice_loss
            losses += loss
        return losses


class Dice_loss(nn.Module):
    def __init__(self,weight=torch.tensor([1.,1.,1.,1.])):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.weight = weight
    
    def one_hot(self,targets):
        targets_one_hot = F.one_hot(targets)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        targets_one_hot = targets_one_hot.to(targets.device)
        return targets_one_hot
    
    def cal_loss(self,inputs,targets_one_hot,weight):
        eps = 1e-6
        smooth = 1.
        loss = 0
        for i in range(inputs.size(1)):
            mask = targets_one_hot[:, i, :, :]
            image = inputs[:, i, :, :]
            intersect = torch.sum(image * mask)
            denominator = torch.sum(image + mask)
            dice_coeff = (2 * intersect + smooth) / (denominator + smooth)
            loss += (1 - dice_coeff) * weight[i]
        return loss

    def forward(self,inputs:torch.Tensor,targets:torch.Tensor):
        """
            inputs: [batch,channel,int,int]
            targets: [batch,channel,int,int]
        """
        inputs = self.softmax(inputs)
        targets = targets.squeeze(1)
        targets_one_hot = self.one_hot(targets)
        outputs = self.cal_loss(inputs,targets_one_hot,weight=self.weight)
        return outputs

class CrossEntropyLoss(nn.Module):
    def __init__(self,weight=None):
        super().__init__()
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,inputs,targets):
        targets = targets.squeeze(1)
        loss = self.cross_loss(inputs,targets)
        return loss
    

class Focal_Loss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2.0,device='cpu',weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.cross_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self,inputs,targets):
        targets = targets.squeeze(1)
        c_loss = self.cross_loss(inputs,targets)

        pt = torch.exp(-c_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * c_loss
        return f_loss