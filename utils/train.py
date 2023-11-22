import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm.notebook import tqdm
from utils.dataset import arc_dataset


class model_setting():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self,model:torch.nn.Module,dataset:arc_dataset,epochs:int,batch_size:int,lr:float=1e-3,decay:float=1e-1):
        self.model = model
        self.model.to(self.device)
        self.loss_weights = torch.tensor([1.0,1.0,1.0]).to(self.device)
        self.criterion = Combinedloss(weights=self.loss_weights,alpha=0.5)
        self.optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.2)
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size

    def dataset_split(self,split:list = [0.4,0.1,0.5]):
        train_data,val_data,test_data = random_split(self.dataset,split)
        self.train_loader = DataLoader(train_data,self.batch_size,shuffle=True)
        self.val_loader = DataLoader(val_data,self.batch_size,shuffle=True)
        self.test_loader = DataLoader(test_data,self.batch_size,shuffle=True)
    
    def validate(self):
        self.model.eval()
        total = 0
        correct = 0
        val_loss = 0
        with torch.no_grad():
            for image,mask in self.val_loader:
                image = image.to(self.device)
                mask = mask.to(self.device)
                output = self.model(image)
                _, predicted = torch.max(output.data, 1)
                total += mask.nelement()
                correct += (predicted == mask).sum().item()
                val_loss += self.criterion(output, mask).item()
        accuracy = correct / total
        return accuracy, val_loss / len(self.val_loader)
    
    def train(self,save_path:str=None,print_info:bool=False):
        best_accuracy = 0.0
        for epoch in range(self.epochs):
            train_accuracy = self.train_step(print_info)
            val_accuracy, val_loss = self.validate()
            self.scheduler.step()
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {val_loss}, Accuracy: {val_accuracy}, Train Accuracy: {train_accuracy}')
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.best_model = copy.deepcopy(self.model)
                if save_path:
                    print('save model')
                    torch.save(self.best_model.state_dict(),save_path + f'model_{val_accuracy:.2f}.pth')
    
    def train_step(self,print_info:bool=False):
        model = self.model
        model.train()
        train_loader = self.train_loader
        total_steps = len(train_loader)
        train_accuracy = 0.0
        for step,(image,mask) in enumerate(tqdm(train_loader)):
            image = image.to(self.device)
            mask = mask.to(self.device)
            output = model(image)
            loss = self.criterion(output,mask)
            # combined_loss = Combinedloss(3,0.5)
            # loss = combined_loss(output,mask)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == mask).sum().item()
            accuracy = correct / (mask.size(0) * mask.size(1) * mask.size(2))
            train_accuracy += accuracy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if print_info:
                print(f'Step [{step+1}/{total_steps}], Loss: {loss.item()}, Accuracy: {accuracy}')
        train_accuracy /= total_steps  # 计算平均准确率
        return train_accuracy
    
    def predict(self,image:torch.Tensor = None,mode:str='test'):
        """
            Predict the output for a given image.

            Args:
                image (torch.Tensor): A 4D tensor of shape (batch_size, channels, height, width).
        """
        if mode == 'test':
            data =  self.test_loader.dataset
        elif mode == 'train':
            data = self.train_loader.dataset
        elif mode == 'val':
            data = self.val_loader.dataset
        num = torch.randint(0,len(data),(1,)).item()
        image = data[num][0]
        true_mask = data[num][1]
        image = image.unsqueeze(0)
        image = image.to(self.device)
        output = self.best_model(image)
        _, predicted = torch.max(output.data, 1)
        return image,true_mask,predicted
    
    def show_predict(self,mode:str='test'):
        image,true_mask,predicted = self.predict(mode=mode)
        image = image.squeeze(0)
        image = image.cpu().numpy()
        image = np.transpose(image,(1,2,0))
        predicted = predicted.squeeze(0).cpu().numpy()
        fig,ax = plt.subplots(1,3,figsize=(15,5))
        im1 = ax[0].imshow(image)
        im2 = ax[1].imshow(true_mask)
        im3 = ax[2].imshow(predicted)
        fig.colorbar(im1)


class Combinedloss(nn.Module):

    def __init__(self,weights:torch.Tensor,alpha=0.5):
        super().__init__()
        self.cross_loss = nn.CrossEntropyLoss(weight=weights)
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=1)

    def dice_loss(self,inputs,targets):
        smooth = 1.
        inputs = self.softmax(inputs)
        targets_one_hot = self.one_hot(targets,inputs.shape[1])
        intersect = torch.sum(inputs * targets_one_hot)
        denominator = torch.sum(inputs + targets_one_hot)
        dice_coeff = (2 * intersect) / (inputs.shape[1] * (denominator))
        dice_loss = 1 - dice_coeff
        return dice_loss

    def one_hot(self,targets,num_classes):
        targets_one_hot = F.one_hot(targets,num_classes)
        targets_one_hot = targets_one_hot.permute(0,3,1,2).float()
        targets_one_hot = targets_one_hot.to(targets.device)
        return targets_one_hot
    
    def forward(self,inputs,targets):
        cross_loss = self.cross_loss(inputs,targets)
        dice_loss = self.dice_loss(inputs,targets)
        loss = self.alpha * cross_loss + (1 - self.alpha) * dice_loss
        return loss
    
class Iouloss(nn.Module):

    def __init__(self,eps=1e-7):
        super().__init__()
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self,inputs,targets):
        inputs = self.softmax(inputs)
        targets = self.one_hot(targets,inputs.shape[1])
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + self.eps) / (union + self.eps)
        loss = 1 - IoU
        return loss
    
    def one_hot(self,targets,num_classes):
        targets_one_hot = F.one_hot(targets,num_classes)
        targets_one_hot = targets_one_hot.permute(0,3,1,2).float()
        targets_one_hot = targets_one_hot.to(targets.device)
        return targets_one_hot