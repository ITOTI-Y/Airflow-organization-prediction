import torch
import copy
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm.notebook import tqdm
from utils.dataset import arc_dataset


class train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self,model:torch.nn.Module,dataset:arc_dataset,epochs:int,batch_size:int):
        self.model = model
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size

    def dataset_split(self,split:list = [0.4,0.1,0.5]):
        train_data,val_data,test_data = random_split(self.dataset,split)
        self.train_loader = DataLoader(train_data,batch_size=4,shuffle=True)
        self.val_loader = DataLoader(val_data,batch_size=4,shuffle=True)
        self.test_loader = DataLoader(test_data,batch_size=4,shuffle=True)
    
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
    
    def train(self,save_path:str=None):

        for epoch in range(self.epochs):
            train_accuracy = 0.0
            total_steps = len(self.train_loader)
            for step,(image,mask) in enumerate(tqdm(self.train_loader)):
                self.model.train()
                image = image.to(self.device)
                mask = mask.to(self.device)
                output = self.model(image)
                loss = self.criterion(output,mask)
                _, predicted = torch.max(output.data, 1) # 返回每一行中最大值的那个元素及索引
                correct = (predicted == mask).sum().item()
                accuracy = correct / (mask.size(0) * mask.size(1) * mask.size(2))
                train_accuracy += accuracy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f'Step [{step+1}/{total_steps}], Loss: {loss.item()}, Accuracy: {train_accuracy/(step+1)}')
            val_accuracy, val_loss = self.validate()
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {val_loss}, Accuracy: {val_accuracy}')
        if save_path:
            torch.save(self.model.state_dict(),save_path)
    
    def predict(self,image:np.ndarray,load_path:str=None):
        """
            input:
                image: torch.Tensor['w','h']
                load_path: str
        """
        if load_path:
            model = copy.deepcopy(self.model)
            model.load_state_dict(torch.load(load_path))
            model.eval()
        else:
            model = self.model
            model.eval() # 将模型设置为评估模式
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        image = image.to(self.device)
        _, predicted = torch.max(model(image),1)

        return model
    
    def show_predict(self,image:np.ndarray,load_path:str=None):
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        im1 = ax[0].imshow(image,cmap='gray')
        im2 = ax[1].imshow(self.predict(image,load_path).cpu().squeeze(0),cmap='gray')
        fig.colorbar(im1)
        fig.colorbar(im2)