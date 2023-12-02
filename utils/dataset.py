import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torch import Tensor

class Detection_data(Dataset):
    label = ['windows','door']
    label_to_int = {label:i for i,label in enumerate(label)}

    def __init__(self,path:str) -> None:
        super().__init__()
        # 搜索data_path下的所有json
        self.data_path = path
        self.json_list = [f for f in os.listdir(path) if f.endswith('.json')]

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self,idx):
        return self.get_item(idx)
    
    def get_item(self,idx):
        json_file = self.json_list[idx]
        with open(self.data_path + '/' + json_file,'r') as f:
            data = json.load(f)

        img_path = self.data_path + '/' + data['imagePath']
        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []
        for shape in data['shapes']:
            box = shape['points']
            label = shape['label']
            boxes.append(box)
            labels.append(self.label_to_int[label])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return img,boxes,labels
    
    def show_image(self,idx):
        fig,ax = plt.subplots(1,1,figsize=(10,10))

        img,boxes,labels = self.get_item(idx)
        ax.imshow(img)

        for box in boxes:
            rect = patches.Rectangle((box[0][0],box[0][1]), box[1][0]-box[0][0], box[1][1]-box[0][1], linewidth=1, edgecolor='r', facecolor='none')

            ax.add_patch(rect)
        plt.show()

class image_data(Dataset):
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32,scale=True),
        transforms.Resize((512,512),antialias=True),
        # transforms.Normalize(mean=[0.5],std=[0.5]),
    ])

    def __init__(self,data_path:str,original:bool=False):
        super().__init__()
        self.data_path = data_path
        if original:
            self.IMAGE_PATH = self.data_path + 'Ori_images/'
            self.MASK_PATH = self.data_path + 'Ori_masks/'
        else:
            self.IMAGE_PATH = self.data_path + 'Images/'
            self.MASK_PATH = self.data_path + 'Masks/'
        self.images_list = os.listdir(self.IMAGE_PATH)
        self.masks_list = os.listdir(self.MASK_PATH)

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self,idx) -> (Tensor,Tensor):
        image,mask = self._get_image(idx)
        mask = self._make_label(mask)
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask)
            mask = mask.long()
            mask = mask.squeeze(0)
        return image,mask
    
    def _get_image(self,idx,cmap1='L',cmap2='L') ->(Image,Image):
        image = Image.open(self.IMAGE_PATH + self.images_list[idx])
        image = image.convert(cmap1)
        mask = Image.open(self.MASK_PATH + self.masks_list[idx])
        mask = mask.convert(cmap2)
        return image,mask
    
    def _make_label(self,image:Image) -> np.ndarray:
        image = image.resize((512,512))
        image = np.array(image)
        image = np.where(image <= 25,0,image) # 室外，mask中为黑色0
        image = np.where(image > 230,1,image) # 室内，mask中为白色255
        image = np.where((image != 0) & (image != 1),2,image) # 墙壁，mask中为灰色128
        return image
    
    def random_show_image(self) -> None:
        idx = np.random.randint(0,len(self.images_list))
        image,mask = self._get_image(idx,'RGB','L')
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        im1 = ax[0].imshow(image)
        im2 = ax[1].imshow(mask)
        fig.colorbar(im1)
        fig.colorbar(im2)