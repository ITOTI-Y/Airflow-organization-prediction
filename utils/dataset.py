import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torch import Tensor

class arc_dataset(Dataset):
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32,scale=True),
        transforms.Lambda(lambda x: x / 255),
        transforms.Normalize(mean=[0.5],std=[0.5]),
    ])

    def __init__(self,data_path:str):
        super().__init__()
        self.data_path = data_path
        self.full_data_list = os.listdir(self.data_path)

    def __len__(self):
        return len(self.full_data_list)

    def __getitem__(self,idx):
        full_data = np.load(self.data_path + self.full_data_list[idx])
        image = self._get_wall(full_data[:,:,0])
        mask = self._get_room(full_data[:,:,0])

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask).long()
            mask = mask.squeeze(0)
        return image,mask


    def random_show_data(self,file_name:str=None):
        if not file_name:
            image_path = np.random.choice(self.full_data_list)
        else:
            image_path = self.full_data_list.index(file_name)
        full_data = np.load(self.data_path + image_path)
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        im1 = ax[0].imshow(self._get_wall(full_data[:,:,0]),
        cmap='gray')
        im2 = ax[1].imshow(self._get_room(full_data[:,:,0]),cmap='gray')
        fig.colorbar(im1)
        fig.colorbar(im2)
        return self._get_wall(full_data[:,:,0])

    def _get_wall(self,full_data:np.ndarray):
        wall_data = full_data.copy()
        wall_data = np.where(wall_data == 9,np.random.randint(0,100,wall_data.shape),np.random.randint(240,255,wall_data.shape))
        # wall_data = np.where(wall_data == 9,0,1)
        return wall_data
    
    def _get_room(self,full_data:np.ndarray):
        room_data = full_data.copy()
        room_data = np.where((room_data != 8) & (room_data != 9) & (room_data != 13),2,room_data) # outdoor
        room_data = np.where(room_data == 9,0,room_data) # wall
        room_data = np.where((room_data == 13) | (room_data == 8),1,room_data) # room
        return room_data
    

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