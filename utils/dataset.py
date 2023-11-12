import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image

class arc_dataset(Dataset):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float() / 13),
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
        ax[0].imshow(self._get_wall(full_data[:,:,0]),
        cmap='gray')
        ax[1].imshow(self._get_room(full_data[:,:,0]),cmap='gray')

    def _get_wall(self,full_data:np.ndarray):
        wall_data = full_data.copy()
        wall_data = np.where(wall_data == 9,0,np.random.randint(1,255,wall_data.shape))
        return wall_data
    
    def _get_room(self,full_data:np.ndarray):
        room_data = full_data.copy()
        room_data = np.where((room_data == 9) | (room_data == 13),0,1) # 如果是9或者13就是背景，否则就是房间
        return room_data
    

if __name__ == '__main__':
    data_path = './data/train'
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = arc_dataset(data_path = './data/train',transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)