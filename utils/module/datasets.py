import torch
import os
import numpy as np
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

class Image_dataset(Dataset):
    transform = v2.Compose([
        v2.ToImage(),
        v2.RandomRotation(45,fill=(255,),expand=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.Resize((512,512),interpolation=Image.NEAREST,antialias=False),
    ])
    transform2 = v2.Compose([
        v2.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.5,hue=0.5),
    ])

    def __init__(self, image_path,mask_path,transform=True):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_list = os.listdir(image_path)
        self.mask_list = os.listdir(mask_path)
        if not transform:
            self.transform = transform = v2.Compose([
                                                v2.ToImage(),
                                                v2.Resize((512,512),interpolation=Image.NEAREST,antialias=False),
                                            ])
        assert len(self.image_list) == len(self.mask_list)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image,mask = self._getitem(idx)
        return image.float(),mask.long()
    
    def _getitem(self,idx):
        image = Image.open(self.image_path + '/' + self.image_list[idx])
        mask = np.load(self.mask_path + '/' + self.image_list[idx].replace('.png','.npy'))
        image = image.convert('RGB')
        mask = Image.fromarray(mask)
        mask = mask.convert('L')
        if self.transform:
            image,mask = self.transform(image,mask)
            if self.transform2:
                image = self.transform2(image)
        mask = self.make_label(mask)
        return image, mask
    
    def make_label(self,mask):
        mask = np.array(mask)
        mask = np.where(mask== 0,0,mask)    # indoor
        mask = np.where(mask== 85,1,mask)   # close
        mask = np.where(mask== 170,2,mask)  # wall
        mask = np.where(mask== 255,3,mask)  # outdoor
        mask = torch.from_numpy(mask)
        return mask
    
    def show_image(self,idx:None) -> torch.Tensor:
        """
            input:
                idx:None -> random idx
            return:
                torch.Tensor -> [C,H,W]
        """
        if idx is None:
            idx = torch.randint(0,len(self),size=(1,)).item()
        image,mask = self._getitem(idx)
        mask_unique = torch.unique(mask)
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        fig1 = ax[0].imshow(image.permute(1,2,0))
        fig2 = ax[1].imshow(mask.permute(1,2,0))
        fig.colorbar(fig1,ax=ax[0])
        fig.colorbar(fig2,ax=ax[1])
        ax[0].set_title('image')
        ax[1].set_title(f'mask_unique:{mask_unique}')
        plt.show()
        return image