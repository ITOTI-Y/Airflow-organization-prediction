import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class Postprocessing:
    name_dic = {'rooms':{'value':0,'color':[127,201,127],'mask':None},
                'closes':{'value':1,'color':[190,174,212],'mask':None},
                'walls':{'value':2,'color':[253,192,134],'mask':None},
                'outdoor':{'value':3,'color':[56,108,176],'mask':None}}

    def __init__(self,image=None,mask=None):
        """
            input:
                image:torch.tensor -> [C,H,W]
                mask:torch.tensor -> [B,C,H,W]
        """
        self.image = image
        self.mask = mask
        self.pre_processing()
        self.segment_mask()

    def pre_processing(self):
        """
            input:
                image:torch.tensor -> [C,H,W]
                mask:torch.tensor -> [B,C,H,W]
            return:
                image:np.array -> [H,W,C]
                mask:np.array -> [H,W]
        """
        image = self.image.permute(1,2,0).cpu().numpy()
        mask = self.mask[0].permute(1,2,0).cpu().numpy()
        self.image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.mask = mask.squeeze()

    def segment_mask(self,show=False,**kwargs):
        """
            input:
                image:np.array -> [H,W,C]
                mask:np.array -> [H,W]
                color:bool -> True or False
            return:
                mask:np.array -> [H,W] uint8
        """
        scale = kwargs.get('scale',1)

        mask = self.mask
        name_dic = self.name_dic
        for name in name_dic.keys():
            name_dic[name]['mask'] = np.where(mask==name_dic[name]['value'],1,0).astype(np.uint8)
        name_dic['closes']['mask'] = self.enhance_closes(scale=scale)
        name_dic['outdoor']['mask'] = self.open_opereation(self.name_dic['outdoor']['mask'])

        if show:
            self.show_mask([name_dic[name]['mask'] for name in name_dic.keys()],**kwargs)
            plt.show()
        self.name_dic = name_dic

    def show_mask(self,mask_list:list=None,**kwargs):
        """
            input:
                mask_list:list -> [[H,W],[H,W],[H,W]]
                color:bool -> True or False
        """
        length = len(mask_list)
        fig,ax = plt.subplots(1,length,figsize=(length*5,5))
        color = kwargs.get('color',False)
        alpha = kwargs.get('alpha',1)
        if color:
            for i in range(length):
                name = list(self.name_dic.keys())[i]
                mask = mask_list[i]
                mask = np.repeat(mask[:,:,np.newaxis],3,axis=2)
                mask = mask * np.array(self.name_dic[name]['color'])
                mask_list[i] = mask
        if length == 1:
            ax.imshow(mask_list[0],alpha = alpha)
        else:
            for i in range(length):
                ax[i].imshow(mask_list[i],alpha = alpha)
        return ax

    def close_opereation(self,mask:np.array,show=False,kernel_size=5,max_iter=20):
        """
            description:
                connect the broken parts and fill the small holes
            input:
                mask:np.array -> [H,W]
            return:
                mask:np.array -> [H,W]
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
        change_mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        temp_mask = np.zeros_like(mask)
        for i in range(max_iter):
            temp_mask = cv2.morphologyEx(change_mask,cv2.MORPH_CLOSE,kernel)
            if np.all(change_mask == temp_mask):
                break
            else:
                change_mask = temp_mask
        if show:
            self.show_mask([mask,change_mask])
            plt.show()
        return change_mask
    
    def open_opereation(self,mask:np.array,show=False,kernel_size=5,max_iter=20):
        """
            description:
                remove the small islands, keep the large islands
            input:
                mask:np.array -> [H,W]
            return:
                mask:np.array -> [H,W]
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
        change_mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        temp_mask = np.zeros_like(mask)
        for i in range(max_iter):
            temp_mask = cv2.morphologyEx(change_mask,cv2.MORPH_OPEN,kernel)
            if np.all(change_mask == temp_mask):
                break
            else:
                change_mask = temp_mask
        if show:
            self.show_mask([mask,change_mask])
            plt.show()
        return change_mask
    
    def split_islands(self,mask:np.array,show=False,kernel_size=5):
        """
            input:
                mask:np.array -> [H,W]
            return:
                mask:np.array -> [H,W]
        """
        change_mask = self.open_opereation(mask,kernel_size=kernel_size)
        change_mask = cv2.connectedComponents(change_mask)[1]
        if show:
            self.show_mask([mask,change_mask])
        plt.show()
        return change_mask
    
    def label_island(self,show=False):
        """
            return:
                position:dict -> {int:[x,y]}
        """
        island = self.split_islands(self.name_dic['rooms']['mask'])
        index = np.unique(island)
        position = {}
        for i in range(0,len(index)):
            single_island = island == index[i]
            density = gaussian_filter(single_island.astype(float), sigma=3)
            y_max, x_max = np.unravel_index(np.argmax(density), density.shape)
            position[index[i]] = [x_max, y_max]
        if show:
            self.show_mask([island],alpha=1)
            plt.imshow(np.where(self.name_dic['closes']['mask']!=0,1,0),alpha=0.4,cmap='gray')
            plt.imshow(np.where(self.name_dic['outdoor']['mask']!=0,1,0),alpha=0.4,cmap='gray')
            for i in position.keys():
                plt.plot(position[i][0],position[i][1],'o',color='r')
                plt.text(position[i][0],position[i][1],str(i),color='black',ha='center',va='center')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(r'./result/picture/label_island.jpg')
            plt.show()
        return position
    
    def find_connected_islands(self,**kwargs):
        show = kwargs.get('show',False)
        bridge_mask = self.name_dic['closes']['mask']
        island_mask = self.split_islands(self.name_dic['rooms']['mask'])
        outdoor_mask = self.name_dic['outdoor']['mask'] * (np.max(island_mask+1))
        position = self.label_island(show=show)

        connections = {int(i):{'position':position[i],'connect':set()} for i in position.keys()}

        for i in range(1,np.max(bridge_mask)+1):
            close_mask = np.where(bridge_mask==i,1,0)
            connect_list = np.unique((island_mask+outdoor_mask) * close_mask)[1:]
            connect_list = np.where(connect_list == np.max(island_mask)+1,0,connect_list)
            if connect_list.size <= 1:
                continue
            connections[connect_list[0]]['connect'].add(connect_list[1])
            connections[connect_list[1]]['connect'].add(connect_list[0])

        return connections
    
    def enhance_closes(self,scale = 1.2,show = False):
        closes = self.split_islands(self.name_dic['closes']['mask'],show=show,kernel_size=1)
        result = np.zeros_like(closes)
        eps = 1e-6
        for i in range(1,closes.max()+1):
            close = np.transpose(np.nonzero(np.where(closes == i,1,0).T))
            center, (width, height), angle = cv2.minAreaRect(close)
            if width < height:
                if width < eps:
                    continue
                else:
                    width *= scale*(height/(1.5*width+eps))
            else:
                if height < eps:
                    continue
                else:
                    height *= scale*(width/(1.5*height+eps))

            new_rect = (center, (width, height), angle)
            box = cv2.boxPoints(new_rect)
            box = np.intp(box)
            result += cv2.drawContours(np.zeros_like(closes), [box], 0, i, -1)
        if show:
            self.show_mask([closes,result])
        return result.astype(np.uint8)