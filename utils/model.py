import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self,in_channels,out_channels,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )
    
    def forward(self,x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,in_channels,out_channels,bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = DoubleConv(in_channels,out_channels,in_channels // 2)
        
        else:
            self.up = nn.ConvTranspose2d(in_channels,in_channels // 2,kernel_size=2,stride=2)
            self.conv = DoubleConv(in_channels,out_channels)
        
    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1,[diffX // 2,diffX - diffX // 2,diffY // 2,diffY - diffY // 2])
        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Convolutional layer for output"""

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
    
    def forward(self,x):
        return self.conv(x)



class UNet(nn.Module):

    def __init__(self,n_channels,n_classes,bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,512)
        self.up1 = Up(1024,256,bilinear)
        self.up2 = Up(512,128,bilinear)
        self.up3 = Up(256,64,bilinear)
        self.up4 = Up(128,64,bilinear)
        self.outc = OutConv(64,n_classes)


    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        logits = self.outc(x)
        return logits
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import arc_dataset
    from torch.utils.data import random_split
    from tqdm import tqdm

    def validate_model(model,val_loader,criterion):
        model.eval() # 将模型设置为评估模式
        total = 0
        correct = 0
        val_loss = 0
        with torch.no_grad():
            for image,mask in val_loader:
                image = image.to(device)
                mask = mask.to(device)
                output = model(image)
                _, predicted = torch.max(output.data, 1)
                total += mask.nelement()
                correct += (predicted == mask).sum().item()
                val_loss += criterion(output, mask).item()
        accuracy = correct / total
        return accuracy, val_loss / len(val_loader)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(1,2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    data = arc_dataset('./data/train/full_out/')
    train_data,val_data,test_data = random_split(data,[0.4,0.1,0.5])



    train_loader = DataLoader(train_data,batch_size=4,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=4,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=4,shuffle=True)
    
    epoch_num = 2
    for epoch in range(epoch_num):
        running_accuracy = 0.0
        total_steps = len(train_loader)
        for step,(image,mask) in enumerate(tqdm(train_loader)):
            model.train()
            image = image.to(device)
            mask = mask.to(device)
            output = model(image)
            loss = criterion(output,mask)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == mask).sum().item()
            accuracy = correct / (mask.size(0) * mask.size(1) * mask.size(2))
            running_accuracy += accuracy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Step [{step+1}/{total_steps}], Loss: {loss.item()}, Accuracy: {running_accuracy/(step+1)}')
        val_accuracy, val_loss = validate_model(model, val_loader, criterion)
        print(f'Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')
    torch.save(model.state_dict(), 'best_model.pth')
    # for name, param in model.named_parameters():
    #     print(f'Parameter {name} has data type {param.dtype}')