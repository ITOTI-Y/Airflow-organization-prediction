import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50,ResNet50_Weights

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.Dropout2d(0.3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
    def forward(self, x):
        return self.body(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            )

    def forward(self, x1, x2):
        weights = torch.randn(x1.size(1),x1.size(1),2,2).to(x1.device)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.body(x)
        return x
    
class InConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)

class Main(nn.Module):
    def __init__(self,num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.inc = InConvBlock(3, 16)
        self.down1 = DownBlock(16, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 256)

        self.up01 = UpBlock(48, 24)
        self.up11 = UpBlock(96, 48)
        self.up21 = UpBlock(192, 96)
        self.up31 = UpBlock(384, 192)

        self.up02 = UpBlock(72, 36)
        self.up12 = UpBlock(144, 72)
        self.up22 = UpBlock(288, 144)

        self.up03 = UpBlock(108, 54)
        self.up13 = UpBlock(216, 108)

        self.up04 = UpBlock(162, 81)

        self.outc = nn.Sequential(
            nn.Conv2d(self.fnn_nums(),self.num_classes, kernel_size=7, stride=1, padding=3),
        )

    def fnn_nums(self):
        num = 0
        for i in range(1,5):
            num += eval(f'self.up0{i}.out_channels')
        return num
    
    def predict(self,image:torch.tensor):
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            output = self(image)
            output = F.softmax(output,dim=1)
            output = torch.argmax(output,dim=1,keepdim=True)
            return output
    
    def forward(self, x):
        x00 = self.inc(x)
        x10 = self.down1(x00)
        x20 = self.down2(x10)
        x30 = self.down3(x20)
        x40 = self.down4(x30)

        x01 = self.up01(x10, x00)
        x11 = self.up11(x20, x10)
        x21 = self.up21(x30, x20)
        x31 = self.up31(x40, x30)

        x02 = self.up02(x11, x01)
        x12 = self.up12(x21, x11)
        x22 = self.up22(x31, x21)
        
        x03 = self.up03(x12, x02)
        x13 = self.up13(x22, x12)

        x04 = self.up04(x13, x03)

        x = torch.cat([x01, x02, x03, x04], dim=1)
        x = self.outc(x)
        return x