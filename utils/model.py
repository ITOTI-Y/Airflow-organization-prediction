import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchinfo import summary

class DoubleConv(nn.Module):

    def __init__(self,in_channels,out_channels,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # add dropout layer
            nn.Conv2d(mid_channels,out_channels,kernel_size=5,padding=2),
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
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    """Convolutional layer for output"""

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
    
    def forward(self,x):
        return self.conv(x)


class UNetplusplus(nn.Module):
    num_branchs = 4
    
    def __init__(self,n_channels:int,n_classes:int,bilinear:bool=True):
        """
            n_channels: 输入图片的通道数
            n_classes: 标签的类别数
            bilinear: 是否使用双线性插值进行上采样
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.outc_weight = nn.Parameter(torch.ones(self.num_branchs))

        self.inc = DoubleConv(n_channels,8)
        self.down1 = Down(8,16)
        self.down2 = Down(16,32)
        self.down3 = Down(32,64)
        self.down4 = Down(64,128)

        self.up0_1 = Up(24,24,bilinear)
        self.up0_2 = Up(48,24,bilinear)
        self.up0_3 = Up(60,30,bilinear)
        self.up0_4 = Up(84,84,bilinear)

        self.up1_1 = Up(48,24,bilinear)
        self.up1_2 = Up(72,36,bilinear)
        self.up1_3 = Up(108,54,bilinear)

        self.up2_1 = Up(96,48,bilinear)
        self.up2_2 = Up(144,72,bilinear)

        self.up3_1 = Up(192,96,bilinear)


        self.aoutc1 = OutConv(24,n_classes)
        self.aoutc2 = OutConv(24,n_classes)
        self.aoutc3 = OutConv(30,n_classes)
        self.aoutc4 = OutConv(84,n_classes)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((512,512))
    
    def forward(self,x):

        x0_0 = self.inc(x)
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        x0_1 = self.up0_1(x1_0,x0_0)
        x1_1 = self.up1_1(x2_0,x1_0)
        x2_1 = self.up2_1(x3_0,x2_0)
        x3_1 = self.up3_1(x4_0,x3_0)

        x0_2 = self.up0_2(x1_1,x0_1)
        x1_2 = self.up1_2(x2_1,x1_1)
        x2_2 = self.up2_2(x3_1,x2_1)

        x0_3 = self.up0_3(x1_2,x0_2)
        x1_3 = self.up1_3(x2_2,x1_2)

        x0_4 = self.up0_4(x1_3,x0_3)


        logits1 = self.aoutc1(x0_1)
        logits2 = self.aoutc2(x0_2)
        logits3 = self.aoutc3(x0_3)
        logits4 = self.aoutc4(x0_4)

        logits = 0
        for i in range(self.num_branchs):
            logits += self.outc_weight[i] * eval(f'logits{i+1}')
        # logits = self.outc_weight[0] * logits1 + self.outc_weight[1] * logits2 + self.outc_weight[2] * logits3

        return logits
    
    def summary(self,input:(int,int,int,int) = (4,1,512,512)):
        print(summary(self,input,col_names=["kernel_size", "output_size", "num_params", "mult_adds"],))

#--------------------------------------------------------------------------#

class DetectionHead(nn.Module):
    def __init__(self,num_classes,hidden_dim,num_queries=100):
        super().__init__()
        self.class_embed = torch.nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = torch.nn.Linear(hidden_dim, 4)
        self.query_embed = torch.nn.Embedding(num_queries, hidden_dim)

    def forward(self,x):
        return self.class_embed(x),self.bbox_embed(x)


class DETR(nn.Module):
    def __init__(self,num_classes,hidden_dim,n_heads,num_encoder_layers,num_decoder_layers):
        super().__init__()
        self.backbone = self.detr_backbone()
        self.transformer = self.create_transformer(hidden_dim,n_heads,num_encoder_layers,num_decoder_layers)
        self.head = DetectionHead(num_classes,hidden_dim)

    def detr_backbone(self):
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        return backbone
    
    def create_transformer(self,hidden_dim,n_heads,num_encoder_layers,num_decoder_layers):
        transformer = nn.Transformer(
            d_model = hidden_dim,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        return transformer

    def forward(self,x):
        features = self.backbone(x)
        features = features.flatten(2).permute(2,0,1)

        output = self.transformer(features, features)
        return self.head(output)