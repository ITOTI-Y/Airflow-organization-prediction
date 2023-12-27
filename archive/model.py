import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchinfo import summary


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # add dropout layer
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //
                   2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """Convolutional layer for output"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetplusplus(nn.Module):
    """

    属性:
        num_branchs (int): 分支数量
        n_channels (int): 输入图片的通道数
        n_classes (int): 标签的类别数
        bilinear (bool): 是否使用双线性插值进行上采样
        outc_weight (torch.Tensor): 输出权重
        inc, down1, down2, down3, down4, up0_1, up0_2, up0_3, up0_4, up1_1, up1_2, up1_3, up2_1, up2_2, up3_1: UNet++的各层
        aoutc1, aoutc2, aoutc3, aoutc4: 输出层
        adaptive_pool (torch.nn.AdaptiveAvgPool2d): 自适应平均池化层

    方法:
        forward(x): 定义模型的前向传播
        summary(input): 打印模型的概要信息
    """
    num_branchs = 4

    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True):
        """
        初始化UNet++模型。

        参数:
            n_channels (int): 输入图片的通道数
            n_classes (int): 标签的类别数
            bilinear (bool): 是否使用双线性插值进行上采样
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.outc_weight = nn.Parameter(torch.ones(self.num_branchs))

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)

        self.up0_1 = Up(24, 24, bilinear)
        self.up0_2 = Up(48, 24, bilinear)
        self.up0_3 = Up(60, 30, bilinear)
        self.up0_4 = Up(84, 84, bilinear)

        self.up1_1 = Up(48, 24, bilinear)
        self.up1_2 = Up(72, 36, bilinear)
        self.up1_3 = Up(108, 54, bilinear)

        self.up2_1 = Up(96, 48, bilinear)
        self.up2_2 = Up(144, 72, bilinear)

        self.up3_1 = Up(192, 96, bilinear)

        self.aoutc1 = OutConv(24, n_classes)
        self.aoutc2 = OutConv(24, n_classes)
        self.aoutc3 = OutConv(30, n_classes)
        self.aoutc4 = OutConv(84, n_classes)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((512, 512))

    def forward(self, x):
        """
        定义模型的前向传播。

        参数:
            x (torch.Tensor): 输入数据

        返回:
            logits (torch.Tensor): 模型的输出
        """
        x0_0 = self.inc(x)
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        x0_1 = self.up0_1(x1_0, x0_0)
        x1_1 = self.up1_1(x2_0, x1_0)
        x2_1 = self.up2_1(x3_0, x2_0)
        x3_1 = self.up3_1(x4_0, x3_0)

        x0_2 = self.up0_2(x1_1, x0_1)
        x1_2 = self.up1_2(x2_1, x1_1)
        x2_2 = self.up2_2(x3_1, x2_1)

        x0_3 = self.up0_3(x1_2, x0_2)
        x1_3 = self.up1_3(x2_2, x1_2)

        x0_4 = self.up0_4(x1_3, x0_3)

        logits1 = self.aoutc1(x0_1)
        logits2 = self.aoutc2(x0_2)
        logits3 = self.aoutc3(x0_3)
        logits4 = self.aoutc4(x0_4)

        logits = 0
        for i in range(self.num_branchs):
            logits += self.outc_weight[i] * eval(f'logits{i+1}')

        return logits

    def summary(self, input: (int, int, int, int) = (4, 1, 512, 512)):
        """
        打印模型的概要信息。

        参数:
            input (tuple): 输入数据的形状，默认为(4,1,512,512)

        返回:
            None
        """
        print(summary(self, input, col_names=[
              "kernel_size", "output_size", "num_params", "mult_adds"],))

# --------------------------------------------------------------------------#


class DetectionHead(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_queries=100):
        super().__init__()
        self.class_embed = torch.nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = torch.nn.Linear(hidden_dim, 4)
        self.query_embed = torch.nn.Embedding(num_queries, hidden_dim)

    def forward(self, x):
        return self.class_embed(x), self.bbox_embed(x)


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, n_heads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.backbone = self.detr_backbone()
        self.transformer = self.create_transformer(
            hidden_dim, n_heads, num_encoder_layers, num_decoder_layers)
        self.head = DetectionHead(num_classes, hidden_dim)
        self.linear = nn.Linear(2048, hidden_dim)

    def detr_backbone(self):
        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        return backbone

    def create_transformer(self, hidden_dim, n_heads, num_encoder_layers, num_decoder_layers):
        transformer = Transformer(
            num_layers=num_encoder_layers,
            d_model=hidden_dim,
            num_heads=n_heads,
            ddf=hidden_dim,
            input_vocab_szie=hidden_dim,
            target_vovab_size=hidden_dim,
            pe_input=1000,
            pe_target=1000,
            rate=0.1
        )
        return transformer

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(2).permute(2, 0, 1)
        features = self.linear(features)
        output = self.transformer(features, features)
        output = self.head(output)
        return output

# --------------------------------------------------------------------------#


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        x = x.transpose(1, 2)
        return x

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_logits = torch.matmul(
            q, k.transpose(-1, -2) / torch.sqrt(torch.tensor(self.depth,dtype=torch.float32)))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        output = self.dense(output)

        return output, attention_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self,x,mask):
        attn_output,_ = self.mha(x,x,x,mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x+attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1+ffn_output)

        return out2
    
class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,dff,rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model,num_heads)
        self.mha2 = MultiHeadAttention(d_model,num_heads)

        self.ffn = FeedForward(d_model,dff)
        
        self.layernorm1 = nn.LayerNorm(d_model,eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model,eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model,eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self,x,enc_output,look_ahead_mask,padding_mask):
        attn1, attn_weights_block1 = self.mha1(x,x,x,look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1+x)

        attn2, attn_weights_block2 = self.mha2(enc_output,enc_output,out1,padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2+out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output+out2)

        return out3, attn_weights_block1, attn_weights_block2
    
class Transformer(nn.Module):
    def __init__(self,num_layers,d_model,num_heads,ddf,input_vocab_szie,
                 target_vovab_size,pe_input,pe_target,rate=0.1):
        super().__init__()

        self.encoder = EncoderLayer(d_model,num_heads,ddf,rate)
        self.decoder = DecoderLayer(d_model,num_heads,ddf,rate)

        self.final_layer = nn.Linear(d_model,target_vovab_size)

    def forward(self,inp,tar):
        enc_padding_mask,look_ahead_mask,dec_padding_mask = self.create_masks(inp,tar)
        enc_output = self.encoder(inp,enc_padding_mask)

        dec_output,_,_ = self.decoder(tar,enc_output,look_ahead_mask,dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output
    
    def create_padding_mask(self,inp,pad_token_id=0):
        mask = torch.eq(inp,pad_token_id).float()
        mask = mask.unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_look_ahead_mask(self,size):
        mask = 1-torch.tril(torch.ones((1,size,size)))
        return mask
    
    def create_masks(self,inp,tar,pad_token_id=0):
        enc_padding_mask = self.create_padding_mask(inp,pad_token_id)
        dec_padding_mask = self.create_padding_mask(inp,pad_token_id)

        dec_inp_padding_mask = self.create_padding_mask(tar,pad_token_id)
        look_ahead_mask = self.create_look_ahead_mask(tar.size(1))
        combined_mask = torch.max(dec_inp_padding_mask,look_ahead_mask)

        return enc_padding_mask,combined_mask,dec_padding_mask
    