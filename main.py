from utils.module.UNet_v2 import *
from utils.module.datasets import Image_dataset
from utils.module.train import Train,Optuna_train

dataset = Image_dataset(image_path=r"data\train_image",mask_path=r"data\train_mask")
n_classes = 4
pretrained_path = r'C:\Users\Administrator\Desktop\Code\U-Net_v2-master\pvt_v2_b2.pth'
model = UNetV2(n_classes=n_classes, deep_supervision=True ,pretrained_path=pretrained_path)
train = Train(dataset=dataset,model=model,lr=1e-6,weight_decay=1e-4,alpha=0.01,batch_size=1,weight=torch.tensor([0.6615,0.4084,0.4065,1.2351]))
train.load_model(r'result\best-0.96.pth')
train.train(epochs=1000,save_path=r'result')
# opt_train = Optuna_train(dataset=dataset,model=model,path=r'./result')
# opt_train.opt(100)