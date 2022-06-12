## sem --> depth map 
## image to image (픽셀 별 rmse 계산)
# 1. 이미지 데이터셋 로드
# 2. 학습 모델 준비 및 학습
# 3. 평가 
#%%
import matplotlib. pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from PIL import Image
import natsort
import os

#%%
#######################################
##custom dataset 및 loader 준비
#######################################
#%%
def make_dataset(mode):
    depth_img = [f for f in natsort.natsorted(os.listdir('data/'+ mode + '/Depth'))]
    sem_img = [f for f in natsort.natsorted(os.listdir('data/'+ mode + '/SEM'))]
    depth_expand = []
    for i in range(len(depth_img)):
        for j in range(4):
            depth_expand.append(depth_img[i])
    return depth_expand, sem_img   

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, mode):
        self.depth_expand, self.sem_img = make_dataset(mode)
        self.sem_dir = 'data/'+ mode + '/SEM/'
        self.depth_dir = 'data/'+ mode + '/Depth/'
        assert(len(self.depth_expand) == len(self.sem_img))

        
    def __len__(self):
        return len(self.depth_expand)
    
    def __getitem__(self, idx):
        
        sem = Image.open(self.sem_dir + self.sem_img[idx]).convert('L')
        depth = Image.open(self.depth_dir + self.depth_expand[idx]).convert('L')
        tf = transforms.ToTensor()
        tnsr_sem = tf(sem)
        tnsr_depth = tf(depth)
        return tnsr_sem, tnsr_depth
    

train_dataset = CustomDataset("Train")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

valid_dataset = CustomDataset("Validation")
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')
#######################################
## main model 준비
#######################################
#%%
def conv_block(in_dim,out_dim,act_fn, k, s):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=k, stride=s, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
        # nn.Conv2d(out_dim,out_dim, kernel_size=(3,3), stride=1, padding=1),
        # nn.BatchNorm2d(out_dim),
    )
    return model   

def maxpool(stride):
    pool = nn.MaxPool2d(kernel_size=(2,2), stride=stride, padding=0)
    return pool

def conv_trans_block(in_dim,out_dim,act_fn, k, s, p , o):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=k, stride=s, padding=p, output_padding=o),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

  
#%%
class Unet(nn.Module):
    def __init__(self,in_dim,out_dim,num_filter):
        super(Unet,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        print("\n------Initiating mini U-Net------\n")

        self.down_1 = conv_block(self.in_dim,self.num_filter,act_fn, 3, 1)
        self.pool_1 = maxpool((3,3))
        self.down_2 = conv_block(self.num_filter*1,self.num_filter*2,act_fn , 3, 1)
        self.pool_2 = maxpool((2,3))
        
        self.bridge = conv_block(self.num_filter*2,self.num_filter*4,act_fn, 3, 1)
    
        self.trans_3 = conv_trans_block(self.num_filter*4,self.num_filter*2,act_fn, 3, (2,3), (1,0), (1,0))
        self.up_3 = conv_block(self.num_filter*4,self.num_filter*2,act_fn, 3, 1)
        self.trans_4 = conv_trans_block(self.num_filter*2, self.num_filter*1, act_fn, 3, 3, 0, 0)
        self.up_4 = conv_block(self.num_filter*2, self.num_filter*1, act_fn, 3, 1)
        self.up_5 = conv_block(self.num_filter*1, self.out_dim, act_fn, 3, 1)
        
    def forward(self, x): # torch.Size([128, 1, 66, 45])
        down_1 = self.down_1(x)  # torch.Size([128, 64, 66, 45])
        pool_1 = self.pool_1(down_1) # torch.Size([128, 64, 22, 15])
        down_2 = self.down_2(pool_1) #  torch.Size([128, 128, 22, 15])
        pool_2 = self.pool_2(down_2) # torch.Size([128, 128, 11, 5])

        bridge = self.bridge(pool_2) # torch.Size([128, 256, 11, 5])
        
        trans_3 = self.trans_3(bridge) # torch.Size([128, 128, 22, 15])
        concat_3 = torch.cat([trans_3,down_2],dim=1) #  torch.Size([128, 256, 22, 15])
        up_3 = self.up_3(concat_3) # torch.Size([128, 128, 22, 15])
        trans_4 = self.trans_4(up_3) # trans_4: torch.Size([128, 64, 66, 45])
        concat_4 = torch.cat([trans_4,down_1],dim=1) # concat_4: torch.Size([128, 128, 66, 45])
        up_4 = self.up_4(concat_4) #torch.Size([128, 64, 66, 45])
        up_5 = self.up_5(up_4) # torch.Size([128, 1, 66, 45])
        return up_5
#%%
#######################################
## train & validation
#######################################
in_dim = 1
out_dim = 1
num_filters = 64


net = Unet(in_dim=in_dim,out_dim=out_dim,num_filter=num_filters).to(device) 
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=5*1e-3)


# hyper-parameters
num_epochs = 10
num_batches = len(train_dataloader)

trn_loss_list = []
val_loss_list = []
for epoch in range(num_epochs):
    trn_loss = 0.0
    for i, data in enumerate(train_dataloader):
        x, sample = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        model_output = net(x)
        loss = criterion(model_output, sample)
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
        del loss
        del model_output
        
        # 학습과정 출력
        if (i+1) % 100 == 0: # every 100 mini-batches
            with torch.no_grad(): # very very very very important!!!
                net.eval()
                val_loss = 0.0
                for j, val in enumerate(valid_dataloader):
                    val_x, val_sample = val[0].to(device), val[1].to(device) 
                    val_output = net(val_x)
                    v_loss = criterion(val_output, val_sample)
                    val_loss += v_loss
                       
            print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f}".format(
                epoch+1, num_epochs, i+1,num_batches,  trn_loss / 100, val_loss / len(valid_dataloader)
            ))            
            
            trn_loss_list.append(trn_loss/100)
            val_loss_list.append(val_loss/len(valid_dataloader))
            trn_loss = 0.0
print('Finished Training')
#%%
#######################################
## model save & load
#######################################
torch.save(net, 'model/unet.pt')
#%%
net = torch.load('model/unet.pt')

#######################################
## inference & image out
#######################################
# %%
img = Image.open('data/Train/SEM/20201001_202940_NE142400C_RAE01_1_S01_M0005-01MS_3_itr1.png').convert('L')
# img = Image.open('data/Test/SEM/20210304_165053_NE142400C_RAA26_1_S01_M0304-01MS_0_itr0.png').convert('L')

tf = transforms.ToTensor()
img_t = tf(img)
img_t.unsqueeze(dim=0)
# print(img_t.shape)

#%%
y = net(img_t.unsqueeze(dim=0).to(device))
tf_rev = transforms.ToPILImage()
img_r = tf_rev(y.squeeze(dim=0).cpu())

print(img_r)
plt.imshow(img_r, 'gray')
plt.show()

#%%
#######################################
## Test 폴더에 5000장 이미지 저장
#######################################

sem_test = [f for f in natsort.natsorted(os.listdir('data/Test/SEM'))]
tf = transforms.ToTensor()
tr = transforms.ToPILImage()
for i in sem_test: 
    sem_test_img = Image.open('data/Test/SEM/' + '{}'.format(i)).convert('L')
    input = tf(sem_test_img).unsqueeze(dim=0)
    output = net(input.to(device))
    depth_test = tr(output.squeeze(dim=0).cpu())
    depth_test.save('data/Test/Depth/{}'.format(i), 'png')

#%%
# data/Train/SEM/20201001_202940_NE142400C_RAE01_1_S01_M0005-01MS_3_itr0.png
# data/Train/Depth/20201001_202940_NE142400C_RAE01_1_S01_M0005-01MS_3.png