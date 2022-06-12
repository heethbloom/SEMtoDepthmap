#%%
import os
import matplotlib. pyplot as plt
from PIL import Image
import numpy as np
from torchvision import datasets, transforms

#%%
#######################################
##파일 개수 세기 및 이미지 확인해보기####
#######################################

len(os.listdir('./data/Test/SEM')), len(os.listdir('./data/Test/Depth')), 
# %%
img = plt.imread('data/Train/Depth/20201001_202940_NE142400C_RAE01_1_S01_M0005-01MS_3.png')
plt.imshow(img)
plt.show()
# %%
img = Image.open('data/Test/SEM/20210304_165053_NE142400C_RAA26_1_S01_M0304-01MS_0_itr0.png').convert('L') #회색조로 변환 
img_gray=np.array(img)
plt.imshow(img_gray, 'gray')
plt.show()
tf = transforms.ToTensor()
img_t = tf(img)

print(img_t.size())

tf_rev = transforms.ToPILImage()
img_r = tf_rev(img_t)
print(img_r)
plt.imshow(img_r, 'gray')
plt.show()
#%%
print(img_t[0][42][32])
#%%

#######################################
##custom dataset 다루기 연습
#######################################
depth_file_dir = natsort.natsorted(os.listdir('data/Train/Depth'))
sem_file_dir = natsort.natsorted(os.listdir('data/Train/SEM'))
#%%
depth_file_dir[0][:-4]

def make_dataset(depthdir, semdir):
    depth_img = [f for f in os.listdir(depthdir)]
#%%
depth_img = [f[:-4] for f in depth_file_dir]
depth_img
#%%
sem_img = [f[:-4] for f in sem_file_dir]
sem_img
#%%
sem_sample = []
for i in range(len(sem_img)//4):
    sem_sample.append(sem_img[i*4:(i+1)*4])
sem_sample
#%%
len(sem_sample)
#%%
depthdir = 'data/Train/Depth'
semdir = 'data/Train/SEM'
# tt = transforms.ToTensor()

def make_dataset(mode):
    depth_img = [f for f in natsort.natsorted(os.listdir('data/'+'{}'.format(mode) + '/Depth'))]
    sem_img = [f for f in natsort.natsorted(os.listdir('data/'+'{}'.format(mode) + '/SEM'))]
    sem_sample = []
    for i in range(len(sem_img)//4):
        sem_sample.append(sem_img[i*4:(i+1)*4])
    return depth_img, sem_sample    