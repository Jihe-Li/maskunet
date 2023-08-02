import torch
import os
import numpy as np
from torchvision import transforms
from dataset import CompletionDataset
from configs import CONFIGS
from torch.utils.data import DataLoader


flags = CONFIGS.dataset
toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
train_test = 'test'

dataset = CompletionDataset(flags.data_process_dir, train_test)
dataloader = DataLoader(dataset, batch_size=1)
for i, data in enumerate(dataloader):
    img = data['x'][0].type(torch.float32)
    # mask = img == -1
    # img = torch.where(mask, 0, 1).type(torch.float32)
    pic = toPIL(img) 
    save_dir = flags.data_process_dir + '/' + train_test + '.jpg' + '/' + 'data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = save_dir + '/' + '%05d'%i + '.jpg'
    pic.save(filename)
