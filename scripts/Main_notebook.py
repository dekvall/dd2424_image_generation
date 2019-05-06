#!/usr/bin/env python
# coding: utf-8

# In[49]:


import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


class DCGAN():
    """
    config fields
    
    BATCH_SIZE = 128,
    BETA1= 0.5,
    DATASET_ROOT= 'datasets/flickr30k_resized_tiny',
    IMAGE_HEIGHT= 375,
    IMAGE_WIDTH= 500,
    LEARNING_RATE= 0.0002,
    N_COLOR_CHANNELS= 3,
    N_DISCRIMINATOR_FEATURE_MAP= 64,
    N_GENERATOR_FEATURE_MAP= 64,
    N_GPU= 1,
    N_LATENT_VECTOR= 100,
    N_TRAINING_EPOCHS= 8,
    TRAINING= True,
    WORKERS= 1
    VERBOSE= [0,1,2,3]
    """
    def __init__(self, config):
        self.config = config
        random.seed(self.config.SEED.value)
        torch.manual_seed(self.config.SEED.value)
        
        if self.config.TRAINING.value:
            self.create_dataset()
        if self.config.VERBOSE.value > 3:
            self.plot_some_training_images()
        
        
    def create_dataset(self):
        self.dataset = dset.ImageFolder(root=self.config.DATASET_ROOT.value,
                           transform=transforms.Compose([
                               transforms.Resize((self.config.IMAGE_HEIGHT.value, self.config.IMAGE_WIDTH.value)),
                               transforms.CenterCrop((self.config.IMAGE_HEIGHT.value, self.config.IMAGE_WIDTH.value)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.BATCH_SIZE.value,
                                         shuffle=True, num_workers=self.config.WORKERS.value)
                               
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.config.N_GPU.value > 0) else "cpu")
                               
    def plot_some_training_images(self):
        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

        
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# In[ ]:


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[50]:


import yaml
from pprint import pprint
from enum import Enum

def main():
    config_path = '../config/config.yml'
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    print('Config loaded from: {}'.format(config_path))
    pprint(config)
    DCGAN(Enum('config',config))
    
if __name__ == '__main__':
    main()

