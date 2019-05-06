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
    """
    def __init__(self, config):
        self.config = config
        print(self.config)
        if self.config.BETA1:
            self.create_dataset()
        
    
    