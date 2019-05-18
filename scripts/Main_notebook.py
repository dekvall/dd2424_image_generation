#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
from PIL import Image


# In[30]:


class CommentedImagesDataset(torch.utils.data.Dataset):
    """Dataset for commented images, each image can have multiple comments"""

    def __init__(self, names, comments, root_dir, transform=None):
        """
        Args:
            names (string): Path to the npy array with the image names.
            comments (string): Path to the text embedded comments npy array
            names and comments have the same row dimension
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.comments = np.load(comments)
        self.names = np.load(names)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                str(self.names[idx])+".jpg")
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
            
        comment = self.comments[idx,:]
        sample = {'image': image, 'comment': comment}

        return sample

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
        self.comment_embedding_size = 600
        random.seed(self.config.SEED.value)
        torch.manual_seed(self.config.SEED.value)
        
        if self.config.TRAINING.value:
            self.create_dataset()
        if self.config.VERBOSE.value > 3:
            self.plot_some_training_images()
        
        self.create_generator()
        self.create_discriminator()
        self.create_optimizer_and_loss()

    
    def create_dataset(self):
        self.dataset = CommentedImagesDataset(root_dir=self.config.DATASET_ROOT.value+'/img',
                                                    names=self.config.DATASET_ROOT.value+'/names.npy',
                                                    comments=self.config.DATASET_ROOT.value+'/comments.npy',
                                                    transform=transforms.Compose([
                               transforms.Resize((self.config.IMAGE_HEIGHT.value, self.config.IMAGE_WIDTH.value)),
                               transforms.CenterCrop((self.config.IMAGE_HEIGHT.value, self.config.IMAGE_WIDTH.value)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.BATCH_SIZE.value,
                                         shuffle=True, num_workers=self.config.WORKERS.value)
                               
        self.device = torch.device("cuda:0" if (self.config.GPU_MODE.value and torch.cuda.is_available() and self.config.N_GPU.value > 0) else "cpu")
                               
    def plot_some_training_images(self):
        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(self.device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    def create_generator(self):
        self.netG = Generator(self.config).to(self.device)

        if (self.device.type == 'cuda') and (self.config.N_GPU.value > 1):
            self.netG = nn.DataParallel(netG, list(range(self.config.N_GPU.value)))

        self.netG.apply(weights_init)
        print(self.netG)
    def create_discriminator(self):
        self.netD = Discriminator(self.config).to(self.device)

        if (self.device.type == 'cuda') and (self.config.N_GPU.value > 1):
            self.netD = nn.DataParallel(netG, list(range(self.config.N_GPU.value)))

        self.netD.apply(weights_init)
        print(self.netD)
        
    def create_optimizer_and_loss(self):
        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, self.config.N_LATENT_VECTOR.value, 1, 1, device=self.device)
        self.fixed_comment_noise = torch.randn(64, self.comment_embedding_size, 1, 1, device=self.device)
        
        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = 0
        self.wrong_label = 0

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.config.LEARNING_RATE.value, betas=(self.config.BETA1.value, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.config.LEARNING_RATE.value, betas=(self.config.BETA1.value, 0.999))

        
    def train_network(self):
        # Training Loop
        # Lists to keep track of progress
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.config.N_TRAINING_EPOCHS.value):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader):
                #Algorithm:
                #1. Train with real image right text, s_r
                #2. Train with real image, wrong text, s_w
                #3. Train with fake image, right text, s_f
                
                #LossD 
                #LossG log(s_f)
                
                #To select the wrong text, we randomize a text embedding
                
                ############################
                # (1) Update D network: maximize log(s_r) + (log(1−s_w) + log(1−s_f))/2
                ###########################
                ## Train with real image, right text
                self.netD.zero_grad()
                # Format batch
                real_img = data['image'].to(self.device)
                #The real comment is slight perturbed by  a normal distribution
                real_comment = data['comment'].to(self.device)
                b_size = real_img.size(0)
                c_size = real_comment.size(0)
                real_comment += 0.001*torch.randn(b_size, self.comment_embedding_size, device=self.device)
                label = torch.full((b_size,), self.real_label, device=self.device)
                # Forward pass real batch through D
                output = self.netD(real_img, real_comment).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()
                
                ## Train with real-image, wrong text
                noise_comment = torch.randn(b_size, self.comment_embedding_size, 1, 1, device=self.device)
                # Generate wrong image batch with G
                label.fill_(self.wrong_label)
                # Classify all fake batch with D
                output = self.netD(real_img, noise_comment).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_wrong = 1/2*self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_wrong.backward()
                D_G_w = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                
                ## Train with real-image, wrong text
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.config.N_LATENT_VECTOR.value, 1, 1, device=self.device)
                # Generate wrong image batch with G
                fake = self.netG(noise, real_comment)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach(), real_comment).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = 1/2*self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_f = output.mean().item()
                
                # Add the gradients 
                errD = errD_real + errD_wrong + errD_fake 
                #errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(s_f)
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake, real_comment).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()
                
                # Output training stats
                if i % 30 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.config.N_TRAINING_EPOCHS.value, i, len(self.dataloader),
                             errD.item(), errG.item(), D_x, D_G_w, D_G_z2))

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.config.N_TRAINING_EPOCHS.value-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise, self.fixed_comment_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
            if epoch % 50 == 0:
                #Backup the model every 50 epochs
                self.save_models(num=epoch)
                self.save_loss_plot(num=epoch)
                
        #self.show_loss_plot()
        #self.create_animation()
        self.show_side_by_side()
        
    def save_models(self, num = 0):
        torch.save(self.netG.state_dict(), '../models/generator_{}.pth'.format(num))
        torch.save(self.netD.state_dict(), '../models/discriminator_{}.pth'.format(num))
        
    def save_loss_plot(self, num=0):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('../results/lossfunc_{}_epocs.png'.format(num))
        plt.close()
        np.save('../results/G_loss', np.array(self.G_losses))
        np.save('../results/D_loss', np.array(self.D_losses))

    def create_animation(self):
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in self.img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())
        
    def show_side_by_side(self):
        real_batch = next(iter(self.dataloader))
        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(self.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(self.img_list[-1],(1,2,0)))
        plt.show()
        
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.ngpu = config.N_GPU.value
        nz = config.N_LATENT_VECTOR.value
        nzc = config.N_TEXT_EMBEDDING.value
        ngf = config.N_GENERATOR_FEATURE_MAP.value
        nc = config.N_COLOR_CHANNELS.value
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz+nzc, ngf * 8, 4, 1, 0, bias=False),
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

    def forward(self, image, comment):
        comment = comment.reshape((comment.shape[0],comment.shape[1],1,1))
        cat = torch.cat((image, comment), dim=1)
        return self.main(cat)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.ngpu = config.N_GPU.value
        ndf = config.N_DISCRIMINATOR_FEATURE_MAP.value
        nc = config.N_COLOR_CHANNELS.value
        nzc = config.N_TEXT_EMBEDDING.value
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=528, out_channels = 1, kernel_size=4, stride=1, padding=0, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(in_channels = ndf * 4, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
        self.conv_image = nn.Sequential(
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
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        
        self.conv_comment = nn.Sequential(
            # input is 600 x 1
            nn.Conv1d(in_channels=nzc, out_channels=ndf*4, kernel_size=1,bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf) x 32 x 32
            #nn.Sigmoid()
        )

    def forward(self, image, comment):
        comment = comment.reshape((comment.shape[0],comment.shape[1],1))
        img = self.conv_image(image)
        com = self.conv_comment(comment)
        com = com.reshape((com.shape[0], 16, 4,4))
        c = torch.cat((img, com), dim=1)
        return self.main(c)


# In[12]:


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[29]:


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
    trainer = DCGAN(Enum('config',config))
    trainer.train_network()
    trainer.save_models()
    
if __name__ == '__main__':
    main()

