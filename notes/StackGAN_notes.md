# StackGAN

## What is it?
Stacked adversarial networks to produce high quality images.
The first GAN sketces the primitive shape and colors of of the scene given the text description, this yields low resolution images. GAN 2 takes the results from the first one AND the text description. This creates high details. 
## Why does it exist?
Normal GANs face resolution issues. Stacks will create high resolution photo-realistic images. 

## Why is a good approach to our problem?

## What is a GAN?
GANs are compsed of a generator and a discriminator. These two have competing goals; the generator is trained to generate samples towards the true data distribution and the discriminator is trained to distinguish between real samples from the true distribution and fake samples produced by the generator. 

GANs are known to be difficult to train. The process is very senstive to hyper parameters. 

