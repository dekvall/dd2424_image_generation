# Generating images from voice

## Installation

Make sure you have `virtualenv` and `python 3.5+` installed


```bash
bash install.sh
```
This will activate the virtualenv and install the proper packages.

You will probably also need [Git LFS](https://git-lfs.github.com/) to track various datasets

To then launch the jupyter instance use
```bash
jupyter notebook
```
And you should be directed to `localhost:8888`. In the future we should set up the gcloud instance with the same thing. But it seems some bureaucracy got in the way for now.

### Project structure
* [Models](models) - The resulting generated models
* [Utilities](utils) - Various tools such as visualization etc.
* [Scripts](scripts) - The scripts used for the project
* [Datasets](datasets) - The datasets used for the project


## Resources

### Voice to text
* Link here

### Text to image

* [Generative Adversarial Text to Image Synthesis](https://arxiv.org/pdf/1605.05396.pdf)

* [AttnGAN](https://arxiv.org/pdf/1711.10485.pdf)

* [Paragraph vectors](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

* [StackGAN++](https://arxiv.org/pdf/1710.10916)

* [CycleGAN](https://junyanz.github.io/CycleGAN/)

* Various papers with code [paperswithcode](https://paperswithcode.com/task/text-to-image-generation)

* Adversarial loss [short](https://www.quora.com/What-is-adversarial-loss-in-machine-learning) [paper](https://arxiv.org/pdf/1901.08753.pdf)

### Datasets

* Conceptual Captions [link](https://ai.google.com/research/ConceptualCaptions/download)

* Flickr 30k [link](https://www.kaggle.com/hsankesara/flickr-image-dataset/version/1)

* TIMIT Speech corpus [link](https://catalog.ldc.upenn.edu/LDC93S1)

I think that we should use the flickr dataset as the 30k images should really be enough in the limited time we have.

Once you have downloaded the flickr dataset extract it and run the resize script
that's located in the flickr30k_images folder, from the folder in question
```bash
bash resize_images.sh
```

## Further notes

### Report
The report Overleaf is available [here](https://www.overleaf.com/4488118745cjmprgwyfxcw)

### Notes

Might have to use a bag of words model or some other form of context presentation to simplify what the sentence says, look into this further.

### Training GANS
* [Tips and Tricks](https://github.com/soumith/ganhacks)

## GCP
### Running StackGAN
Run StackGAN on GCP from the code folder with
```bash
python2 main.py --cfg cfg/coco_eval.yml --gpu 0
```
Contrary to popular belief setting `--gpu 0` here actually refers to the id of the gpu. In most other cases `gpu 0` refers to cpu mode. Weird.

The generated images will be stored in the `models/coco/netG_epoch_90` directory.

### Jupyter notebooks
To use jupyter notebooks, run this on the remote
```bash
david@torcher-vm:~/StackGAN-Pytorch$ jupyter notebook --no-browser
```

Then tunnel your connection through
```bash
david@fridge:~$ ssh -N -L localhost:8888:localhost:8888 david@<EXTERNAL_IP_OF_VM>
```
Then simply open a browser on `localhost:8888` and provide it with the token that should be visible in the commandline window on the vm to connect.
### Show results
The images are viewable in python notebooks and can also, be downloaded from there. 

#### DCGAN
Result of the first DCGAN, trained over 200 epochs with the flickr2k_dataset. Based only on the latent vector, no scene information.

![Results for flickr2k dataset, 200 epochs.](results/dcgan_2k_200e_32b.png)

## TODO
* Implement the correct loss function
* Integrate the text embedding into the discriminator

