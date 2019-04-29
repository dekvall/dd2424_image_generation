# Generating images from voice

## Installation

Make sure you have `virtualenv` and `python 3.5+` installed


```bash
bash install.sh
```
This will activate the virtualenv and install the proper packages.

You will probably also need [Git LFS](https://git-lfs.github.com/) to track various datasets


### Project structure
* [Models](models) - The resulting generated models
* [Utilities](utils) - Various tools such as visualization etc.
* [Scripts](scripts) - The scripts used for the project
* [Datasets](datasets) - The datasets used for the project


## Resources

### Unpaired Image translation notes
* Adversarial loss [short](https://www.quora.com/What-is-adversarial-loss-in-machine-learning) [paper](https://arxiv.org/pdf/1901.08753.pdf) Important for GANs!

* Cycle consistency loss, train both mapping at once, just like translating english to french and then back to french.
* [CycleGAN](https://junyanz.github.io/CycleGAN/)

### Voice to text
* Link here

### Text to image

* [Generative Adversarial Text to Image Synthesis](https://arxiv.org/pdf/1605.05396.pdf)

* [AttnGAN](https://arxiv.org/pdf/1711.10485.pdf)

* [Paragraph vectors](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

* Various papers with code [paperswithcode](https://paperswithcode.com/task/text-to-image-generation)

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

Might have to use a bag of words model or some other form of context presentation to simplify what the sentence says, look into this further.
