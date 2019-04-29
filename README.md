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
l
## Resources

### Unpaired Image translation notes
* Adversarial loss [short](https://www.quora.com/What-is-adversarial-loss-in-machine-learning) [paper](https://arxiv.org/pdf/1901.08753.pdf) Important for GANs!

* Cycle consistency loss, train both mapping at once, just like translating english to french and then back to french.
* [CycleGAN](https://junyanz.github.io/CycleGAN/)

### Voice to text
* Link here
### Text to image

* [Generative Adversarial Text to Image Synthesis](https://arxiv.org/pdf/1605.05396.pdf)

### Datasets

* Conceptual Captions [link](https://ai.google.com/research/ConceptualCaptions/download)

* Flickr 30k [link](https://www.kaggle.com/hsankesara/flickr-image-dataset/version/1)

I think that we should use the flickr dataset as the 30k images should really be enough in the limited time we have.

## Further notes

Might have to use a bag of words model or some other form of context presentation to simplify what the sentence says, look into this further.
