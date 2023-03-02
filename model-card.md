# Overview

These are diffusion models and consistency models described in the paper [Consistency Models](https://arxiv.org/abs/2303.01469). We include the following models in this release:

 * Consistency models trained by CD (with both l2 and LPIPS metrics) on ImageNet 64x64, LSUN Bedroom 256x256, and LSUN Cat 256x256.
 * Consistency models trained by CT on ImageNet 64x64, LSUN Bedroom 256x256, and LSUN Cat 256x256.

# Datasets

The models that we are making available have been trained on the [ILSVRC 2012 subset of ImageNet](http://www.image-net.org/challenges/LSVRC/2012/) or on individual categories from [LSUN](https://arxiv.org/abs/1506.03365). Here we outline the characteristics of these datasets that influence the behavior of the models:

**ILSVRC 2012 subset of ImageNet**: This dataset was curated in 2012 and has around a million pictures, each of which belongs to one of 1,000 categories. A significant number of the categories in this dataset are animals, plants, and other naturally occurring objects. Although many photographs include humans, these humans are typically not represented by the class label (for example, the category "Tench, tinca tinca" includes many photographs of individuals holding fish).

**LSUN**: This dataset was collected in 2015 by a combination of human labeling via Amazon Mechanical Turk and automated data labeling. Both classes that we consider have more than a million images. The dataset creators discovered that when assessed by trained experts, the label accuracy was approximately 90% throughout the entire LSUN dataset. The pictures are gathered from the internet, and those in the cat class often follow a "meme" format. Occasionally, people, including faces, appear in these photographs.


# Performance

These models are intended to generate samples consistent with their training distributions.
This has been measured in terms of FID, Inception Score, Precision, and Recall.
These metrics all rely on the representations of a [pre-trained Inception-V3 model](https://arxiv.org/abs/1512.00567),
which was trained on ImageNet, and so is likely to focus more on the ImageNet classes (such as animals) than on other visual features (such as human faces).


# Intended Use

These models are intended to be used for research purposes only. In particular, they can be used as a baseline for generative modeling research, or as a starting point for advancing such research. These models are not intended to be commercially deployed. Additionally, they are not intended to be used to create propaganda or offensive imagery.

# Limitations

These models sometimes produce highly unrealistic outputs, particularly when generating images containing human faces.
This may stem from ImageNet's emphasis on non-human objects.

In consistency distillation and training, minimizing LPIPS results in better sample quality, as evidenced by improved FID and Inception scores. However, it also carries the risk of overestimating model performance, because LPIPS uses a VGG network pre-trained on ImageNet, while FID and Inception scores also rely on convolutional neural networks (the Inception network in particular) pre-trained on the same ImageNet dataset. Although these two convolutional neural networks do not share the same architecture and we extract latents from them in substantially different ways, knowledge leakage is still plausible which can undermine the fidelity of FID and Inception scores.

Because ImageNet and LSUN contain images from the internet, they include photos of real people, and the model may have memorized some of the information contained in these photos. However, these images are already publicly available, and existing generative models trained on ImageNet have not demonstrated significant leakage of this information.
