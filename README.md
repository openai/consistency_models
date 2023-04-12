# Consistency Models

This repository contains the codebase for [Consistency Models](https://arxiv.org/abs/2303.01469), implemented using PyTorch for conducting large-scale experiments on ImageNet-64, LSUN Bedroom-256, and LSUN Cat-256. We have based our repository on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), which was initially released under the MIT license. Our modifications have enabled support for consistency distillation, consistency training, as well as several sampling and editing algorithms discussed in the paper.

The repository for CIFAR-10 experiments is in JAX and will be released separately.

# Pre-trained models

We have released checkpoints for the main models in the paper. Before using these models, please review the corresponding [model card](model-card.md) to understand the intended use and limitations of these models.

Here are the download links for each model checkpoint:

 * EDM on ImageNet-64: [edm_imagenet64_ema.pt](https://openaipublic.blob.core.windows.net/consistency/edm_imagenet64_ema.pt)
 * CD on ImageNet-64 with l2 metric: [cd_imagenet64_l2.pt](https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_l2.pt)
 * CD on ImageNet-64 with LPIPS metric: [cd_imagenet64_lpips.pt](https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_lpips.pt)
 * CT on ImageNet-64: [ct_imagenet64.pt](https://openaipublic.blob.core.windows.net/consistency/ct_imagenet64.pt)
 * EDM on LSUN Bedroom-256: [edm_bedroom256_ema.pt](https://openaipublic.blob.core.windows.net/consistency/edm_bedroom256_ema.pt)
 * CD on LSUN Bedroom-256 with l2 metric: [cd_bedroom256_l2.pt](https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_l2.pt)
 * CD on LSUN Bedroom-256 with LPIPS metric: [cd_bedroom256_lpips.pt](https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_lpips.pt)
 * CT on LSUN Bedroom-256: [ct_bedroom256.pt](https://openaipublic.blob.core.windows.net/consistency/ct_bedroom256.pt)
 * EDM on LSUN Cat-256: [edm_cat256_ema.pt](https://openaipublic.blob.core.windows.net/consistency/edm_cat256_ema.pt)
 * CD on LSUN Cat-256 with l2 metric: [cd_cat256_l2.pt](https://openaipublic.blob.core.windows.net/consistency/cd_cat256_l2.pt)
 * CD on LSUN Cat-256 with LPIPS metric: [cd_cat256_lpips.pt](https://openaipublic.blob.core.windows.net/consistency/cd_cat256_lpips.pt)
 * CT on LSUN Cat-256: [ct_cat256.pt](https://openaipublic.blob.core.windows.net/consistency/ct_cat256.pt)

# Dependencies

To install all packages in this codebase along with their dependencies, run
```sh
pip install -e .
```

# Model training and sampling

We provide examples of EDM training, consistency distillation, consistency training, single-step generation, and multistep generation in [cm/scripts/launch.sh](scripts/launch.sh).

# Evaluations

To compare different generative models, we use FID, Precision, Recall, and Inception Score. These metrics can all be calculated using batches of samples stored in `.npz` (numpy) files. One can evaluate samples with [cm/evaluations/evaluator.py](evaluations/evaluator.py) in the same way as described in [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with reference dataset batches provided therein.

# Citation

If you find this method and/or code useful, please consider citing

```bibtex
@article{song2023consistency,
  title={Consistency Models},
  author={Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2303.01469},
  year={2023},
}
```
