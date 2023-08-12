# Consistency Models

This repository contains the codebase for [Consistency Models](https://arxiv.org/abs/2303.01469), implemented using PyTorch for conducting large-scale experiments on ImageNet-64, LSUN Bedroom-256, and LSUN Cat-256. We have based our repository on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), which was initially released under the MIT license. Our modifications have enabled support for consistency distillation, consistency training, as well as several sampling and editing algorithms discussed in the paper.

The repository for CIFAR-10 experiments is in JAX and can be found at [openai/consistency_models_cifar10](https://github.com/openai/consistency_models_cifar10).

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

To install with Docker, run the following commands:
```sh
cd docker && make build && make run
```

# Model training and sampling

We provide examples of EDM training, consistency distillation, consistency training, single-step generation, and multistep generation in [scripts/launch.sh](scripts/launch.sh).

# Evaluations

To compare different generative models, we use FID, Precision, Recall, and Inception Score. These metrics can all be calculated using batches of samples stored in `.npz` (numpy) files. One can evaluate samples with [cm/evaluations/evaluator.py](evaluations/evaluator.py) in the same way as described in [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with reference dataset batches provided therein.

## Use in 🧨 diffusers

Consistency models are supported in [🧨 diffusers](https://github.com/huggingface/diffusers) via the [`ConsistencyModelPipeline` class](https://huggingface.co/docs/diffusers/main/en/api/pipelines/consistency_models). Below we provide an example:

```python
import torch

from diffusers import ConsistencyModelPipeline

device = "cuda"
# Load the cd_imagenet64_l2 checkpoint.
model_id_or_path = "openai/diffusers-cd_imagenet64_l2"
pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Onestep Sampling
image = pipe(num_inference_steps=1).images[0]
image.save("consistency_model_onestep_sample.png")

# Onestep sampling, class-conditional image generation
# ImageNet-64 class label 145 corresponds to king penguins

class_id = 145
class_id = torch.tensor(class_id, dtype=torch.long)

image = pipe(num_inference_steps=1, class_labels=class_id).images[0]
image.save("consistency_model_onestep_sample_penguin.png")

# Multistep sampling, class-conditional image generation
# Timesteps can be explicitly specified; the particular timesteps below are from the original Github repo.
# https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L77
image = pipe(timesteps=[22, 0], class_labels=class_id).images[0]
image.save("consistency_model_multistep_sample_penguin.png")
```
You can further speed up the inference process by using `torch.compile()` on `pipe.unet` (only supported from PyTorch 2.0). For more details, please check out the [official documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/consistency_models). This support was contributed to 🧨 diffusers by [dg845](https://github.com/dg845) and [ayushtues](https://github.com/ayushtues).

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
