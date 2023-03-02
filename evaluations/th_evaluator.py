from .inception_v3 import InceptionV3
import blobfile as bf
import torch
import torch.distributed as dist
import torch.nn as nn
from cm import dist_util
import numpy as np
import warnings
from scipy import linalg
from PIL import Image
from tqdm import tqdm


def clip_preproc(preproc_fn, x):
    return preproc_fn(Image.fromarray(x.astype(np.uint8)))


def all_gather(x, dim=0):
    xs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(xs, x)
    return torch.cat(xs, dim=dim)


class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, resolution: int):
        self.mu = mu
        self.sigma = sigma
        self.resolution = resolution

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class FIDAndIS:
    def __init__(
        self,
        softmax_batch_size=512,
        clip_score_batch_size=512,
        path="https://openaipublic.blob.core.windows.net/consistency/inception/inception-2015-12-05.pt",
    ):
        import clip

        super().__init__()

        self.softmax_batch_size = softmax_batch_size
        self.clip_score_batch_size = clip_score_batch_size
        self.inception = InceptionV3()
        with bf.BlobFile(path, "rb") as f:
            self.inception.load_state_dict(torch.load(f))
        self.inception.eval()
        self.inception.to(dist_util.dev())

        self.inception_softmax = self.inception.create_softmax_model()

        if dist.get_rank() % 8 == 0:
            clip_model, self.clip_preproc_fn = clip.load(
                "ViT-B/32", device=dist_util.dev()
            )
        dist.barrier()
        if dist.get_rank() % 8 != 0:
            clip_model, self.clip_preproc_fn = clip.load(
                "ViT-B/32", device=dist_util.dev()
            )
        dist.barrier()

        # Compute the probe features separately from the final projection.
        class ProjLayer(nn.Module):
            def __init__(self, param):
                super().__init__()
                self.param = param

            def forward(self, x):
                return x @ self.param

        self.clip_visual = clip_model.visual
        self.clip_proj = ProjLayer(self.clip_visual.proj)
        self.clip_visual.proj = None

        class TextModel(nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip_model = clip_model

            def forward(self, x):
                return self.clip_model.encode_text(x)

        self.clip_tokenizer = lambda captions: clip.tokenize(captions, truncate=True)
        self.clip_text = TextModel(clip_model)
        self.clip_logit_scale = clip_model.logit_scale.exp().item()
        self.ref_features = {}
        self.is_root = not dist.is_initialized() or dist.get_rank() == 0

    def get_statistics(self, activations: np.ndarray, resolution: int):
        """
        Compute activation statistics for a batch of images.

        :param activations: an [N x D] batch of activations.
        :return: an FIDStatistics object.
        """
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma, resolution)

    def get_preds(self, batch, captions=None):
        with torch.no_grad():
            batch = 127.5 * (batch + 1)
            np_batch = batch.to(torch.uint8).cpu().numpy().transpose((0, 2, 3, 1))

            pred, spatial_pred = self.inception(batch)
            pred, spatial_pred = pred.reshape(
                [pred.shape[0], -1]
            ), spatial_pred.reshape([spatial_pred.shape[0], -1])

            clip_in = torch.stack(
                [clip_preproc(self.clip_preproc_fn, img) for img in np_batch]
            )
            clip_pred = self.clip_visual(clip_in.half().to(dist_util.dev()))
            if captions is not None:
                text_in = self.clip_tokenizer(captions)
                text_pred = self.clip_text(text_in.to(dist_util.dev()))
            else:
                # Hack to easily deal with no captions
                text_pred = self.clip_proj(clip_pred.half())
            text_pred = text_pred / text_pred.norm(dim=-1, keepdim=True)

        return pred, spatial_pred, clip_pred, text_pred, np_batch

    def get_inception_score(
        self, activations: np.ndarray, split_size: int = 5000
    ) -> float:
        """
        Compute the inception score using a batch of activations.
        :param activations: an [N x D] batch of activations.
        :param split_size: the number of samples per split. This is used to
                           make results consistent with other work, even when
                           using a different number of samples.
        :return: an inception score estimate.
        """
        softmax_out = []
        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i : i + self.softmax_batch_size]
            with torch.no_grad():
                softmax_out.append(
                    self.inception_softmax(torch.from_numpy(acts).to(dist_util.dev()))
                    .cpu()
                    .numpy()
                )
        preds = np.concatenate(softmax_out, axis=0)
        # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L46
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i : i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))

    def get_clip_score(
        self, activations: np.ndarray, text_features: np.ndarray
    ) -> float:
        # Sizes should never mismatch, but if they do we want to compute
        # _some_ value instead of crash looping.
        size = min(len(activations), len(text_features))
        activations = activations[:size]
        text_features = text_features[:size]

        scores_out = []
        for i in range(0, len(activations), self.clip_score_batch_size):
            acts = activations[i : i + self.clip_score_batch_size]
            sub_features = text_features[i : i + self.clip_score_batch_size]
            with torch.no_grad():
                image_features = self.clip_proj(
                    torch.from_numpy(acts).half().to(dist_util.dev())
                )
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                image_features = image_features.detach().cpu().float().numpy()
            scores_out.extend(np.sum(sub_features * image_features, axis=-1).tolist())
        return np.mean(scores_out) * self.clip_logit_scale

    def get_activations(self, data, num_samples, global_batch_size, pr_samples=50000):
        if self.is_root:
            preds = []
            spatial_preds = []
            clip_preds = []
            pr_images = []

        for _ in tqdm(range(0, int(np.ceil(num_samples / global_batch_size)))):
            batch, cond, _ = next(data)
            batch, cond = batch.to(dist_util.dev()), {
                k: v.to(dist_util.dev()) for k, v in cond.items()
            }
            pred, spatial_pred, clip_pred, _, np_batch = self.get_preds(batch)
            pred, spatial_pred, clip_pred = (
                all_gather(pred).cpu().numpy(),
                all_gather(spatial_pred).cpu().numpy(),
                all_gather(clip_pred).cpu().numpy(),
            )
            if self.is_root:
                preds.append(pred)
                spatial_preds.append(spatial_pred)
                clip_preds.append(clip_pred)
                if len(pr_images) * np_batch.shape[0] < pr_samples:
                    pr_images.append(np_batch)

        if self.is_root:
            preds, spatial_preds, clip_preds, pr_images = (
                np.concatenate(preds, axis=0),
                np.concatenate(spatial_preds, axis=0),
                np.concatenate(clip_preds, axis=0),
                np.concatenate(pr_images, axis=0),
            )
            # assert len(pr_images) >= pr_samples
            return (
                preds[:num_samples],
                spatial_preds[:num_samples],
                clip_preds[:num_samples],
                pr_images[:pr_samples],
            )
        else:
            return [], [], [], []

    def get_virtual_batch(self, data, num_samples, global_batch_size, resolution):
        preds, spatial_preds, clip_preds, batch = self.get_activations(
            data, num_samples, global_batch_size, pr_samples=10000
        )
        if self.is_root:
            fid_stats = self.get_statistics(preds, resolution)
            spatial_stats = self.get_statistics(spatial_preds, resolution)
            clip_stats = self.get_statistics(clip_preds, resolution)
            return batch, dict(
                mu=fid_stats.mu,
                sigma=fid_stats.sigma,
                mu_s=spatial_stats.mu,
                sigma_s=spatial_stats.sigma,
                mu_clip=clip_stats.mu,
                sigma_clip=clip_stats.sigma,
            )
        else:
            return None, dict()

    def set_ref_batch(self, ref_batch):
        with bf.BlobFile(ref_batch, "rb") as f:
            data = np.load(f)
            fid_stats = FIDStatistics(mu=data["mu"], sigma=data["sigma"], resolution=-1)
            spatial_stats = FIDStatistics(
                mu=data["mu_s"], sigma=data["sigma_s"], resolution=-1
            )
            clip_stats = FIDStatistics(
                mu=data["mu_clip"], sigma=data["sigma_clip"], resolution=-1
            )

        self.ref_features[ref_batch] = (fid_stats, spatial_stats, clip_stats)

    def get_ref_batch(self, ref_batch):
        return self.ref_features[ref_batch]
