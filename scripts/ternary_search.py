"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from functools import cache
from mpi4py import MPI

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import stochastic_iterative_sampler
from evaluations.th_evaluator import FIDAndIS


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    fid_is = FIDAndIS()
    fid_is.set_ref_batch(args.ref_batch)
    (
        ref_fid_stats,
        ref_spatial_stats,
        ref_clip_stats,
    ) = fid_is.get_ref_batch(args.ref_batch)

    def sample_generator(ts):
        logger.log("sampling...")
        all_images = []
        all_labels = []
        all_preds = []

        generator = get_generator(args.generator, args.num_samples, args.seed)
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0,
                    high=NUM_CLASSES,
                    size=(args.batch_size,),
                    device=dist_util.dev(),
                )
                model_kwargs["y"] = classes

            def denoiser(x_t, sigma):
                _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
                if args.clip_denoised:
                    denoised = denoised.clamp(-1, 1)
                return denoised

            x_T = (
                generator.randn(
                    *(args.batch_size, 3, args.image_size, args.image_size),
                    device=dist_util.dev(),
                )
                * args.sigma_max
            )

            sample = stochastic_iterative_sampler(
                denoiser,
                x_T,
                ts,
                t_min=args.sigma_min,
                t_max=args.sigma_max,
                rho=diffusion.rho,
                steps=args.steps,
                generator=generator,
            )
            pred, spatial_pred, clip_pred, text_pred, _ = fid_is.get_preds(sample)

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [
                th.zeros_like(sample) for _ in range(dist.get_world_size())
            ]
            gathered_preds = [th.zeros_like(pred) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            dist.all_gather(gathered_preds, pred)
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            all_preds.extend([pred.cpu().numpy() for pred in gathered_preds])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

            logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        preds = np.concatenate(all_preds, axis=0)
        preds = preds[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]

        dist.barrier()
        logger.log("sampling complete")

        return arr, preds

    @cache
    def get_fid(p, begin=(0,), end=(args.steps - 1,)):

        samples, preds = sample_generator(begin + (p,) + end)
        is_root = dist.get_rank() == 0
        if is_root:
            fid_stats = fid_is.get_statistics(preds, -1)
            fid = ref_fid_stats.frechet_distance(fid_stats)
            fid = MPI.COMM_WORLD.bcast(fid)
            # spatial_stats = fid_is.get_statistics(spatial_preds, -1)
            # sfid = ref_spatial_stats.frechet_distance(spatial_stats)
            # clip_stats = fid_is.get_statistics(clip_preds, -1)
            IS = fid_is.get_inception_score(preds)
            IS = MPI.COMM_WORLD.bcast(IS)
            # clip_fid = fid_is.get_clip_score(clip_preds, text_preds)
            # fcd = ref_clip_stats.frechet_distance(clip_stats)
        else:
            fid = MPI.COMM_WORLD.bcast(None)
            IS = MPI.COMM_WORLD.bcast(None)

        dist.barrier()
        return fid, IS

    def ternary_search(before=(0,), after=(17,)):
        left = before[-1]
        right = after[0]
        is_root = dist.get_rank() == 0
        while right - left >= 3:
            m1 = int(left + (right - left) / 3.0)
            m2 = int(right - (right - left) / 3.0)
            f1, is1 = get_fid(m1, before, after)
            if is_root:
                logger.log(f"fid at m1 = {m1} is {f1}, IS is {is1}")
            f2, is2 = get_fid(m2, before, after)
            if is_root:
                logger.log(f"fid at m2 = {m2} is {f2}, IS is {is2}")
            if f1 < f2:
                right = m2
            else:
                left = m1
            if is_root:
                logger.log(f"new interval is [{left}, {right}]")

        if right == left:
            p = right
        elif right - left == 1:
            f1, _ = get_fid(left, before, after)
            f2, _ = get_fid(right, before, after)
            p = m1 if f1 < f2 else m2
        elif right - left == 2:
            mid = left + 1
            f1, _ = get_fid(left, before, after)
            f2, _ = get_fid(right, before, after)
            fmid, ismid = get_fid(mid, before, after)
            if is_root:
                logger.log(f"fmid at mid = {mid} is {fmid}, ISmid is {ismid}")
            if fmid < f1 and fmid < f2:
                p = mid
            elif f1 < f2:
                p = m1
            else:
                p = m2

        return p

    # convert comma separated numbers to tuples
    begin = tuple(int(x) for x in args.begin.split(","))
    end = tuple(int(x) for x in args.end.split(","))

    optimal_p = ternary_search(begin, end)
    if dist.get_rank() == 0:
        logger.log(f"ternary_search_results: {optimal_p}")
        fid, IS = get_fid(optimal_p, begin, end)
        logger.log(f"fid at optimal p = {optimal_p} is {fid}, IS is {IS}")


def create_argparser():
    defaults = dict(
        begin="0",
        end="39",
        training_mode="consistency_distillation",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        ref_batch="",
        seed=42,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
