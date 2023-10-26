from pathlib import Path

import torch

from models.diffusion import (
    ContinuousTimeGaussianDiffusion,
    DiscreteTimeGaussianDiffusion,
)
from models.efficient_unet import EfficientUNet
from utils.lidar import LiDARUtility

from .training import TrainingConfig, count_parameters


def setup_model(
    ckpt,
    device: torch.device | str = "cpu",
    ema: bool = True,
    show_info: bool = True,
    compile_denoiser: bool = False,
):
    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location="cpu")
    cfg = TrainingConfig(**ckpt["cfg"])

    in_channels = [0, 0]
    if cfg.train_depth:
        in_channels[0] = 1
    if cfg.train_reflectance:
        in_channels[1] = 1
    in_channels = sum(in_channels)

    if cfg.model_name == "efficient_unet":
        unet = EfficientUNet(
            in_channels=in_channels,
            resolution=cfg.resolution,
            base_channels=cfg.model_base_channels,
            temb_channels=cfg.model_temb_channels,
            channel_multiplier=cfg.model_channel_multiplier,
            num_residual_blocks=cfg.model_num_residual_blocks,
            gn_num_groups=cfg.model_gn_num_groups,
            gn_eps=cfg.model_gn_eps,
            attn_num_heads=cfg.model_attn_num_heads,
            coords_embedding=cfg.model_coords_embedding,
            ring=True,
        )
    else:
        raise ValueError(f"Unknown: {cfg.model_name}")

    if cfg.diffusion_timesteps_type == "discrete":
        diffusion = DiscreteTimeGaussianDiffusion(
            denoiser=unet,
            criterion=cfg.criterion,
            num_training_steps=cfg.diffusion_num_training_steps,
            objective=cfg.diffusion_objective,
            beta_schedule=cfg.diffusion_beta_schedule,
        )
    elif cfg.diffusion_timesteps_type == "continuous":
        diffusion = ContinuousTimeGaussianDiffusion(
            denoiser=unet,
            criterion=cfg.criterion,
            objective=cfg.diffusion_objective,
            beta_schedule=cfg.diffusion_beta_schedule,
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion_timesteps_type}")

    state_dict = ckpt["ema_weights"] if ema else ckpt["weights"]
    diffusion.load_state_dict(state_dict)
    diffusion.eval()
    diffusion.to(device)

    if compile_denoiser:
        diffusion.denoiser = torch.compile(diffusion.denoiser)

    lidar_utils = LiDARUtility(
        resolution=cfg.resolution,
        image_format=cfg.image_format,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        ray_angles=diffusion.denoiser.coords,
    )
    lidar_utils.eval()
    lidar_utils.to(device)

    if show_info:
        print(
            *[
                f"resolution: {unet.resolution}",
                f"denoiser: {unet.__class__.__name__}",
                f"diffusion: {diffusion.__class__.__name__}",
                f'#steps:  {ckpt["global_step"]:,}',
                f"#params: {count_parameters(diffusion):,}",
            ],
            sep="\n",
        )

    return diffusion, lidar_utils, cfg


def setup_rng(seeds: list[int], device: torch.device | str):
    return [torch.Generator(device=device).manual_seed(i) for i in seeds]
