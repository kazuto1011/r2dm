import dataclasses
import math
from typing import Literal

import torch
from torch.optim.lr_scheduler import LambdaLR


@dataclasses.dataclass
class TrainingConfig:
    dataset: Literal["kitti_raw", "kitti_360"] = "kitti_360"
    image_format: str = "log_depth"
    lidar_projection: Literal[
        "unfolding-2048",
        "spherical-2048",
        "unfolding-1024",
        "spherical-1024",
    ] = "spherical-1024"
    train_depth: bool = True
    train_reflectance: bool = True
    resolution: tuple[int, int] = (64, 1024)
    min_depth = 1.45
    max_depth = 80.0
    batch_size_train: int = 8
    batch_size_eval: int = 8
    num_workers: int = 4
    num_steps: int = 300_000
    save_image_steps: int = 5_000
    save_model_steps: int = 10_000
    gradient_accumulation_steps: int = 1
    criterion: str = "l2"
    lr: float = 1e-4
    lr_warmup_steps: int = 10_000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    ema_decay: float = 0.995
    ema_update_every: int = 10
    output_dir: str = "logs/diffusion"
    seed: int = 0
    mixed_precision: str = "fp16"
    dynamo_backend: str = "inductor"
    model_name: str = "efficient_unet"
    model_base_channels: int = 64
    model_temb_channels: int | None = None
    model_channel_multiplier: tuple[int] | int = (1, 2, 4, 8)
    model_num_residual_blocks: tuple[int] | int = 3
    model_gn_num_groups: int = 32 // 4
    model_gn_eps: float = 1e-6
    model_attn_num_heads: int = 8
    model_coords_embedding: Literal[
        "spherical_harmonics", "polar_coordinates", "fourier_features", None
    ] = "fourier_features"
    model_dropout: float = 0.0
    diffusion_num_training_steps: int = 1024
    diffusion_num_sampling_steps: int = 128
    diffusion_objective: Literal["eps", "v", "x_0"] = "eps"
    diffusion_beta_schedule: str = "cosine"
    diffusion_timesteps_type: Literal["continuous", "discrete"] = "continuous"


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
