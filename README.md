# R2DM

R2DM is a denoising diffusion probabilistic model (DDPM) for LiDAR range/reflectance generation based on the equirectangular representation.

![samples](https://github.com/kazuto1011/r2dm-dev/assets/9032347/9deb97b0-a33b-4c85-9925-5df5bb7d7b82)

[**LiDAR Data Synthesis with Denoising Diffusion Probabilistic Models**](https://arxiv.org/abs/2309.09256)<br>
[Kazuto Nakashima](https://kazuto1011.github.io), Ryo Kurazume<br>
arXiv 2023

**Quick demo:**

```sh
pip install torch torchvision numpy einops tqdm
```

```py
import torch.hub

# sampling
ddpm, lidar_utils, _ = torch.hub.load("kazuto1011/r2dm", "pretrained_r2dm")
output = ddpm.sample(batch_size=1, num_steps=256)  # (B, 2, H, W)

# postprocessing
output = lidar_utils.denormalize(output.clamp(-1, 1))
range_image = lidar_utils.revert_depth(output[:, [0]])
reflectance_image = output[:, [1]]
```

## Setup

### Python & CUDA

w/ [conda](https://docs.conda.io/projects/miniconda/en/latest/) framework:

```sh
conda env create -f environment.yaml
conda activate r2dm
```

If you are stuck with an endless installation, [try `libmamba` for the conda solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community).

### Dataset

For training & evaluation, please download the [KITTI-360 dataset](http://www.cvlibs.net/datasets/kitti-360/) (163 GB) and make a symlink:

```sh
ln -sf $PATH_TO_KITTI360_ROOT data/kitti360/dataset
```

Please set the environment variable `$HF_DATASETS_CACHE` to cache the processed dataset (default: `~/.cache/huggingface/datasets`).

## Training

To start training DDPMs:

```sh
accelerate launch train.py
```

- The initial run takes about an hour to preprocess & cache the whole dataset.
- The default configuration is `config H` (R2DM) in our paper.
- Distributed training and mixed precision are enabled by default.
- Run with `--help` to list the available options.

To monitor the training progress:

```sh
tensorboard --logdir logs/
```

To generate samples w/ a training checkpoint (\*.pth) at `$CHECKPOINT_PATH`:

```sh
python generate.py --ckpt $CHECKPOINT_PATH
```

## Evaluation

To generate, save, and evaluate samples:

```sh
accelerate launch sample_and_save.py --ckpt $CHECKPOINT_PATH --output_dir $OUTPUT_DIR
python evaluate.py --ckpt $CHECKPOINT_PATH --sample_dir $OUTPUT_DIR
```

The generated samples are saved in `$OUTPUT_DIR`.

## Completion demo

```sh
python completion_demo.py --ckpt $CHECKPOINT_PATH
```

![completion_T-0032_r-0016_j-0001](https://github.com/kazuto1011/r2dm-dev/assets/9032347/0ac5f257-9e8d-4c20-a3ab-967f5e5b7afb)

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@misc{nakashima2023lidar,
	title  = {LiDAR Data Synthesis with Denoising Diffusion Probabilistic Models},
	author = {Kazuto Nakashima and Ryo Kurazume},
	year   = 2023,
	eprint = {arXiv:2309.09256}
}
```

## Acknowledgements

- The discrete/continuous diffusion processes are based on [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
- The BEV-based evaluation metrics are based on [vzyrianov/lidargen](https://github.com/vzyrianov/lidargen).
