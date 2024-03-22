# R2DM

R2DM is a denoising diffusion probabilistic model (DDPM) for LiDAR range/reflectance generation based on the equirectangular representation.

![sampled in 256 steps](https://github.com/kazuto1011/r2dm/assets/9032347/0947620b-fd82-4a17-a614-890d4de13554)

**LiDAR Data Synthesis with Denoising Diffusion Probabilistic Models**<br>
[Kazuto Nakashima](https://kazuto1011.github.io), [Ryo Kurazume](https://robotics.ait.kyushu-u.ac.jp/kurazume/en/)<br>
ICRA 2024<br>
[project](https://kazuto1011.github.io/r2dm) | [arxiv](https://arxiv.org/abs/2309.09256) | [online demo](https://huggingface.co/spaces/kazuto1011/r2dm)

**Quick demo:**

```sh
pip install torch torchvision einops tqdm pydantic
```

```py
import torch

# Setup our pre-trained model & sampling
r2dm, lidar_utils, cfg = torch.hub.load("kazuto1011/r2dm", "pretrained_r2dm", device="cuda")
lidar_image = r2dm.sample(batch_size=1, num_steps=256)  # (batch size, 2, height, width)

# Postprocessing
lidar_image = lidar_utils.denormalize(lidar_image.clamp(-1, 1))  # [-1,1] -> [0,1]
range_image = lidar_utils.revert_depth(lidar_image[:, [0]])  # Range
rflct_image = lidar_image[:, [1]]  # Reflectance
point_cloud = lidar_utils.to_xyz(range_image)  # Point cloud
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
ln -sf $PATH_TO_KITTI360_ROOT data/kitti_360/dataset
```

Please set the environment variable `$HF_DATASETS_CACHE` to cache the processed dataset (default: `~/.cache/huggingface/datasets`).

## Training

To start training DDPMs:

```sh
accelerate launch train.py
```

- The initial run takes about 15 min to preprocess & cache the whole dataset.
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

![completion](https://github.com/kazuto1011/r2dm/assets/9032347/f2b89329-c43f-4cd9-b0ff-210184c01632)

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{nakashima2023lidar,
    title   = {LiDAR Data Synthesis with Denoising Diffusion Probabilistic Models},
    author  = {Kazuto Nakashima and Ryo Kurazume},
    year    = 2023,
    journal = {arXiv:2309.09256}
}
```

## Acknowledgements

- The discrete/continuous diffusion processes are based on [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
- The BEV-based evaluation metrics are based on [vzyrianov/lidargen](https://github.com/vzyrianov/lidargen).
