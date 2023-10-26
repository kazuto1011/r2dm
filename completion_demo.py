import random
from argparse import ArgumentParser
from pathlib import Path

import datasets as ds
import einops
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import utils.inference
import utils.render
from metrics import rangenet


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =================================================================================
    # Prepare pre-trained models
    # =================================================================================

    ddpm, lidar_utils, cfg = utils.inference.setup_model(args.ckpt, device=device)

    H, W = cfg.resolution
    semseg, preprocess = rangenet.rangenet53(
        weights=f"SemanticKITTI_{H}x{W}",
        compile=False,
        device=device,
    )
    cmap = rangenet.make_semantickitti_cmap()

    def make_semseg_inputs(sample):
        sample = lidar_utils.denormalize(sample)
        depth = lidar_utils.revert_depth(sample[:, [0]])
        mask = (depth > lidar_utils.min_depth).float()
        mask *= (depth < lidar_utils.max_depth).float()
        xyz = lidar_utils.to_xyz(depth)
        rflct = sample[:, [1]]
        inputs = torch.cat([depth, xyz, rflct], dim=1)
        inputs = preprocess(inputs, mask)
        return inputs

    # =================================================================================
    # Prepare inputs
    # =================================================================================

    dataset = ds.load_dataset(
        path=f"data/{cfg.dataset}",
        name=cfg.lidar_projection,
        split=ds.Split.TEST,
    ).with_format("torch")

    if args.sample_id == -1:
        args.sample_id = random.randint(0, len(dataset))
    print(f"sample id: {args.sample_id}")

    item = dataset[args.sample_id]
    depth = item["depth"][None].float().to(device)
    depth = lidar_utils.convert_depth(depth)
    depth = lidar_utils.normalize(depth)
    rflct = item["reflectance"][None].float().to(device)
    rflct = lidar_utils.normalize(rflct)
    rydrp = item["mask"][None].float().to(device)
    x_orig = torch.cat([depth, rflct], dim=1)
    x_orig = rydrp * x_orig + (1 - rydrp) * -1
    x_orig = F.interpolate(x_orig, size=cfg.resolution, mode="nearest-exact")

    # =================================================================================
    # Simulate corruptions
    # =================================================================================

    batch_size = 4
    mask = torch.zeros_like(x_orig).repeat_interleave(batch_size, dim=0)
    mask[0, ...] = 1
    mask[1, :, ::4] = 1  # 25% beams
    mask[2, :] = torch.empty(H, 1).bernoulli_(0.5)  # random 50% beams
    mask[3, :] = torch.empty(H, W).bernoulli_(0.1)  # random 10% points
    x_in = mask * x_orig + (1 - mask) * -1

    # =================================================================================
    # Completion
    # =================================================================================
    x_out = ddpm.repaint(
        known=x_in,
        mask=mask,
        num_steps=args.num_steps,
        num_resample_steps=args.num_resample_steps,
        jump_length=args.jump_length,
        rng=utils.inference.setup_rng(range(batch_size), device=device),
    ).clamp(-1, 1)

    # =================================================================================
    # Semantic segmentation
    # =================================================================================

    logits = semseg(make_semseg_inputs(x_out))
    labels = logits.argmax(dim=1, keepdim=True)

    # =================================================================================
    # Visualize
    # =================================================================================

    def to_img(x):
        img = lidar_utils.denormalize(x)
        img[:, [0]] = lidar_utils.revert_depth(img[:, [0]]) / lidar_utils.max_depth
        return img.clamp(0, 1)

    def to_bev(x, colors=None):
        R, t = utils.render.make_Rt(
            pitch=torch.pi / 4, yaw=torch.pi / 4, z=0.6, device=x.device
        )
        depth = lidar_utils.revert_depth(lidar_utils.denormalize(x)[:, [0]])
        xyz = lidar_utils.to_xyz(depth) / lidar_utils.max_depth
        if colors is None:
            z_min, z_max = -2 / lidar_utils.max_depth, 0.5 / lidar_utils.max_depth
            z = (xyz[:, [2]] - z_min) / (z_max - z_min)
            colors = utils.render.colorize(z.clamp(0, 1), cm.viridis) / 255
        points = einops.rearrange(xyz, "B C H W -> B (H W) C")
        colors = 1 - einops.rearrange(colors, "B C H W -> B (H W) C")
        bev = 1 - utils.render.render_point_clouds(
            points=points, colors=colors, R=R, t=t
        )
        bev = einops.rearrange(bev, "B C H W -> B H W C")
        return bev.cpu().clamp(0, 1)

    img_in = einops.rearrange(to_img(x_in), "B C H W -> B (C H) W 1").cpu()
    bev_in = to_bev(x_in)
    img_out = einops.rearrange(to_img(x_out), "B C H W -> B (C H) W 1").cpu()
    colors = utils.render.colorize(labels.float() / 19, cmap) / 255
    img_cls = einops.rearrange(colors, "B C H W -> B H W C").cpu()
    bev_out = to_bev(x_out, colors)

    fig, ax = plt.subplots(
        nrows=5,
        ncols=batch_size,
        figsize=(13, 9),
        gridspec_kw={"height_ratios": [H * 2, W, H * 2, H, W]},
        constrained_layout=True,
    )
    kwargs = dict(interpolation="none", vmin=0, vmax=1)
    for i in range(batch_size):
        ax[0][i].imshow(img_in[i], cmap="turbo", **kwargs)
        ax[1][i].imshow(bev_in[i], **kwargs)
        ax[2][i].imshow(img_out[i], cmap="turbo", **kwargs)
        ax[3][i].imshow(img_cls[i], **kwargs)
        ax[4][i].imshow(bev_out[i], **kwargs)
    ax[1][0].set_ylabel("Input")
    ax[4][0].set_ylabel("Completion & Segmentation")
    ax[0][0].set_title(r"$64\times1024$ (full)")
    ax[0][1].set_title("25% beams")
    ax[0][2].set_title("Random 50% beams")
    ax[0][3].set_title("Random 10% points")
    for a in ax.flatten():
        a.set_xticks([])
        a.set_yticks([])
    save_path = f"completion_T-{args.num_steps:04d}_r-{args.num_resample_steps:04d}_j-{args.jump_length:04d}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f'Saved to "{save_path}"')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--num_resample_steps", type=int, default=16)
    parser.add_argument("--jump_length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_id", type=int, default=-1)
    args = parser.parse_args()
    print(vars(args))
    main(args)
