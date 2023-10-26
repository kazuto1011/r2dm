import argparse
import datetime
import json
import pickle
import random
from pathlib import Path

import datasets as ds
import einops
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import utils.inference
from metrics import bev, distribution
from metrics.extractor import pointnet, rangenet

# from LiDARGen
EVAL_MAX_DEPTH = 63.0
EVAL_MIN_DEPTH = 0.5
DATASET_MAX_DEPTH = 80.0


def resize(x, size):
    return F.interpolate(x, size=size, mode="nearest-exact")


class Features10k(torch.utils.data.Dataset):
    def __init__(self, root, helper, train_reflectance, train_mask):
        self.sample_path_list = sorted(Path(root).glob("*.pth"))[:10_000]
        self.helper = helper
        self.train_reflectance = train_reflectance
        self.train_mask = train_mask

    def __getitem__(self, index):
        sample_path = self.sample_path_list[index]
        img = torch.load(sample_path, map_location="cpu")
        assert img.shape[0] == 5, img.shape
        depth = img[[0]]
        mask = torch.logical_and(depth > EVAL_MIN_DEPTH, depth < EVAL_MAX_DEPTH).float()
        img = img * mask
        return img.float(), mask.float()

    def __len__(self):
        return len(self.sample_path_list)


@torch.no_grad()
def evaluate(args):
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, lidar_utils, cfg = utils.inference.setup_model(args.ckpt, device=device)
    lidar_utils.to(device)

    print(f"{cfg.train_reflectance=}")
    print(f"{cfg.train_mask=}")

    H, W = lidar_utils.resolution
    model_img, preprocess_img = rangenet.rangenet53(
        weights=f"SemanticKITTI_{H}x{W}", device=device, compile=True
    )
    model_pts = pointnet.pretrained_pointnet(
        dataset="shapenet", device=device, compile=True
    )

    if args.dataset == "test":
        split = ds.Split.TEST
    elif args.dataset == "train":
        split = ds.Split.TRAIN
    elif args.dataset == "all":
        split = ds.Split.ALL
    else:
        raise ValueError

    dataset = ds.load_dataset(
        path=f"data/{cfg.dataset}",
        name=cfg.lidar_projection,
        split=split,
        num_proc=args.num_workers,
    ).with_format("torch")
    print(dataset)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    results = dict(img=dict(), pts=dict(), bev=dict(), info=dict())
    results["info"]["phase"] = args.dataset
    results["info"]["directory"] = args.sample_dir

    # =====================================================
    # real set
    # =====================================================

    cache_file_path = (
        f"real_set_{cfg.dataset}_{cfg.lidar_projection}_{H}x{W}_{args.dataset}.pkl"
    )
    if Path(cache_file_path).exists():
        print(f"found cached {cache_file_path}")
        real_set = pickle.load(open(cache_file_path, "rb"))
    else:
        real_set = dict(img_feats=list(), pts_feats=list(), bev_hists=list())
        for batch in tqdm(loader, desc="real"):
            depth = resize(batch["depth"], (H, W)).to(device)
            xyz = resize(batch["xyz"], (H, W)).to(device)
            rflct = resize(batch["reflectance"], (H, W)).to(device)
            mask = resize(batch["mask"], (H, W)).to(device)
            mask = mask * torch.logical_and(
                depth > EVAL_MIN_DEPTH, depth < EVAL_MAX_DEPTH
            )
            imgs_frd = torch.cat([depth, xyz, rflct], dim=1)
            with torch.inference_mode():
                feats_img = model_img(
                    preprocess_img(imgs_frd, mask), feature="lidargen"
                )
            real_set["img_feats"].append(feats_img.cpu())

            pcs = einops.rearrange(xyz * mask, "B C H W -> B C (H W)")
            for pc in pcs:
                pc = einops.rearrange(pc, "C N -> N C")
                hist = bev.point_cloud_to_histogram(pc)
                real_set["bev_hists"].append(hist.cpu())

            with torch.inference_mode():
                feats_pts = model_pts(pcs / DATASET_MAX_DEPTH)
            real_set["pts_feats"].append(feats_pts.cpu())

        real_set["img_feats"] = torch.cat(real_set["img_feats"], dim=0).numpy()
        real_set["pts_feats"] = torch.cat(real_set["pts_feats"], dim=0).numpy()
        real_set["bev_hists"] = torch.stack(real_set["bev_hists"], dim=0).numpy()
        pickle.dump(real_set, open(cache_file_path, "wb"))

    results["info"]["#real"] = len(real_set["pts_feats"])

    # =====================================================
    # gen set
    # =====================================================

    dataset = Features10k(
        args.sample_dir,
        helper=lidar_utils.cpu(),
        train_reflectance=cfg.train_reflectance,
        train_mask=cfg.train_mask,
    )
    gen_loader = DataLoader(dataset, batch_size=64, num_workers=4)
    gen_set = dict(img_feats=list(), pts_feats=list(), bev_hists=list())
    for imgs_frd, mask in tqdm(gen_loader, desc="gen"):
        imgs_frd, mask = imgs_frd.to(device), mask.to(device)
        if cfg.train_reflectance:
            with torch.inference_mode():
                feats_img = model_img(
                    preprocess_img(imgs_frd, mask), feature="lidargen"
                )
            gen_set["img_feats"].append(feats_img.cpu())

        xyz = imgs_frd[:, 1:4]
        pcs = einops.rearrange(xyz * mask, "B C H W -> B C (H W)")
        for pc in pcs:
            pc = einops.rearrange(pc, "C N -> N C")
            hist = bev.point_cloud_to_histogram(pc)
            gen_set["bev_hists"].append(hist.cpu())

        with torch.inference_mode():
            feats_pts = model_pts(pcs / DATASET_MAX_DEPTH)
        gen_set["pts_feats"].append(feats_pts.cpu())

    if cfg.train_reflectance:
        gen_set["img_feats"] = torch.cat(gen_set["img_feats"], dim=0).numpy()
    gen_set["pts_feats"] = torch.cat(gen_set["pts_feats"], dim=0).numpy()
    gen_set["bev_hists"] = torch.stack(gen_set["bev_hists"], dim=0).numpy()

    results["info"]["#fake"] = len(gen_set["pts_feats"])

    # =====================================================
    # evaluation
    # =====================================================
    torch.cuda.empty_cache()

    if cfg.train_reflectance:
        results["img"]["frechet_distance"] = distribution.compute_frechet_distance(
            real_set["img_feats"], gen_set["img_feats"]
        )
        results["img"]["squared_mmd"] = distribution.compute_squared_mmd(
            real_set["img_feats"], gen_set["img_feats"]
        )

    results["pts"]["frechet_distance"] = distribution.compute_frechet_distance(
        real_set["pts_feats"], gen_set["pts_feats"]
    )
    results["pts"]["squared_mmd"] = distribution.compute_squared_mmd(
        real_set["pts_feats"], gen_set["pts_feats"]
    )

    perm = list(range(len(real_set["bev_hists"])))
    random.Random(0).shuffle(perm)
    perm = perm[:10_000]

    results["bev"]["jsd"] = bev.compute_jsd_2d(
        torch.from_numpy(real_set["bev_hists"][perm]).to(device).float(),
        torch.from_numpy(gen_set["bev_hists"]).to(device).float(),
    )

    results["bev"]["mmd"] = bev.compute_mmd_2d(
        torch.from_numpy(real_set["bev_hists"][perm]).to(device).float(),
        torch.from_numpy(gen_set["bev_hists"]).to(device).float(),
    )

    print(results)

    save_path = (
        args.sample_dir + f"_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    )
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--dataset", choices=["train", "test", "all"], default="all")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    evaluate(args)
