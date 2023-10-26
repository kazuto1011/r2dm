import io
import os
import random
import sys
import tarfile
from collections import OrderedDict
from typing import Any, Literal
from urllib.parse import urlparse

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _count_in_ch(inputs: list[str]):
    channels = {"xyz": 3, "range": 1, "remission": 1, "mask": 1}
    in_ch = 0
    for modality in inputs:
        in_ch += channels[modality]
    return in_ch


class ConvNormLReLU(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        padding: int,
        momentum: float,
        transposed: bool = False,
        bias: bool = False,
    ):
        conv_cls = nn.ConvTranspose2d if transposed else nn.Conv2d
        super().__init__(
            conv_cls(in_ch, out_ch, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.LeakyReLU(0.1),
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, momentum: float = 0.1):
        super().__init__()
        self.residual = nn.Sequential(
            ConvNormLReLU(in_ch, mid_ch, 1, 1, 0, momentum),
            ConvNormLReLU(mid_ch, out_ch, 3, 1, 1, momentum),
        )

    def forward(self, h):
        return h + self.residual(h)


class Block(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        momentum: float,
        mode: Literal["same", "up", "down"] = "same",
    ):
        super().__init__()

        if mode == "same":
            kernel_size = (3, 3)
            stride = (1, 1)
            padding = (1, 1)
        elif mode == "up":
            kernel_size = (1, 4)
            stride = (1, 2)
            padding = (0, 1)
        elif mode == "down":
            kernel_size = (3, 3)
            stride = (1, 2)
            padding = (1, 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.conv = ConvNormLReLU(
            in_ch=in_ch,
            out_ch=out_ch,
            momentum=momentum,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            transposed=mode == "up",
            bias=mode == "up",  # following the official
        )
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.residual_blocks.append(ResidualBlock(out_ch, in_ch, out_ch, momentum))

    def forward(self, x):
        h = self.conv(x)
        for block in self.residual_blocks:
            h = block(h)
        return h


class RangeNet(nn.Module):
    def __init__(
        self,
        inputs: list[str],
        num_classes: int,
        momentum: list[float] | float = 0.01,
        dropout: list[float] | float = 0.05,
        backbone: Literal[21, 53] = 53,
    ):
        super().__init__()
        self.inputs = inputs
        self.num_classes = num_classes
        self.in_ch = _count_in_ch(inputs)
        self.backbone = backbone
        assert backbone in (21, 53)
        num_resblocks = {21: [1, 1, 2, 2, 1], 53: [1, 2, 8, 8, 4]}[backbone]
        ch = lambda i: 32 << i
        momentum = nn.modules.utils._pair(momentum)
        dropout = nn.modules.utils._triple(dropout)

        self.stem = ConvNormLReLU(self.in_ch, 32, 3, 1, 1, momentum[0])
        self.enc1 = Block(ch(0), ch(1), num_resblocks[0], momentum[0], "down")
        self.enc2 = Block(ch(1), ch(2), num_resblocks[1], momentum[0], "down")
        self.enc3 = Block(ch(2), ch(3), num_resblocks[2], momentum[0], "down")
        self.enc4 = Block(ch(3), ch(4), num_resblocks[3], momentum[0], "down")
        self.enc5 = Block(ch(4), ch(5), num_resblocks[4], momentum[0], "down")
        self.dropout1 = nn.Dropout2d(p=dropout[0])

        self.dec5 = Block(ch(5), ch(4), 1, momentum[1], "up")
        self.dec4 = Block(ch(4), ch(3), 1, momentum[1], "up")
        self.dec3 = Block(ch(3), ch(2), 1, momentum[1], "up")
        self.dec2 = Block(ch(2), ch(1), 1, momentum[1], "up")
        self.dec1 = Block(ch(1), ch(0), 1, momentum[1], "up")
        self.dropout2 = nn.Dropout2d(p=dropout[1])  # following the official
        self.head = nn.Sequential(
            nn.Dropout2d(p=dropout[2]),
            nn.Conv2d(ch(0), num_classes, 3, 1, 1),
        )

    def flatten_and_subsample(self, fmaps):
        B, C, H, W = fmaps.shape
        feats = fmaps.view(B, C * H * W)
        random.seed(0)
        feats = feats[:, random.sample(range(C * H * W), 4096)]
        return feats

    def forward(self, img, feature=None):
        h0 = self.stem(img)
        h1 = self.enc1(h0)
        h2 = self.enc2(self.dropout1(h1))
        h3 = self.enc3(self.dropout1(h2))
        h4 = self.enc4(self.dropout1(h3))
        h5 = self.enc5(self.dropout1(h4))
        h = self.dec5(self.dropout1(h5)) + h4.detach()
        h = self.dec4(h) + h3.detach()
        h = self.dec3(h) + h2.detach()
        h = self.dec2(h) + h1.detach()
        h = self.dec1(h) + h0.detach()
        if feature == "lidargen":
            return self.flatten_and_subsample(h)
        logit = self.head(self.dropout2(h))
        return logit

    def extra_repr(self):
        return f"inputs={self.inputs}"


def _get_gaussian_kernel(kernel_size, sigma, device="cpu"):
    H, W = nn.modules.utils._pair(kernel_size)
    assert H % 2 == 1 and W % 2 == 1, "must be odd"
    hs = torch.arange(H, device=device) - H // 2
    ws = torch.arange(W, device=device) - W // 2
    coord = torch.meshgrid(hs, ws, indexing="ij")
    pdist = torch.stack(coord, dim=-1).pow(2).sum(dim=-1)
    kernel = torch.exp(-pdist / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


class kNN(nn.Module):
    """
    from https://github.com/kazuto1011/dusty-gan-v2/tree/main/semseg

    - Simplified version of k-NN filtering introduced in RangeNet++ [Milioto et al. IROS 2019]
    - Reference: https://github.com/PRBonn/lidar-bonnetal/blob/master/train/tasks/semantic/postproc/KNN.py
    """

    def __init__(self, num_classes, k=3, kernel_size=3, sigma=1.0, cutoff=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.sigma = sigma
        self.cutoff = cutoff

        # inverse gaussian kernel
        gaussian_kernel = _get_gaussian_kernel(self.kernel_size, self.sigma)
        self.register_buffer("dist_kernel", (1 - gaussian_kernel)[None, None])

    def forward(self, depth, label):
        B, C, H, W = depth.shape
        device = depth.device

        # point-wise distance
        depth_anchor = einops.rearrange(depth, "B C H W -> B C 1 (H W)")
        depth_neighbor = F.unfold(depth, self.kernel_size, padding=self.padding)
        depth_neighbor = einops.rearrange(depth_neighbor, "B (C K) HW -> B C K HW", C=C)
        depth_neighbor[depth_neighbor < 0] = float("inf")
        jump = torch.abs(depth_neighbor - depth_anchor)  # -> B C K HW

        # penalize far pixels
        jump = einops.rearrange(jump, "B C K (H W) -> B (C K) H W", H=H, W=W)
        kernel = self.dist_kernel.repeat_interleave(jump.shape[1], dim=0)
        dist = F.conv2d(jump, kernel, padding=self.padding, groups=kernel.shape[0])
        dist = einops.rearrange(dist, "B (C K) H W -> B C K (H W)", C=C)

        # find nearest points
        _, ids_topk = dist.topk(k=self.k, dim=2, largest=False, sorted=False)

        # gather labels
        label = label[:, None].float()  # add channel dim
        label_neighbor = F.unfold(label, self.kernel_size, padding=self.padding)
        label_neighbor = einops.rearrange(label_neighbor, "B (1 K) HW -> B 1 K HW")
        label_topk = label_neighbor.gather(dim=2, index=ids_topk)

        # cutoff
        if self.cutoff > 0:
            dist_topk = dist.gather(dim=2, index=ids_topk)
            label_topk[dist_topk > self.cutoff] = self.num_classes

        # majority voting
        ones = torch.ones_like(label_topk).to(depth)
        label_bins = torch.zeros(B, 1, self.num_classes + 1, H * W, device=device)
        label_bins.scatter_add_(dim=2, index=label_topk.long(), src=ones)
        refined_label = label_bins[:, :, :-1].argmax(dim=2)
        refined_label = einops.rearrange(refined_label, "B 1 (H W) -> B H W", H=H, W=W)

        return refined_label


class CRFRNN(nn.Module):
    """
    from https://github.com/kazuto1011/dusty-gan-v2/tree/main/semseg

    - CRF-RNN [Zheng et al. ICCV'15] used in SqueezeSeg [Wu et al. ICRA'18]
    """

    def __init__(
        self,
        num_classes,
        kernel_size=(3, 5),
        init_weight_smoothness=0.02,
        init_weight_appearance=0.1,
        theta_gamma=0.9,
        theta_alpha=0.9,
        theta_beta=0.015,
        num_iters=3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_iters = num_iters
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        _fn = nn.modules.utils._ntuple(self.num_classes)
        self.register_buffer("theta_gamma", torch.tensor(_fn(theta_gamma)))
        self.register_buffer("theta_alpha", torch.tensor(_fn(theta_alpha)))
        self.register_buffer("theta_beta", torch.tensor(_fn(theta_beta)))

        # fixed smoothness kernel
        kernel_gamma = self.get_smoothness_kernel(self.kernel_size, self.theta_gamma)
        self.register_buffer("kernel_gamma", kernel_gamma)
        kernel_alpha = self.get_smoothness_kernel(self.kernel_size, self.theta_alpha)
        self.register_buffer("kernel_alpha", kernel_alpha)

        # trainable label-wise weights for balancing kernels
        init_kernel_weights = lambda scale: torch.ones(1, num_classes, 1, 1) * scale
        weight_appearance = init_kernel_weights(init_weight_appearance)
        self.register_parameter("weight_appearance", nn.Parameter(weight_appearance))
        weight_smoothness = init_kernel_weights(init_weight_smoothness)
        self.register_parameter("weight_smoothness", nn.Parameter(weight_smoothness))

        # trainable label compatibility
        init_potts_model = 1 - torch.eye(num_classes)[..., None, None]  # [i!=j]
        self.label_compatibility = nn.Conv2d(num_classes, num_classes, 1, 1, bias=False)
        self.label_compatibility.weight.data = init_potts_model

    def get_smoothness_kernel(self, kernel_size, theta, device="cpu"):
        H, W = nn.modules.utils._pair(kernel_size)
        assert H % 2 == 1 and W % 2 == 1, "must be odd"
        hs = torch.arange(H, device=device) - H // 2
        ws = torch.arange(W, device=device) - W // 2
        coord = torch.meshgrid(hs, ws, indexing="ij")
        pdist = torch.stack(coord, dim=-1).pow(2).sum(dim=-1)
        kernel = torch.zeros(self.num_classes, self.num_classes, H, W)
        for c in range(self.num_classes):
            _kernel = torch.exp(-pdist / (2 * theta[c] ** 2))
            _kernel[H // 2, W // 2] = 0  # do not penalize the center
            kernel[c, c] = _kernel
        return kernel

    def apply(self, fn):
        return self  # do nothing

    def unfold_neighbors(self, x):
        B, C, H, W = x.shape
        # unfolding pixels within a kernel
        x = F.unfold(x, self.kernel_size, padding=self.padding)
        x = einops.rearrange(x, "B (C K) HW -> B C K HW", C=C)
        # exluding the kernel center
        kernel_numel = x.shape[2]  # == np.prod(self.kernel_size)
        kernel_index = torch.arange(kernel_numel, device=x.device)
        kernel_index = kernel_index[kernel_index != kernel_numel // 2]
        x = x.index_select(dim=2, index=kernel_index)
        return x  # -> B C K-1 (H W)

    def precompute_kernel_beta(self, xyz):
        xyz_anchor = einops.rearrange(xyz, "B C H W -> B C 1 (H W)")
        xyz_neighbors = self.unfold_neighbors(xyz)  # -> B C K-1 (H W)
        pdist = (xyz_neighbors - xyz_anchor).pow(2).sum(dim=1, keepdim=True)
        theta = self.theta_beta[None, :, None, None]
        kernel = torch.exp(-pdist / (2 * theta**2))
        return kernel

    def message_passing_smoothness(self, Q, kernel):
        # gaussian filtering by group convolution
        assert kernel.shape[0] == self.num_classes
        return F.conv2d(Q, kernel, padding=self.padding)

    def message_passing_appearance(self, Q, kernel_beta, mask):
        masked_Q = Q * mask
        exp_appearance = torch.ones_like(masked_Q).flatten(2)  # B C (H W)
        for i in range(exp_appearance.shape[0]):
            # sample-by-sample basis due to high memory requirements
            Q_neighbors_i = self.unfold_neighbors(masked_Q[[i]])  # B C K-1 (H W)
            exp_appearance[[i]] = (Q_neighbors_i * kernel_beta[[i]]).sum(dim=2)
        exp_appearance = exp_appearance.reshape_as(masked_Q) * mask  # B C H W
        exp_smoothness = self.message_passing_smoothness(Q, self.kernel_alpha)
        return exp_appearance * exp_smoothness  # bilateral kernel

    def weighting_kernels(self, k_smoothness, k_appearance):
        k_smoothness = self.weight_smoothness * k_smoothness
        k_appearance = self.weight_appearance * k_appearance
        return k_smoothness + k_appearance

    def forward(self, unary, xyz, mask):
        """
        unary: (B,N,H,W)
        xyz  : (B,3,H,W)
        mask : (B,H,W)
        """
        # initialization
        Q = unary
        kernel_beta = self.precompute_kernel_beta(xyz).detach()
        mask = mask[:, None] if mask.ndim == 3 else mask
        # mean-field approximation
        for _ in range(self.num_iters):
            # normalize
            Q = F.softmax(Q, dim=1)
            # message passing (#filters=2)
            k_smoothness = self.message_passing_smoothness(Q, self.kernel_gamma)
            k_appearance = self.message_passing_appearance(Q, kernel_beta, mask)
            weighted_k = self.weighting_kernels(k_smoothness, k_appearance)
            # compatibility transform
            pairwise = self.label_compatibility(weighted_k)
            # iterative update
            Q = unary - pairwise
        return Q


def _download_pretrained_weights(
    arch: str,
    progress=True,
):
    def _translate_param_name(src_name) -> str:
        src_name: list[str] = src_name.split(".")
        tgt_name: list[str] = src_name
        if src_name[0] == "1":
            tgt_name[0] = "head.1"
        elif src_name[0] == "conv1":
            tgt_name[0] = "stem.0"
        elif src_name[0] == "bn1":
            tgt_name[0] = "stem.1"
        elif src_name[1] in ("conv", "upconv"):
            tgt_name[1] = "conv.0"
        elif src_name[1] in ("bn"):
            tgt_name[1] = "conv.1"
        elif src_name[1] == "residual":
            tgt_name[1] = "residual_blocks.0.residual"
            if src_name[2].startswith("conv"):
                n = int(src_name[2][-1])
                tgt_name[2] = f"{n-1}.0"
            elif src_name[2].startswith("bn"):
                n = int(src_name[2][-1])
                tgt_name[2] = f"{n-1}.1"
        elif src_name[1].startswith("residual_"):
            n = int(src_name[1].split("_")[-1])
            tgt_name[1] = f"residual_blocks.{n}.residual"
            if src_name[2].startswith("conv"):
                n = int(src_name[2][-1])
                tgt_name[2] = f"{n-1}.0"
            elif src_name[2].startswith("bn"):
                n = int(src_name[2][-1])
                tgt_name[2] = f"{n-1}.1"
        return ".".join(tgt_name)

    # set the cache directory
    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    # download the tar file
    url = f"http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/{arch}.tar.gz"
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        torch.hub.download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    # parse the downloaded tar file
    state_dict = OrderedDict()
    with tarfile.open(cached_file, "r:gz") as tar:
        members = list(map(lambda m: m.name, tar.getmembers()))
        for member in (
            f"{arch}/backbone",
            f"{arch}/segmentation_decoder",
            f"{arch}/segmentation_head",
        ):
            assert member in members
            file = tar.extractfile(member)
            _state_dict = torch.load(io.BytesIO(file.read()), map_location="cpu")
            for name, params in _state_dict.items():
                new_name = _translate_param_name(name)
                assert new_name not in state_dict, new_name
                state_dict[new_name] = params.cpu()
    return state_dict


class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = torchvision.transforms.Normalize(
            # range, x, y, z, remission
            mean=[12.12, 10.88, 0.23, -1.04, 0.21],
            std=[12.32, 11.47, 6.91, 0.86, 0.16],
        )

    def forward(self, img: torch.Tensor, mask: torch.Tensor):
        assert img.ndim == mask.ndim == 4
        assert img.shape[1] == 5
        return self.transforms(img) * mask


_ID2LABEL = {
    0: "unlabeled",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign",
}


def _build_model(
    inputs: list[str],
    num_classes: int,
    backbone: int,
    weights: str | None,
    compile: bool,
    device: str = "cpu",
    **kwargs: Any,
):
    model = RangeNet(
        inputs=inputs,
        num_classes=num_classes,
        backbone=backbone,
        **kwargs,
    )
    if weights is not None:
        state_dict = _download_pretrained_weights(weights)
        model.load_state_dict(state_dict, strict=True)
        model.eval().requires_grad_(False)
    preprocess = Preprocess()
    if compile:
        model = torch.compile(model)
        preprocess = torch.compile(preprocess)
    return model.to(device), preprocess.to(device)


def rangenet21(
    weights: str | None = None,
    compile: bool = False,
    device: str = "cpu",
    **kwargs: Any,
):
    """RangeNet with Darknet21 backbone from `RangeNet++: Fast and Accurate LiDAR Semantic Segmentation`."""
    base_kwargs = {
        "SemanticKITTI_64x2048": {
            "inputs": ["range", "xyz", "remission"],
            "num_classes": 20,
            "backbone": 21,
            "weights": "darknet21",
        },
        None: {
            "backbone": 21,
            "weights": None,
        },
    }[weights]
    return _build_model(compile=compile, device=device, **base_kwargs, **kwargs)


def rangenet53(
    weights: str | None = None,
    compile: bool = False,
    device: str = "cpu",
    **kwargs: Any,
):
    """RangeNet with Darknet53 backbone from `RangeNet++: Fast and Accurate LiDAR Semantic Segmentation`."""
    base_kwargs = {
        "SemanticKITTI_64x2048": {
            "inputs": ["range", "xyz", "remission"],
            "num_classes": 20,
            "backbone": 53,
            "weights": "darknet53",
        },
        "SemanticKITTI_64x1024": {
            "inputs": ["range", "xyz", "remission"],
            "num_classes": 20,
            "backbone": 53,
            "weights": "darknet53-1024",
        },
        "SemanticKITTI_64x512": {
            "inputs": ["range", "xyz", "remission"],
            "num_classes": 20,
            "backbone": 53,
            "weights": "darknet53-512",
        },
        None: {
            "backbone": 53,
            "weights": None,
        },
    }[weights]
    return _build_model(compile=compile, device=device, **base_kwargs, **kwargs)


def knn(num_classes: int, **kwargs: Any) -> kNN:
    """kNN post-processor from `RangeNet++: Fast and Accurate LiDAR Semantic Segmentation`."""
    postprocess = kNN(num_classes=num_classes, **kwargs)
    return postprocess


def crf_rnn(num_classes: int, **kwargs: Any) -> CRFRNN:
    """CRF-RNN post-processor from `Conditional Random Fields as Recurrent Neural Networks`."""
    postprocess = CRFRNN(num_classes=num_classes, **kwargs)
    return postprocess


def make_semantickitti_cmap():
    from matplotlib.colors import ListedColormap

    label_colors = {
        0: [0, 0, 0],
        1: [245, 150, 100],
        2: [245, 230, 100],
        3: [150, 60, 30],
        4: [180, 30, 80],
        5: [255, 0, 0],
        6: [30, 30, 255],
        7: [200, 40, 255],
        8: [90, 30, 150],
        9: [255, 0, 255],
        10: [255, 150, 255],
        11: [75, 0, 75],
        12: [75, 0, 175],
        13: [0, 200, 255],
        14: [50, 120, 255],
        15: [0, 175, 0],
        16: [0, 60, 135],
        17: [80, 240, 150],
        18: [150, 240, 255],
        19: [0, 0, 255],
    }
    num_classes = max(label_colors.keys()) + 1
    label_colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    for label_id, color in label_colors.items():
        label_colormap[label_id] = color[::-1]  # BGR -> RGB
    cmap = ListedColormap(label_colormap / 255.0)
    return cmap
