from torch.hub import load_state_dict_from_url

from metrics.extractor.rangenet import build_rangenet as _build_rangenet
from metrics.extractor.rangenet import crf_rnn as _crf_rnn
from metrics.extractor.rangenet import knn as _knn
from metrics.extractor.rangenet import rangenet21 as _rangenet21
from metrics.extractor.rangenet import rangenet53 as _rangenet53
from utils.inference import setup_model as _setup_model

dependencies = ["torch", "torchvision", "numpy", "einops", "tqdm", "pydantic"]

# =================================================================================
# Pre-trained R2DM
# =================================================================================


def _get_r2dm_url(key: str) -> str:
    return f"https://github.com/kazuto1011/r2dm/releases/download/weights/{key}.pth"


def pretrained_r2dm(config: str = "r2dm-h-kitti360-300k", ckpt: str = None, **kwargs):
    """
    R2DM models proposed in "LiDAR Data Synthesis with Denoising Diffusion Probabilistic Models" https://arxiv.org/abs/2309.09256
    Please refer to the project release page for available pre-trained weights: https://github.com/kazuto1011/r2dm/releases/tag/weights

    Args:
        config (str): Configuration string. (default: "r2dm-h-kitti360-300k")
        ckpt (str): Path to a checkpoint file. If specified, config will be ignored. (default: None)
        **kwargs: Additional keyword arguments for model setup.

    Returns:
        tuple: A tuple of the model, LiDAR utilities, and a configuration dict.
    """
    if ckpt is None:
        ckpt = load_state_dict_from_url(_get_r2dm_url(config), map_location="cpu")
    ddpm, lidar_utils, cfg = _setup_model(ckpt, **kwargs)
    return ddpm, lidar_utils, cfg


# =================================================================================
# Bonus! RangeNet++ ported from https://github.com/PRBonn/lidar-bonnetal
# =================================================================================


def rangenet(url_or_file: str, **kwargs):
    """
    Dynamic building of RangeNet-21/53

    Args:
        url_or_file (str): URL or local path to the checkpoint file (*.tar.gz).

    Returns:
        tuple: A tuple of the model and a preprocessing function.
    """
    model, preprocess = _build_rangenet(url_or_file, **kwargs)
    return model, preprocess


def rangenet21(weights: str = "SemanticKITTI_64x2048", **kwargs):
    """
    RangeNet-21 pre-trained on SemanticKITTI.

    Args:
        weights (str): "SemanticKITTI_64x2048"

    Returns:
        tuple: A tuple of the model and a preprocessing function.
    """
    model, preprocess = _rangenet21(weights, **kwargs)
    return model, preprocess


def rangenet53(weights: str = "SemanticKITTI_64x2048", **kwargs):
    """
    RangeNet-53 pre-trained on SemanticKITTI.

    Args:
        weights (str): "SemanticKITTI_64x2048", "SemanticKITTI_64x1024", "SemanticKITTI_64x512"

    Returns:
        tuple: A tuple of the model and a preprocessing function.
    """
    model, preprocess = _rangenet53(weights, **kwargs)
    return model, preprocess


def knn(num_classes: int = 20, **kwargs):
    """
    KNN post-processing for RangeNet++.

    Args:
        num_classes (int): Number of classes (default: 20)
    """
    return _knn(num_classes, **kwargs)


def crf_rnn(num_classes: int = 20, **kwargs):
    """
    CRF-RNN post-processing for RangeNet++.

    Args:
        num_classes (int): Number of classes (default: 20)
    """
    return _crf_rnn(num_classes, **kwargs)
