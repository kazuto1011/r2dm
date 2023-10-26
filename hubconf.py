from torch.hub import load_state_dict_from_url

import utils.inference
from metrics.extractor.rangenet import crf_rnn as _crf_rnn
from metrics.extractor.rangenet import knn as _knn
from metrics.extractor.rangenet import rangenet21 as _rangenet21
from metrics.extractor.rangenet import rangenet53 as _rangenet53

dependencies = ["torch", "torchvision", "numpy", "einops", "tqdm"]

# =================================================================================
# Pre-trained R2DM
# =================================================================================


def _get_url(key: str) -> str:
    return f"https://github.com/kazuto1011/r2dm/releases/download/weights/{key}.pth"


def pretrained_r2dm(config: str = "r2dm-h-kitti360-300k", **kwargs):
    """
    Pre-trained R2DM:
    - `config`: `'r2dm-[a-h]-[dataset]-[steps]'` (default: `'r2dm-h-kitti360-300k'`)
    """
    ckpt = load_state_dict_from_url(_get_url(config), map_location="cpu")
    ddpm, lidar_utils, cfg = utils.inference.setup_model(ckpt, **kwargs)
    return ddpm, lidar_utils, cfg


# =================================================================================
# Bonus! RangeNet++ ported from https://github.com/PRBonn/lidar-bonnetal
# =================================================================================


def rangenet21(weights: str = "SemanticKITTI_64x2048", **kwargs):
    """
    RangeNet-21 pre-trained on SemanticKITTI:
    - `weights`: `'SemanticKITTI_64x2048'`, `None`
    """
    model, preprocess = _rangenet21(weights, **kwargs)
    return model, preprocess


def rangenet53(weights: str = "SemanticKITTI_64x2048", **kwargs):
    """
    RangeNet-53 pre-trained on SemanticKITTI:
    - `weights`: `'SemanticKITTI_64x2048'`, `'SemanticKITTI_64x1024'`, `'SemanticKITTI_64x512'`, `None`
    """
    model, preprocess = _rangenet53(weights, **kwargs)
    return model, preprocess


def knn(num_classes: int = 20, **kwargs):
    """
    KNN post-processing for RangeNet++:
    - `num_classes`: `20` for SemanticKITTI
    """
    return _knn(num_classes, **kwargs)


def crf_rnn(num_classes: int = 20, **kwargs):
    """
    CRF-RNN post-processing for RangeNet++:
    - `num_classes`: `20` for SemanticKITTI
    """
    return _crf_rnn(num_classes, **kwargs)
