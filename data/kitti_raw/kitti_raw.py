from pathlib import Path

import datasets as ds
import numba
import numpy as np

_CITATION = """\
@inproceedings{geiger2013vision,
    title={Vision meets robotics: The KITTI dataset},
    author={Geiger, Andreas and Lenz, Philip and Stiller, Christoph and Urtasun, Raquel},
    booktitle={The International Journal of Robotics Research},
    volume={32},
    number={11},
    pages={1231--1237},
    year={2013},
}
"""

_SEQUENCE_SPLITS = {
    ds.Split.TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    ds.Split.VALIDATION: [8],
    ds.Split.TEST: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}


_RAW_RECORDS = {
    "calibration": [
        "2011_09_26_drive_0119_sync",
        "2011_09_28_drive_0225_sync",
        "2011_09_29_drive_0108_sync",
        "2011_09_30_drive_0072_sync",
        "2011_10_03_drive_0058_sync",
    ],
    "campus": [
        "2011_09_28_drive_0016_sync",
        "2011_09_28_drive_0021_sync",
        "2011_09_28_drive_0034_sync",
        "2011_09_28_drive_0035_sync",
        "2011_09_28_drive_0037_sync",
        "2011_09_28_drive_0038_sync",
        "2011_09_28_drive_0039_sync",
        "2011_09_28_drive_0043_sync",
        "2011_09_28_drive_0045_sync",
        "2011_09_28_drive_0047_sync",
    ],
    "city": [
        "2011_09_26_drive_0001_sync",
        "2011_09_26_drive_0002_sync",
        "2011_09_26_drive_0005_sync",
        "2011_09_26_drive_0009_sync",
        "2011_09_26_drive_0011_sync",
        "2011_09_26_drive_0013_sync",
        "2011_09_26_drive_0014_sync",
        "2011_09_26_drive_0017_sync",
        "2011_09_26_drive_0018_sync",
        "2011_09_26_drive_0048_sync",
        "2011_09_26_drive_0051_sync",
        "2011_09_26_drive_0056_sync",
        "2011_09_26_drive_0057_sync",
        "2011_09_26_drive_0059_sync",
        "2011_09_26_drive_0060_sync",
        "2011_09_26_drive_0084_sync",
        "2011_09_26_drive_0091_sync",
        "2011_09_26_drive_0093_sync",
        "2011_09_26_drive_0095_sync",
        "2011_09_26_drive_0096_sync",
        "2011_09_26_drive_0104_sync",
        "2011_09_26_drive_0106_sync",
        "2011_09_26_drive_0113_sync",
        "2011_09_26_drive_0117_sync",
        "2011_09_28_drive_0001_sync",
        "2011_09_28_drive_0002_sync",
        "2011_09_29_drive_0026_sync",
        "2011_09_29_drive_0071_sync",
    ],
    "person": [
        "2011_09_28_drive_0053_sync",
        "2011_09_28_drive_0054_sync",
        "2011_09_28_drive_0057_sync",
        "2011_09_28_drive_0065_sync",
        "2011_09_28_drive_0066_sync",
        "2011_09_28_drive_0068_sync",
        "2011_09_28_drive_0070_sync",
        "2011_09_28_drive_0071_sync",
        "2011_09_28_drive_0075_sync",
        "2011_09_28_drive_0077_sync",
        "2011_09_28_drive_0078_sync",
        "2011_09_28_drive_0080_sync",
        "2011_09_28_drive_0082_sync",
        "2011_09_28_drive_0086_sync",
        "2011_09_28_drive_0087_sync",
        "2011_09_28_drive_0089_sync",
        "2011_09_28_drive_0090_sync",
        "2011_09_28_drive_0094_sync",
        "2011_09_28_drive_0095_sync",
        "2011_09_28_drive_0096_sync",
        "2011_09_28_drive_0098_sync",
        "2011_09_28_drive_0100_sync",
        "2011_09_28_drive_0102_sync",
        "2011_09_28_drive_0103_sync",
        "2011_09_28_drive_0104_sync",
        "2011_09_28_drive_0106_sync",
        "2011_09_28_drive_0108_sync",
        "2011_09_28_drive_0110_sync",
        "2011_09_28_drive_0113_sync",
        "2011_09_28_drive_0117_sync",
        "2011_09_28_drive_0119_sync",
        "2011_09_28_drive_0121_sync",
        "2011_09_28_drive_0122_sync",
        "2011_09_28_drive_0125_sync",
        "2011_09_28_drive_0126_sync",
        "2011_09_28_drive_0128_sync",
        "2011_09_28_drive_0132_sync",
        "2011_09_28_drive_0134_sync",
        "2011_09_28_drive_0135_sync",
        "2011_09_28_drive_0136_sync",
        "2011_09_28_drive_0138_sync",
        "2011_09_28_drive_0141_sync",
        "2011_09_28_drive_0143_sync",
        "2011_09_28_drive_0145_sync",
        "2011_09_28_drive_0146_sync",
        "2011_09_28_drive_0149_sync",
        "2011_09_28_drive_0153_sync",
        "2011_09_28_drive_0154_sync",
        "2011_09_28_drive_0155_sync",
        "2011_09_28_drive_0156_sync",
        "2011_09_28_drive_0160_sync",
        "2011_09_28_drive_0161_sync",
        "2011_09_28_drive_0162_sync",
        "2011_09_28_drive_0165_sync",
        "2011_09_28_drive_0166_sync",
        "2011_09_28_drive_0167_sync",
        "2011_09_28_drive_0168_sync",
        "2011_09_28_drive_0171_sync",
        "2011_09_28_drive_0174_sync",
        "2011_09_28_drive_0177_sync",
        "2011_09_28_drive_0179_sync",
        "2011_09_28_drive_0183_sync",
        "2011_09_28_drive_0184_sync",
        "2011_09_28_drive_0185_sync",
        "2011_09_28_drive_0186_sync",
        "2011_09_28_drive_0187_sync",
        "2011_09_28_drive_0191_sync",
        "2011_09_28_drive_0192_sync",
        "2011_09_28_drive_0195_sync",
        "2011_09_28_drive_0198_sync",
        "2011_09_28_drive_0199_sync",
        "2011_09_28_drive_0201_sync",
        "2011_09_28_drive_0204_sync",
        "2011_09_28_drive_0205_sync",
        "2011_09_28_drive_0208_sync",
        "2011_09_28_drive_0209_sync",
        "2011_09_28_drive_0214_sync",
        "2011_09_28_drive_0216_sync",
        "2011_09_28_drive_0220_sync",
        "2011_09_28_drive_0222_sync",
    ],
    "residential": [
        "2011_09_26_drive_0019_sync",
        "2011_09_26_drive_0020_sync",
        "2011_09_26_drive_0022_sync",
        "2011_09_26_drive_0023_sync",
        "2011_09_26_drive_0035_sync",
        "2011_09_26_drive_0036_sync",
        "2011_09_26_drive_0039_sync",
        "2011_09_26_drive_0046_sync",
        "2011_09_26_drive_0061_sync",
        "2011_09_26_drive_0064_sync",
        "2011_09_26_drive_0079_sync",
        "2011_09_26_drive_0086_sync",
        "2011_09_26_drive_0087_sync",
        "2011_09_30_drive_0018_sync",
        "2011_09_30_drive_0020_sync",
        "2011_09_30_drive_0027_sync",
        "2011_09_30_drive_0028_sync",
        "2011_09_30_drive_0033_sync",
        "2011_09_30_drive_0034_sync",
        "2011_10_03_drive_0027_sync",
        "2011_10_03_drive_0034_sync",
    ],
    "road": [
        "2011_09_26_drive_0015_sync",
        "2011_09_26_drive_0027_sync",
        "2011_09_26_drive_0028_sync",
        "2011_09_26_drive_0029_sync",
        "2011_09_26_drive_0032_sync",
        "2011_09_26_drive_0052_sync",
        "2011_09_26_drive_0070_sync",
        "2011_09_26_drive_0101_sync",
        "2011_09_29_drive_0004_sync",
        "2011_09_30_drive_0016_sync",
        "2011_10_03_drive_0042_sync",
        "2011_10_03_drive_0047_sync",
    ],
}

_RAW_TRAINVAL = (
    "2011_10_03_drive_0027_sync",
    "2011_10_03_drive_0042_sync",
    "2011_10_03_drive_0034_sync",
    "2011_09_26_drive_0067_sync",
    "2011_09_30_drive_0016_sync",
    "2011_09_30_drive_0018_sync",
    "2011_09_30_drive_0020_sync",
    "2011_09_30_drive_0027_sync",
    "2011_09_30_drive_0028_sync",
    "2011_09_30_drive_0033_sync",
    "2011_09_30_drive_0034_sync",
)

_ODOMETRY_TO_RAW = {
    # sequence number, sequence name, start, end
    "00": ("2011_10_03_drive_0027_sync", int("000000"), int("004540")),
    "01": ("2011_10_03_drive_0042_sync", int("000000"), int("001100")),
    "02": ("2011_10_03_drive_0034_sync", int("000000"), int("004660")),
    "03": ("2011_09_26_drive_0067_sync", int("000000"), int("000800")),
    "04": ("2011_09_30_drive_0016_sync", int("000000"), int("000270")),
    "05": ("2011_09_30_drive_0018_sync", int("000000"), int("002760")),
    "06": ("2011_09_30_drive_0020_sync", int("000000"), int("001100")),
    "07": ("2011_09_30_drive_0027_sync", int("000000"), int("001100")),
    "08": ("2011_09_30_drive_0028_sync", int("001100"), int("005170")),
    "09": ("2011_09_30_drive_0033_sync", int("000000"), int("001590")),
    "10": ("2011_09_30_drive_0034_sync", int("000000"), int("001200")),
}


@numba.jit(nopython=True, parallel=False)
def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array


def load_points_as_images(
    point_path: str,
    scan_unfolding: bool = True,
    H: int = 64,
    W: int = 2048,
    min_depth: float = 1.45,
    max_depth: float = 80.0,
):
    # load xyz & intensity and add depth & mask
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    xyz = points[:, :3]  # xyz
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask = (depth >= min_depth) & (depth <= max_depth)
    points = np.concatenate([points, depth, mask], axis=1)

    if scan_unfolding:
        # the i-th quadrant
        # suppose the points are ordered counterclockwise
        quads = np.zeros_like(x, dtype=np.int32)
        quads[(x >= 0) & (y >= 0)] = 0  # 1st
        quads[(x < 0) & (y >= 0)] = 1  # 2nd
        quads[(x < 0) & (y < 0)] = 2  # 3rd
        quads[(x >= 0) & (y < 0)] = 3  # 4th

        # split between the 3rd and 1st quadrants
        diff = np.roll(quads, shift=1, axis=0) - quads
        delim_inds, _ = np.where(diff == 3)  # number of lines
        inds = list(delim_inds) + [len(points)]  # add the last index

        # vertical grid
        grid_h = np.zeros_like(x, dtype=np.int32)
        cur_ring_idx = H - 1  # ...0
        for i in reversed(range(len(delim_inds))):
            grid_h[inds[i] : inds[i + 1]] = cur_ring_idx
            if cur_ring_idx >= 0:
                cur_ring_idx -= 1
            else:
                break
    else:
        h_up, h_down = np.deg2rad(3), np.deg2rad(-25)
        elevation = np.arcsin(z / depth) + abs(h_down)
        grid_h = 1 - elevation / (h_up - h_down)
        grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)

    # horizontal grid
    azimuth = -np.arctan2(y, x)  # [-pi,pi]
    grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0,1]
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)

    grid = np.concatenate((grid_h, grid_w), axis=1)

    # projection
    order = np.argsort(-depth.squeeze(1))
    proj_points = np.zeros((H, W, 4 + 2), dtype=points.dtype)
    proj_points = scatter(proj_points, grid[order], points[order])

    return proj_points.astype(np.float32)


class KITTIRaw(ds.GeneratorBasedBuilder):
    """KITTI Raw dataset"""

    BUILDER_CONFIGS = [
        # 64x2048
        ds.BuilderConfig(
            name="unfolding-2048",
            description="scan unfolding, 64x2048 resolution",
            data_dir="data/kitti_raw/dataset",
        ),
        ds.BuilderConfig(
            name="spherical-2048",
            description="spherical projection, 64x2048 resolution",
            data_dir="data/kitti_raw/dataset",
        ),
        # 64x1024
        ds.BuilderConfig(
            name="unfolding-1024",
            description="scan unfolding, 64x1024 resolution",
            data_dir="data/kitti_raw/dataset",
        ),
        ds.BuilderConfig(
            name="spherical-1024",
            description="spherical projection, 64x1024 resolution",
            data_dir="data/kitti_raw/dataset",
        ),
    ]

    DEFAULT_CONFIG_NAME = "spherical-1024"

    def _parse_config_name(self):
        projection, width = self.config.name.split("-")
        return projection, int(width)

    def _info(self):
        _, width = self._parse_config_name()
        features = {
            "sample_id": ds.Value("int32"),
            "xyz": ds.Array3D((3, 64, width), "float32"),
            "reflectance": ds.Array3D((1, 64, width), "float32"),
            "depth": ds.Array3D((1, 64, width), "float32"),
            "mask": ds.Array3D((1, 64, width), "float32"),
        }
        return ds.DatasetInfo(features=ds.Features(features))

    def _split_generators(self, _):
        splits = list()
        for split, subsets in _SEQUENCE_SPLITS.items():
            file_paths = list()
            if split in (ds.Split.TRAIN, ds.Split.VALIDATION):
                for subset in subsets:
                    if subset == 3:
                        # kitti raw does not have 03 sequence
                        continue
                    subset_idx = f"{subset:02d}"
                    seq_name, start_idx, end_idx = _ODOMETRY_TO_RAW[subset_idx]
                    for point_idx in range(start_idx, end_idx + 1):
                        subset_dir = f"{self.config.data_dir}/{seq_name[:10]}/{seq_name}/velodyne_points/data"
                        file_paths.append(f"{subset_dir}/{point_idx:010d}.bin")
            elif split in (ds.Split.TEST,):
                for category in ["city", "road", "residential"]:
                    for seq_name in _RAW_RECORDS[category]:
                        if seq_name not in _RAW_TRAINVAL:
                            subset_dir = f"{self.config.data_dir}/{seq_name[:10]}/{seq_name}/velodyne_points/data"
                            file_paths += sorted(Path(subset_dir).glob("*.bin"))
            splits.append(
                ds.SplitGenerator(
                    name=split,
                    gen_kwargs={"items": list(zip(range(len(file_paths)), file_paths))},
                )
            )
        return splits

    def _generate_examples(self, items):
        projection, width = self._parse_config_name()
        for sample_id, file_path in items:
            xyzrdm = load_points_as_images(
                file_path,
                scan_unfolding=projection == "unfolding",
                W=width,
            )
            xyzrdm = xyzrdm.transpose(2, 0, 1)
            xyzrdm *= xyzrdm[[5]]
            yield (
                sample_id,
                {
                    "sample_id": sample_id,
                    "xyz": xyzrdm[:3],
                    "reflectance": xyzrdm[[3]],
                    "depth": xyzrdm[[4]],
                    "mask": xyzrdm[[5]],
                },
            )
