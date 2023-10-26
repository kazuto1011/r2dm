from pathlib import Path

import datasets as ds
import numba
import numpy as np

_CITATION = """
@article{liao2022kitti,
    title        = {KITTI-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d},
    author       = {Liao, Yiyi and Xie, Jun and Geiger, Andreas},
    journal      = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
    volume       = 45,
    number       = 3,
    pages        = {3292--3310}
    year         = 2022,
}
"""

_SEQUENCE_SPLITS = {
    "lidargen": {
        ds.Split.TRAIN: [3, 4, 5, 6, 7, 9, 10],
        ds.Split.TEST: [0, 2],
    }
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


class KITTI360(ds.GeneratorBasedBuilder):
    """KITTI 360 dataset"""

    BUILDER_CONFIGS = [
        # 64x2048
        ds.BuilderConfig(
            name="unfolding-2048",
            description="scan unfolding, 64x2048 resolution",
            data_dir="data/kitti_360/dataset/data_3d_raw",
        ),
        ds.BuilderConfig(
            name="spherical-2048",
            description="spherical projection, 64x2048 resolution",
            data_dir="data/kitti_360/dataset/data_3d_raw",
        ),
        # 64x1024
        ds.BuilderConfig(
            name="unfolding-1024",
            description="scan unfolding, 64x1024 resolution",
            data_dir="data/kitti_360/dataset/data_3d_raw",
        ),
        ds.BuilderConfig(
            name="spherical-1024",
            description="spherical projection, 64x1024 resolution",
            data_dir="data/kitti_360/dataset/data_3d_raw",
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
        for split, subsets in _SEQUENCE_SPLITS["lidargen"].items():
            file_paths = list()
            for subset in subsets:
                wildcard = f"*_{subset:04d}_sync/velodyne_points/data/*.bin"
                file_paths += sorted(Path(self.config.data_dir).glob(wildcard))
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
            yield sample_id, {
                "sample_id": sample_id,
                "xyz": xyzrdm[:3],
                "reflectance": xyzrdm[[3]],
                "depth": xyzrdm[[4]],
                "mask": xyzrdm[[5]],
            }
