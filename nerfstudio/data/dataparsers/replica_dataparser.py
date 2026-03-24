from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, cast

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class ReplicaDataParserConfig(DataParserConfig):
    _target: type = field(default_factory=lambda: Replica)
    data: Path = Path("data/Replica/office0")
    scale_factor: float = 1.0
    scene_scale: float = 1.0
    center_method: Literal["poses", "focus", "none"] = "poses"
    auto_scale_poses: bool = True
    train_split_fraction: float = 0.9
    depth_unit_scale_factor: Optional[float] = None
    load_3D_points: bool = True
    point_cloud_color: bool = True
    max_3d_points: Optional[int] = 100000
    frame_stride: int = 1


class Replica(DataParser):
    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        del kwargs
        config = cast(ReplicaDataParserConfig, self.config)
        scene_dir = config.data
        results_dir = scene_dir / "results"
        traj_path = scene_dir / "traj.txt"
        intrinsics_path = scene_dir.parent / "cam_params.json"

        image_filenames = sorted(results_dir.glob("frame*.jpg"), key=lambda path: int(path.stem.replace("frame", "")))
        depth_filenames = sorted(results_dir.glob("depth*.png"), key=lambda path: int(path.stem.replace("depth", "")))
        pose_lines = traj_path.read_text(encoding="utf-8").strip().splitlines()

        if config.frame_stride > 1:
            image_filenames = image_filenames[:: config.frame_stride]
            depth_filenames = depth_filenames[:: config.frame_stride]
            pose_lines = pose_lines[:: config.frame_stride]

        num_frames = min(len(image_filenames), len(depth_filenames), len(pose_lines))
        image_filenames = image_filenames[:num_frames]
        depth_filenames = depth_filenames[:num_frames]
        pose_lines = pose_lines[:num_frames]

        if num_frames == 0:
            raise ValueError(f"No frames found in {results_dir}")

        intrinsics_json = load_from_json(intrinsics_path)
        camera_json = intrinsics_json["camera"]
        width = int(camera_json["w"])
        height = int(camera_json["h"])
        depth_scale = config.depth_unit_scale_factor
        if depth_scale is None:
            depth_scale = 1.0 / float(camera_json["scale"])

        fx = torch.full((num_frames,), float(camera_json["fx"]), dtype=torch.float32)
        fy = torch.full((num_frames,), float(camera_json["fy"]), dtype=torch.float32)
        cx = torch.full((num_frames,), float(camera_json["cx"]), dtype=torch.float32)
        cy = torch.full((num_frames,), float(camera_json["cy"]), dtype=torch.float32)

        poses = []
        valid_image_filenames = []
        valid_depth_filenames = []
        for image_path, depth_path, pose_line in zip(image_filenames, depth_filenames, pose_lines):
            pose = np.fromstring(pose_line, sep=" ", dtype=np.float32).reshape(4, 4)
            pose = torch.from_numpy(pose)
            if torch.isinf(pose).any() or torch.isnan(pose).any():
                continue
            pose[:3, 1] *= -1
            pose[:3, 2] *= -1
            poses.append(pose)
            valid_image_filenames.append(image_path)
            valid_depth_filenames.append(depth_path)

        if not poses:
            raise ValueError(f"No valid poses found in {traj_path}")

        num_images = len(valid_image_filenames)
        num_train_images = math.ceil(num_images * config.train_split_fraction)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(0, num_images - 1, num_train_images, dtype=int)
        i_eval = np.setdiff1d(i_all, i_train)
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = torch.from_numpy(np.stack(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method="none",
            center_method=config.center_method,
        )

        scale_factor = 1.0
        if config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= config.scale_factor
        poses[:, :3, 3] *= scale_factor

        image_filenames = [valid_image_filenames[i] for i in indices]
        depth_filenames = [valid_depth_filenames[i] for i in indices]
        poses = poses[indices.tolist()]
        fx = fx[indices.tolist()]
        fy = fy[indices.tolist()]
        cx = cx[indices.tolist()]
        cy = cy[indices.tolist()]

        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-config.scene_scale, -config.scene_scale, -config.scene_scale],
                    [config.scene_scale, config.scene_scale, config.scene_scale],
                ],
                dtype=torch.float32,
            )
        )

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {
            "depth_filenames": depth_filenames,
            "depth_unit_scale_factor": depth_scale,
        }

        if config.load_3D_points:
            point_cloud_data = self._load_3D_points(
                ply_file_path=scene_dir.parent / f"{scene_dir.name}_mesh.ply",
                transform_matrix=transform_matrix,
                scale_factor=scale_factor,
                points_color=config.point_cloud_color,
            )
            if point_cloud_data:
                metadata.update(point_cloud_data)

        return DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata=metadata,
        )

    def _load_3D_points(
        self, ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float, points_color: bool
    ):
        import open3d as o3d

        config = cast(ReplicaDataParserConfig, self.config)

        point_cloud = o3d.io.read_point_cloud(str(ply_file_path))
        if len(point_cloud.points) == 0:
            return {}

        points3d = np.asarray(point_cloud.points, dtype=np.float32)
        colors = np.asarray(point_cloud.colors)

        max_points = config.max_3d_points
        if max_points is not None and len(points3d) > max_points:
            indices = np.linspace(0, len(points3d) - 1, max_points, dtype=np.int64)
            points3d = points3d[indices]
            if len(colors) > 0:
                colors = colors[indices]

        points3d = torch.from_numpy(points3d)
        points3d = torch.cat((points3d, torch.ones_like(points3d[..., :1])), dim=-1) @ transform_matrix.T
        points3d *= scale_factor

        output = {"points3D_xyz": points3d}
        if points_color and len(colors) > 0:
            output["points3D_rgb"] = torch.from_numpy((colors * 255).astype(np.uint8))
        else:
            output["points3D_rgb"] = torch.zeros((points3d.shape[0], 3), dtype=torch.uint8)
        return output
