import os
import json

import cv2
import numpy as np
import torch

from data.dataset_util import *
from data.base_dataset import BaseDataset
from utils.geometry import closed_form_inverse_se3


class ScanNetPPDataset(BaseDataset):
    """
    Reads the raw ScanNet++ layout described in the GGPT README (section 4.1):

        <root>/<scene_id>/dslr/
            nerfstudio/transforms_undistorted.json
            resized_undistorted_images/*.JPG
            render_depth/*.png

    Only the "get all images of a scene" access pattern is implemented (`get_all_imgs=True`),
    which is what view-sampling (data/sample_views.py) needs: it wants every view of a
    scene at once to compute pairwise visibility/overlap statistics, not a fixed-size
    training tuple.
    """

    def __init__(
        self,
        common_conf,
        root: str,
        split: str = "train",
        name: str = "scannetpp",
        splits_root: str = None,
        step: int = 1,
        vis_downsample: int = 4,
    ):
        super().__init__(common_conf)
        assert split in ("train", "val"), f"Invalid split {split}, should be 'train' or 'val'."
        self.root = root
        self.split = split
        self.name = name
        self.step = step
        self.vis_downsample = vis_downsample
        self.training = split == "train"
        self.get_all_imgs = True

        # By default the ScanNet++ toolkit ships split files alongside <root>/../splits/.
        splits_root = splits_root or os.path.join(self.root, "..", "splits")
        split_file = os.path.join(splits_root, f"nvs_sem_{self.split}.txt")

        if os.path.isfile(split_file):
            scene_names = [line.strip() for line in open(split_file, "r").readlines() if line.strip()]
        else:
            # Fall back to whatever scenes are actually present under root (e.g. a small
            # local subset downloaded for testing without the official split files).
            scene_names = sorted(os.listdir(self.root))

        self.scene_names = [
            sn for sn in scene_names
            if os.path.isdir(os.path.join(self.root, sn, "dslr", "render_depth"))
        ]
        assert len(self.scene_names) > 0, (
            f"No ScanNet++ scenes found under {self.root} for split '{self.split}' "
            f"(checked split file {split_file})."
        )

        self.scene2image = {}
        for scene in self.scene_names:
            # Index by the rendered-depth images: some frames are skipped upstream (is_bad=True),
            # so this is the authoritative set of usable frames. Depths are '.png', RGB is '.JPG'.
            image_list = sorted(os.listdir(os.path.join(self.root, scene, "dslr", "render_depth")))
            image_list = [im[:-4] for im in image_list]  # strip '.png'
            self.scene2image[scene] = image_list

    def __len__(self):
        return len(self.scene_names)

    def read_scene_pose(self, scene_name):
        transforms = json.load(
            open(os.path.join(self.root, scene_name, "dslr", "nerfstudio", "transforms_undistorted.json"), "r")
        )
        fx, fy, cx, cy = transforms["fl_x"], transforms["fl_y"], transforms["cx"], transforms["cy"]
        cx, cy = cx - 0.5, cy - 0.5  # convert to OpenCV pixel-center convention
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        height, width = transforms["h"], transforms["w"]

        imgname2pose = {}
        for frame in transforms["frames"]:
            imgname = frame["file_path"][:-4]  # strip '.JPG'
            opengl_c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            opencv_c2w = opengl_c2w.copy()
            opencv_c2w[:3, 1:3] *= -1
            opencv_c2w = opencv_c2w[[1, 0, 2, 3], :]
            opencv_c2w[2, :] *= -1
            w2c = closed_form_inverse_se3(opencv_c2w)
            imgname2pose[imgname] = {
                "c2w": opencv_c2w, "w2c": w2c, "K": K,
                "width": width, "height": height, "intr_convention": "opencv",
            }
        return imgname2pose

    def read_img_rgb(self, scene_name, img_name):
        path = os.path.join(self.root, scene_name, "dslr", "resized_undistorted_images", f"{img_name}.JPG")
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image {img_name} not found in {scene_name} ({path})")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def read_img_depth(self, scene_name, img_name):
        path = os.path.join(self.root, scene_name, "dslr", "render_depth", f"{img_name}.png")
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth image {img_name} not found in {scene_name} ({path})")
        valid_mask = (depth > 0.05 * 1000) & (depth < 20.0 * 1000)
        depth = depth.astype(np.float32) / 1000.0
        depth[~valid_mask] = 0
        return depth

    def get_all_images(self, scene_idx):
        scene_name = self.scene_names[scene_idx]
        scene_pose = self.read_scene_pose(scene_name)

        # Downsample target resolution: this is only used to compute visibility/overlap
        # statistics between views, so we keep it small to bound the O(N^2 * H * W) cost
        # of the pairwise reprojection done downstream in sample_views.py.
        target_width = int(self.img_size / self.vis_downsample)

        views = []
        for i in range(0, len(self.scene2image[scene_name]), self.step):
            img_name = self.scene2image[scene_name][i]
            image = self.read_img_rgb(scene_name, img_name)
            depth = self.read_img_depth(scene_name, img_name)
            w2c = scene_pose[img_name]["w2c"]
            K = scene_pose[img_name]["K"]
            height, width = image.shape[:2]

            target_height = int(round(target_width / width * height))
            target_shape = np.array([target_height, target_width])
            center_pp = (K[0, 2] == width / 2 - 0.5) and (K[1, 2] == height / 2 - 0.5)

            if height != target_shape[0] or width != target_shape[1] or not center_pp:
                processed_image, processed_depth, extri, processed_intri, _, _, _, _ = self.process_one_image(
                    image=image, depth_map=depth, extri_opencv=w2c, intri_opencv=K,
                    original_size=np.array([height, width]),
                    target_image_shape=target_shape,
                    filepath=f"{self.name}-{scene_name}-{img_name}",
                )
            else:
                processed_image, processed_depth, extri, processed_intri = image, depth, w2c, K

            if processed_depth.max() < 1e-5:  # empty frame
                continue
            views.append({
                "image_name": img_name,
                "image": processed_image,
                "depth": processed_depth,
                "extrinsics": extri,
                "intrinsics": processed_intri,
            })

        return {
            "seq_name": f"{self.name}-{scene_name}",
            "images": torch.from_numpy(np.stack([v["image"] for v in views], 0)).permute(0, 3, 1, 2).float() / 255.0,
            "depths": torch.from_numpy(np.stack([v["depth"] for v in views], 0)).float(),
            "extrinsics": torch.from_numpy(np.stack([v["extrinsics"] for v in views], 0)).float(),  # w2c
            "intrinsics": torch.from_numpy(np.stack([v["intrinsics"] for v in views], 0)).float(),
            "image_names": [v["image_name"] for v in views],
        }
