from .base_dataset import BaseDataset
import os
import json

import cv2
import numpy as np

from utils.geometry import closed_form_inverse_se3


class ScanNetPPRawDataset(BaseDataset):
    """
    Reads the raw ScanNet++ layout described in the GGPT README (section 4.1):

        <root>/<scene_id>/dslr/
            nerfstudio/transforms_undistorted.json
            resized_undistorted_images/*.JPG
            render_depth/*.png

    Used with `sample_extracted=False` and a `sampled_file` produced by view sampling
    (section 4.2, data/sample_views.py): each entry in that JSON is `[scene_name, [img_name, ...]]`,
    and the base class (sfm/dataloader/base_dataset.py) takes care of turning that list into
    the actual per-sequence batches consumed by sfm/run_benchmark_sfm.py.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pose_cache = {}

    def read_scene_pose(self, scene_name):
        if scene_name in self._pose_cache:
            return self._pose_cache[scene_name]

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
        self._pose_cache[scene_name] = imgname2pose
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
