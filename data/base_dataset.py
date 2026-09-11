import numpy as np
from PIL import Image, ImageFile

from torch.utils.data import Dataset
from .dataset_util import *

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    """
    Base dataset class for GGPT's own raw-data readers (e.g. view sampling).

    Adapted from vggt/training/data/base_dataset.py (Meta's VGGT training code), with one
    addition: datasets that set `self.get_all_imgs = True` are indexed by a plain integer
    (the sequence/scene index) and must implement `get_all_images(seq_index)` instead of
    `get_data(...)`. This is used by view-sampling scripts that need every view of a scene
    at once to compute pairwise visibility statistics, rather than a fixed-size training tuple.

    Attributes:
        img_size: Target image size (typically the width)
        patch_size: Size of patches for vit
        augs.scales: Scale range for data augmentation [min, max]
        rescale: Whether to rescale images
        rescale_aug: Whether to apply augmentation during rescaling
        landscape_check: Whether to handle landscape vs portrait orientation
    """
    def __init__(
        self,
        common_conf,
    ):
        super().__init__()
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.aug_scale = common_conf.augs.scales
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        self.landscape_check = common_conf.landscape_check
        self.get_all_imgs = False

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        """
        Args:
            idx_N: a plain sequence index if `self.get_all_imgs`, else a tuple
                (seq_index, img_per_seq, aspect_ratio)
        """
        if self.get_all_imgs:
            return self.get_all_images(idx_N)
        seq_index, img_per_seq, aspect_ratio = idx_N
        return self.get_data(
            seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio
        )

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0):
        """
        Abstract method to retrieve a fixed-size training tuple for a given sequence.
        Must be implemented by subclasses that don't use `get_all_imgs`.
        """
        raise NotImplementedError(
            "This is an abstract method and should be implemented in the subclass, i.e., each dataset should implement its own get_data method."
        )

    def get_all_images(self, seq_index):
        """
        Abstract method to retrieve every image of a given sequence/scene.
        Must be implemented by subclasses that set `self.get_all_imgs = True`.
        """
        raise NotImplementedError(
            "This is an abstract method and should be implemented in the subclass when get_all_imgs=True."
        )

    def get_target_shape(self, aspect_ratio):
        """
        Calculate the target shape based on the given aspect ratio.

        Args:
            aspect_ratio: Target aspect ratio

        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size

        # ensure the input shape is friendly to vision transformer
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size

        image_shape = np.array([short_size, self.img_size])
        return image_shape

    def process_one_image(
        self,
        image,
        depth_map,
        extri_opencv,
        intri_opencv,
        original_size,
        target_image_shape,
        track=None,
        filepath=None,
        safe_bound=4,
    ):
        """
        Process a single image and its associated data: resize, crop, and convert
        coordinates to world/camera 3D points.

        Returns:
            tuple: (image, depth_map, extri_opencv, intri_opencv, world_coords_points,
                    cam_coords_points, point_mask, track)
        """
        # Make copies to avoid in-place operations affecting original data
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        # Apply random scale augmentation during training if enabled
        if self.training and self.aug_scale:
            random_h_scale, random_w_scale = np.random.uniform(
                self.aug_scale[0], self.aug_scale[1], 2
            )
            # Avoid random padding by capping at 1.0
            random_h_scale = min(random_h_scale, 1.0)
            random_w_scale = min(random_w_scale, 1.0)
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            aug_size = original_size

        # Move principal point to the image center and crop if necessary
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath,
        )

        original_size = np.array(image.shape[:2])  # update original_size
        target_shape = target_image_shape

        # Handle landscape vs. portrait orientation
        rotate_to_portrait = False
        if self.landscape_check:
            # Switch between landscape and portrait if necessary
            if original_size[0] > 1.25 * original_size[1]:
                if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                    target_shape = np.array([target_image_shape[1], target_image_shape[0]])
                    rotate_to_portrait = True

        # Resize images and update intrinsics
        if self.rescale:
            image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                image, depth_map, intri_opencv, target_shape, original_size, track=track,
                safe_bound=safe_bound,
                rescale_aug=self.rescale_aug
            )
        else:
            print("Not rescaling the images")

        # Ensure final crop to target shape
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath, strict=True,
        )

        # Apply 90-degree rotation if needed
        if rotate_to_portrait:
            assert self.landscape_check
            clockwise = np.random.rand() > 0.5
            image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                clockwise=clockwise,
                track=track,
            )

        # Convert depth to world and camera coordinates
        world_coords_points, cam_coords_points, point_mask = (
            depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
        )

        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
        )
