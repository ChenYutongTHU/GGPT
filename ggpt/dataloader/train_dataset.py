import torch
import torch.distributed as dist
import numpy as np
import os
from torch.utils.data import IterableDataset, get_worker_info

from ggpt.dataloader.base_dataset import BaseDataset
from glob import glob


class OctreeExhaustiveIterableTrain(IterableDataset):
    """
    No upfront pass over all scenes. Scenes are assigned round-robin to global slots
    (DDP rank × DataLoader worker): scene index s goes to slot (s % (world * num_workers)).
    Each worker loads only its scenes, splits octree lazily, yields all full batches (drop_last).
    Different workers may yield different numbers of steps (scene count and chunks/scene vary).

    At the start of each iterator pass (logical epoch), scene visit order and within-scene batch
    order are shuffled. Call ``set_epoch`` on this object before ``iter(dataloader)`` so workers
    pick up the same epoch when the DataLoader respawns them (persistent_workers=False).
    """

    def __init__(self, train_dataset: "TrainDataset"):
        super().__init__()
        self.d = train_dataset
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        d = self.d
        wi = get_worker_info()
        num_workers = wi.num_workers if wi is not None else 1
        worker_id = wi.id if wi is not None else 0
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        gw_count = max(1, world * num_workers)
        gw_self = rank * num_workers + worker_id

        bs = max(1, int(d.train_octree_chunks_per_batch))
        n_scenes = len(d.scene_list)
        base_seed = int(getattr(d, "_octree_shuffle_seed", 42))
        rng = np.random.default_rng(
            base_seed + self._epoch * 1_000_003 + gw_self * 17_791
        )
        scene_order = rng.permutation(n_scenes)

        for s in scene_order:
            if s % gw_count != gw_self:
                continue
            # Physical scene index s only (one _predata.bin / folder per scene). Not the old
            # virtual idx = s * num_data_per_scene used for repeated samples per scene.
            chunks, scene = d._load_octree_exhaust_chunks(s)
            nb = len(chunks) // bs
            if nb <= 0:
                continue
            batch_order = rng.permutation(nb)
            for batch_i in batch_order:
                start = int(batch_i) * bs
                batch_chunks = chunks[start : start + bs]
                if len(batch_chunks) < bs:
                    raise RuntimeError(
                        f"octree iterable scene={scene.get('scene_name')}: "
                        f"expected {bs} chunks at offset {start}, got {len(batch_chunks)}"
                    )
                if bs == 1:
                    yield batch_chunks[0], scene
                else:
                    yield batch_chunks, scene

class TrainDataset(BaseDataset):
    def __init__(self, data_dict,
        num_data_per_scene=None, seed=42,
        downsample_stride=1,
        **kwargs):
        super().__init__(**kwargs)
        self.mode = 'train'
        self.data_dict = data_dict
        self.scene_list = []
        self._octree_shuffle_seed = int(seed)
        self.generator = np.random.default_rng(seed)
        self.downsample_stride = downsample_stride
        for dataset_name, folders in self.data_dict.items():
            for path in glob(folders):
                if os.path.isfile(path) and path.endswith('_predata.bin'):
                    # New format: single _predata.bin file per scene
                    scene_name = os.path.basename(path).replace('_predata.bin', '')
                    self.scene_list.append({'dataset_name': dataset_name,
                                            'scene_name': scene_name,
                                            'predata_file': path})
                elif os.path.isdir(path):
                    # Legacy format: foldter containing ff_outputs.bin, sfm_dlt_outputs.bin, gt.bin
                    scene_name = os.path.basename(path.rstrip('/'))
                    if os.path.isfile(os.path.join(path, 'ff_outputs.bin')) \
                        and os.path.isfile(os.path.join(path, 'sfm_dlt_outputs.bin')) \
                        and os.path.isfile(os.path.join(path, 'gt.bin')):
                        self.scene_list.append({'dataset_name': dataset_name,
                                                'scene_name': scene_name,
                                                'folder': path})
                    else:
                        print(f"Warning: missing files for folder {scene_name} in dataset {dataset_name}, skipped.")
        print(f"TrainDataset: found {len(self.scene_list)} scenes.")
        if num_data_per_scene is None:
            # One file per physical scene; virtual repeats only used for non-octree map training.
            num_data_per_scene = 1 if self.chunk_sample == "octree" else 100
        self.num_data_per_scene = int(num_data_per_scene)
        if self.chunk_sample == "octree" and self.octree_exhaustive:
            self.sparse_idx_cycle_length = len(self.scene_list)
        else:
            self.sparse_idx_cycle_length = len(self.scene_list) * self.num_data_per_scene

    def _load_octree_exhaust_chunks(self, scene_idx: int):
        scene = self.load_scene_(scene_idx)
        if self.pca_transform:
            scene = self.pca_transform_scene(scene)
        chunks = self.normalize_chunks(self.split_scenes_octree(scene))
        return chunks, scene

    def _next_idx_after_sparse_alignment(self, idx):
        if self.chunk_sample == "octree" and self.octree_exhaustive:
            return idx + 1
        return idx + self.num_data_per_scene

    def __len__(self):
        if self.chunk_sample == "octree":
            raise TypeError(
                "TrainDataset length is undefined for chunk_sample='octree' (lazy IterableDataset; "
                "no global batch count without scanning all scenes)."
            )
        return len(self.scene_list) * self.num_data_per_scene

    def load_scene(self, idx):
        idx = int(idx)
        n = len(self.scene_list)
        if self.chunk_sample == "octree" and self.octree_exhaustive:
            scene_idx = idx % n
        else:
            scene_idx = (idx // self.num_data_per_scene) % n
        entry = self.scene_list[scene_idx]
        if 'predata_file' in entry:
            try:
                data = torch.load(entry['predata_file'], map_location='cpu')
            except Exception as e:
                step = 1 if self.chunk_sample == "octree" and self.octree_exhaustive else self.num_data_per_scene
                return self.load_scene(idx + step)
            scene = {
                'dataset_name': entry['dataset_name'],
                'scene_name': entry['scene_name'],
                'ff_pts': data['ff_pts'].float(),              # (N,H,W,3)
                'ff_conf': data['ff_pts_conf'].float(),        # (N,H,W)
                'ff_extrinsics': None,
                'ff_intrinsics': None,
                'geo_pts': data['dlt_pts'].float(),            # (N,H,W,3)
                'geo_msks': data['dlt_msks'],                  # (N,H,W)
                'geo_extrinsics': data['dlt_extrinsics'].float() if 'dlt_extrinsics' in data else None,
                'geo_intrinsics': data['dlt_intrinsics'].float() if 'dlt_intrinsics' in data else None,
                'gt_pts': data['gt_pts'].float(),              # (N,H,W,3)
                'gt_msks': data['gt_msks'],                    # (N,H,W)
                'gt_extrinsics': data['gt_extrinsics'].float() if 'gt_extrinsics' in data else None,
                'gt_intrinsics': data['gt_intrinsics'].float() if 'gt_intrinsics' in data else None,
                'images': data['images'].permute(0, 2, 3, 1).float(),  # (N,C,H,W) -> (N,H,W,C)
            }
        else:
            folder = entry['folder']
            ff_data = torch.load(os.path.join(folder, 'ff_outputs.bin'), map_location='cpu')
            geo_data = torch.load(os.path.join(folder, 'sfm_dlt_outputs.bin'), map_location='cpu')
            gt_data = torch.load(os.path.join(folder, 'gt.bin'), map_location='cpu')
            scene = {
                'dataset_name': entry['dataset_name'],
                'scene_name': entry['scene_name'],
                'ff_pts': ff_data['points'].float(),           # (N,H,W,3)
                'ff_conf': ff_data['points_conf'].float(),     # (N,H,W)
                'ff_extrinsics': ff_data['extrinsics'].float(),
                'ff_intrinsics': ff_data['intrinsics'].float(),
                'geo_pts': geo_data['points'].float(),         # (N,H,W,3)
                'geo_msks': geo_data['point_masks'],           # (N,H,W)
                'geo_extrinsics': geo_data['extrinsics'].float() if 'extrinsics' in geo_data else None,
                'geo_intrinsics': geo_data['intrinsics'].float() if 'intrinsics' in geo_data else None,
                'gt_pts': gt_data['points'].float(),
                'gt_msks': gt_data['point_masks'],
                'gt_extrinsics': gt_data['extrinsics'].float() if 'extrinsics' in gt_data else None,
                'gt_intrinsics': gt_data['intrinsics'].float() if 'intrinsics' in gt_data else None,
                'images': ff_data['images_ff'].float(),        # (N,H,W,3)
            }
        if self.downsample_stride > 1:
            scene['images'] = scene['images'][::self.downsample_stride, ::self.downsample_stride, :]
            scene['ff_pts'] = scene['ff_pts'][::self.downsample_stride, ::self.downsample_stride, :]
            scene['ff_conf'] = scene['ff_conf'][::self.downsample_stride, ::self.downsample_stride]
            scene['geo_pts'] = scene['geo_pts'][::self.downsample_stride, ::self.downsample_stride, :]
            scene['geo_msks'] = scene['geo_msks'][::self.downsample_stride, ::self.downsample_stride]
            scene['gt_pts'] = scene['gt_pts'][::self.downsample_stride, ::self.downsample_stride, :]
            scene['gt_msks'] = scene['gt_msks'][::self.downsample_stride, ::self.downsample_stride]
        return scene


class TrainSubsetEvalDataset(TrainDataset):
    """
    Eval (mode=val) on a small subset of the same scenes/files as TrainDataset
    (_predata.bin or legacy folder layout). Indices are 0..len-1 per scene in the subset.
    """

    def __init__(self, max_scenes: int = 24, subset_shuffle_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = "val"
        self.octree_exhaustive = False
        self.scene_list.sort(key=lambda x: (x["dataset_name"], x["scene_name"]))
        if subset_shuffle_seed is not None:
            rng = np.random.default_rng(int(subset_shuffle_seed))
            order = rng.permutation(len(self.scene_list))
            self.scene_list = [self.scene_list[i] for i in order]
        self.scene_list = self.scene_list[: int(max_scenes)]

    def __len__(self):
        return len(self.scene_list)

    def load_scene(self, idx):
        return super().load_scene(int(idx) * self.num_data_per_scene)
