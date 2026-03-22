import torch 
import torch.nn as nn
import numpy as np
import zlib
import os, random, cv2, math
from torch.utils.data import  DataLoader
from tqdm import tqdm
# import pycolmap 
from utils.points import umeyama_alignment
from ggpt.dataloader import points_utils


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, 
            chunk_sample='random', # or octree or tile
            chunk_size=0.2, max_ff_pts_perchunk=400000, clip_method='shrink',
            min_geo_pts_perchunk=100, min_ff_pts_perchunk=10000, 
            max_num_chunks_after_converge = 100, pca_transform=False, overlap_ratio=0.02,
        ):
        """
        A Base Dataset consists of multiple scenes.
        It patchifies each scene into multiple overlapping sub-scenes (chunks) during data loading.
        """
        super().__init__()
        self.chunk_sample = chunk_sample
        self.chunk_size, self.max_ff_pts_perchunk, self.clip_method = chunk_size, max_ff_pts_perchunk, clip_method
        self.min_geo_pts_perchunk, self.min_ff_pts_perchunk = min_geo_pts_perchunk,min_ff_pts_perchunk
        self.max_num_chunks_after_converge = max_num_chunks_after_converge
        self.pca_transform = pca_transform
        self.overlap_ratio = overlap_ratio
        self.disable_gt_input_filters = _env_flag("GGPT_DISABLE_GT_INPUT_FILTERS", False)
        if self.disable_gt_input_filters:
            print(
                "[GGPT] GGPT_DISABLE_GT_INPUT_FILTERS=1: "
                "disabling GT-based input filtering (tandt ROI and 4ddress mask)."
            )
    
    def load_scene_(self, idx):
        scene = self.load_scene(idx)
        """
        !! IMPORTANT: don't forget to do the alignment!
        Align geo_pts to ff_pts 
        Align gt_pts to geo_pts (if exists)
        """
        #scene['geo_pts'] = umeyama_alignment(B=scene['ff_pts'],A=scene['geo_pts'], mask=scene['geo_msks'])[0]
        #scene['geo_pts'][~scene['geo_msks']] = 0
        scene['ff_pts_original'] = scene['ff_pts'].clone()
        scene['ff_pts'] = umeyama_alignment(B=scene['geo_pts'], A=scene['ff_pts'], mask=scene['geo_msks'])[0]
        if self.mode in ['train', 'val']:
            assert 'gt_pts' in scene
            scene['gt_pts_metric'] = scene['gt_pts'].clone()
            scene['gt_pts'] = umeyama_alignment(B=scene['geo_pts'], A=scene['gt_pts'], mask=scene['gt_msks']& scene['geo_msks'])[0]
            scene['gt_pts'][~scene['gt_msks']] = 0

            if self.disable_gt_input_filters is False:
                if 'tandt' in scene['dataset_name']: #HARD-CODED. Here Otherwise, scene scale for TnT outdoor scenes are too large.
                    roi_min, roi_max = self.get_bbox(scene)
                    scene['geo_msks'] &= ( (scene['geo_pts']>=roi_min) & (scene['geo_pts']<=roi_max) ).all(dim=-1)
                    scene['ff_msks'] = ( (scene['ff_pts']>=roi_min) & (scene['ff_pts']<=roi_max) ).all(dim=-1)
                if '4ddress' in scene['dataset_name']: #Use human segmentation to filter out non-human points. TODO: Replace the mask with SAM
                    scene['geo_msks'] &= scene['gt_msks']
                    scene['ff_msks'] = scene['gt_msks']
        #scene['radius'] = torch.std(scene['geo_pts'][scene['geo_msks']], dim=0).mean().item()*3 #a scalar
        #geo_pts might have outlier we use ff_pts to compute the radius
        if 'ff_msks' in scene:
            scene['radius'] = torch.std(scene['ff_pts'][scene['ff_msks']], dim=0).mean().item()*3 #a scalar
        else:
            scene['radius'] = torch.std(scene['ff_pts'], dim=0).mean().item()*3 #a scalar

        return scene
    
    def get_crop_mask(self, points, mins, maxs, mask):
        #mins (,3) max(,3)
        in_msk = (points>=mins)& (points<=maxs) # (...,3)
        in_msk = in_msk.all(dim=-1) # (...)
        if mask is not None:
            in_msk = in_msk & mask
        return points[in_msk], in_msk

    def sample_a_chunk(self, scene, tosample_geo_msk, min_geo_pts_perchunk, min_ff_pts_perchunk, generator):
        chunk_radius = self.chunk_size * scene['radius'] # a scalar
        center_pt = scene['geo_pts'][tosample_geo_msk].view(-1,3)[generator.integers(0, tosample_geo_msk.sum().item())]
        while True:
            _, ff_msk_chunk = self.get_crop_mask(scene['ff_pts'], center_pt - chunk_radius, center_pt + chunk_radius, None)
            if 'ff_msks' in scene:
                ff_msk_chunk = ff_msk_chunk & scene['ff_msks']
            _, geo_msk_chunk = self.get_crop_mask(scene['geo_pts'], center_pt - chunk_radius, center_pt + chunk_radius, tosample_geo_msk)
            ff_outofchunk_msk = (~ff_msk_chunk) & geo_msk_chunk
            #TODO in some experiments, it is better to directly replace the out-of-chunk ff points with geo points
            msk_chunk = ff_outofchunk_msk | ff_msk_chunk
            if msk_chunk.sum()<=self.max_ff_pts_perchunk:
                break
            elif self.clip_method == 'shrink': # Shrink the chunk size to recrop
                chunk_radius = chunk_radius * 0.9
            elif self.clip_method == 'subsample': # Subsample points to reduce the number
                selected_idx = torch.where(msk_chunk)[0][generator.choice(msk_chunk.sum().item(), self.max_ff_pts_perchunk, replace=False)]
                msk_chunk_ = torch.zeros_like(msk_chunk.view(-1), dtype=torch.bool)
                msk_chunk_[selected_idx] = True
                msk_chunk = msk_chunk_.view_as(msk_chunk)
                #TODO in some experiments, it is better to preserve all geo points in the chunk 
            else:
                raise NotImplementedError
        a_chunk = {'msks_in_scene': msk_chunk, 'chunk_center': center_pt, 'chunk_radius': chunk_radius,
                    'ff_pts': scene['ff_pts'][msk_chunk], 'ff_pts_conf': scene['ff_conf'][msk_chunk],
                    'geo_pts': scene['geo_pts'][msk_chunk], 'geo_msks': scene['geo_msks'][msk_chunk],
                    'rgbs': scene['images'][msk_chunk],
                }
        if self.mode in ['train', 'val']:
            a_chunk['gt_pts'] = scene['gt_pts'][msk_chunk]
            a_chunk['gt_msks'] = scene['gt_msks'][msk_chunk]
        return a_chunk
    
    def normalize_a_chunk(self, a_chunk):
        for key in ['ff_pts', 'geo_pts', 'gt_pts']:
            if key in a_chunk:
                a_chunk[key] = (a_chunk[key] - a_chunk['chunk_center']) / a_chunk['chunk_radius']
                # TODO scale the value may achieve better performance 
        return a_chunk
    
    def normalize_chunks(self, scene_chunks):
        normalized_chunks = []
        for a_chunk in scene_chunks:
            a_chunk = self.normalize_a_chunk(a_chunk)
            normalized_chunks.append(a_chunk)
        return normalized_chunks

    def unnormalize_pts(self, chunk, pts):
        return pts * chunk['chunk_radius'] + chunk['chunk_center']

    def get_bbox(self, scene):
        # Use GT bbox (region of interest, this can be replaced by a user defined box)
        roi_radius = torch.std(scene['gt_pts'][scene['gt_msks']], dim=0).mean().item()*3
        roi_min = scene['gt_pts'][scene['gt_msks']].min(dim=0).values - roi_radius*0.1
        roi_max = scene['gt_pts'][scene['gt_msks']].max(dim=0).values + roi_radius*0.1
        return roi_min, roi_max

    def split_scenes_random(self, scene):
        scene_chunks = []
        tosample_geo_msk = scene['geo_msks'].clone()
        # Use a process-stable scene seed; Python's built-in hash is salted per process.
        generator = np.random.default_rng(zlib.crc32(scene['scene_name'].encode('utf-8')) % (2**32))
        min_geo_pts_perchunk, min_ff_pts_perchunk = self.min_geo_pts_perchunk, self.min_ff_pts_perchunk
            
        fail_count = 0
        while tosample_geo_msk.sum() > 0:
            a_chunk = self.sample_a_chunk(scene, tosample_geo_msk, min_geo_pts_perchunk, min_ff_pts_perchunk, generator)
            num_geo_pts_inchunk = a_chunk['geo_msks'].sum().item()
            num_ff_pts_inchunk = a_chunk['ff_pts'].shape[0]
            if num_geo_pts_inchunk < min_geo_pts_perchunk:
                min_geo_pts_perchunk = max(num_geo_pts_inchunk//2, self.min_geo_pts_perchunk//10)
                fail_count += 1
            elif num_ff_pts_inchunk < min_ff_pts_perchunk:
                min_ff_pts_perchunk = max(num_ff_pts_inchunk//2, self.min_ff_pts_perchunk//10)
                fail_count += 1
            else:
                fail_count = 0
                inc_geo_pts_num = (tosample_geo_msk & a_chunk['msks_in_scene']).sum().item()
                if len(scene_chunks) >= self.max_num_chunks_after_converge and inc_geo_pts_num==0:
                    break
                tosample_geo_msk = tosample_geo_msk & (~a_chunk['msks_in_scene'])
                scene_chunks.append(a_chunk)
            if fail_count >= 10:
                break
        return scene_chunks

    def split_scenes_octree(self, scene):
        scene_chunks = []
        #TODO Pre-crop the scene 
        #TODO speed up octree. Downsampling points
        leaves = points_utils.chunk_by_octree(scene['ff_pts'].reshape(-1,3), MAX=self.max_ff_pts_perchunk)
        for leaf in leaves:
            half_size = leaf.half_size*(1+self.overlap_ratio)
            chunk_mins = leaf.center - half_size
            chunk_maxs = leaf.center + half_size
            chunk_pts, msk_chunk = self.get_crop_mask(scene['ff_pts'], chunk_mins, chunk_maxs, None)
            #TODO add geo_pts in the chunk, whose ff_pts fall outside the chunk
            num_ff_pts = msk_chunk.sum().item()
            num_geo_pts = scene['geo_msks'][msk_chunk].sum().item()
            if num_ff_pts ==0:
                continue 
            '''
            #This cannot improve the performance.
            if num_geo_pts < self.min_geo_pts_perchunk:
                # Expand the chunk to include enough geo pts
                search_xyzs = scene['geo_pts'][scene['geo_msks']& (~msk_chunk)] # pts outside the current chunk
                while (num_geo_pts < self.min_geo_pts_perchunk ): #and num_ff_pts < self.max_ff_pts_perchunk*10):
                    geo_dists = torch.norm(search_xyzs - leaf.center.view(1,3), dim=-1)
                    geo_knn = torch.topk(geo_dists, k=int(self.min_geo_pts_perchunk/10), largest=False)
                    geo_knn_indices = geo_knn.indices
                    # compute the new chunk_mins, chunk_maxs
                    expand_mins = torch.min(search_xyzs[geo_knn_indices], dim=0).values
                    expand_maxs = torch.max(search_xyzs[geo_knn_indices], dim=0).values
                    chunk_mins = torch.min(chunk_mins, expand_mins)
                    chunk_maxs = torch.max(chunk_maxs, expand_maxs)
                    _, msk_chunk_ff = self.get_crop_mask(scene['ff_pts'], chunk_mins, chunk_maxs, None)
                    _, msk_chunk_geo = self.get_crop_mask(scene['geo_pts'], chunk_mins, chunk_maxs, scene['geo_msks'])
                    msk_chunk = msk_chunk_ff | msk_chunk_geo
                    num_ff_pts = msk_chunk.sum().item()
                    num_geo_pts = scene['geo_msks'][msk_chunk].sum().item()
                    search_xyzs = scene['geo_pts'][scene['geo_msks']& (~msk_chunk)] # update pts outside the current chunk
                    leaf.center = (chunk_mins + chunk_maxs)/2.0
                    half_size = (chunk_maxs - chunk_mins).max()/2.0
                    assert num_geo_pts >= int(self.min_geo_pts_perchunk/10)
            '''
   
            chunk = {'msks_in_scene': msk_chunk, 'chunk_center': leaf.center, 'chunk_radius': half_size,
                    'ff_pts': scene['ff_pts'][msk_chunk], 'ff_pts_conf': scene['ff_conf'][msk_chunk],
                    'geo_pts': scene['geo_pts'][msk_chunk], 'geo_msks': scene['geo_msks'][msk_chunk],
                    'rgbs': scene['images'][msk_chunk],
                }
            if self.mode in ['train', 'val']:
                chunk['gt_pts'] = scene['gt_pts'][msk_chunk]
                chunk['gt_msks'] = scene['gt_msks'][msk_chunk]
            scene_chunks.append(chunk)
        return scene_chunks

    def pca_transform_scene(self, scene):
        _, eigvecs, mean = points_utils.pca_transform(scene['ff_pts'])
        for key in ['ff_pts', 'geo_pts', 'gt_pts', 'gt_pts_metric']:
            if key in scene:
                scene[key] = points_utils.pca_transform(scene[key], eigvecs, mean)[0]
                #TODO handle extrinsics
        return scene

    def __getitem__(self, idx):
        scene = self.load_scene_(idx)
        if self.pca_transform:
            scene = self.pca_transform_scene(scene)
        if self.mode == 'train':
            raise NotImplementedError
        else:
            if self.chunk_sample == 'random':
                scene_chunks = self.split_scenes_random(scene)
            elif self.chunk_sample == 'octree':
                scene_chunks = self.split_scenes_octree(scene)
            elif self.chunk_sample == 'tiling':
                raise NotImplementedError
            else:
                raise NotImplementedError    
            scene_chunks = self.normalize_chunks(scene_chunks)
            return scene_chunks, scene
        
    

