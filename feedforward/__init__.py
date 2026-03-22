import torch
from utils.geometry import (
    assert_centered_principal_point,
    closed_form_inverse_se3,
    normalize_centered_intrinsics_to_opencv,
    unproject_depth_map_to_point_map_torch,
)
from PIL import Image
import numpy as np
import sys 
import cv2


def _resize_images_lanczos(images, output_height, output_width):
    images_np = images.detach().cpu().numpy()
    resized = []
    for image in images_np:
        resized.append(cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4))
    return torch.from_numpy(np.stack(resized, axis=0)).float()

def preprocess(images, output_width=518):
    # Used to reproduce the submitted results
    if isinstance(images, torch.Tensor)==False:
        # first convert to torch tensor 
        images = torch.from_numpy(np.stack(images, axis=0)).float()/255.0
    _, original_height, original_width, _ = images.shape
    if original_width==output_width and original_height%14==0:
        return images
    if original_width<original_height:
        images = images.permute(0,2,1,3)  #(N,H,W,C)
        original_width, original_height = original_height, original_width
    output_height = round(output_width*original_height/original_width/14)*14
    images_ff = _resize_images_lanczos(images, output_height, output_width)
    return images_ff


def _normalize_intrinsics_hw(intrinsics, height, width):
    return normalize_centered_intrinsics_to_opencv(intrinsics, height, width)


def _assert_intrinsics_hw(intrinsics, height, width, *, name):
    assert_centered_principal_point(
        intrinsics,
        int(height),
        int(width),
        tol_px=1e-3,
        name=name,
    )
    return intrinsics


class FeedForward_Model(torch.nn.Module):
    def __init__(self, configs):
        super(FeedForward_Model, self).__init__()
        self.configs = configs
        if 'vggt' in self.configs.model:
            if 'point' in self.configs.model:
                self.vggt_branch = 'point'
            else:
                self.vggt_branch = 'depth'
            sys.path.append('vggt') #Or replace with the path to your vggt folder
            from vggt.models.vggt import VGGT
            '''
            self.model = VGGT(enable_point=(self.vggt_branch=='point'), enable_track=False, enable_depth=(self.vggt_branch=='depth'), enable_camera=True)
            downloaded_ckpt = 'feedforward/checkpoints/vggt-b/model.pt'
            self.model.load_state_dict(torch.load(downloaded_ckpt), strict=False)
            '''
            self.model = VGGT.from_pretrained("facebook/VGGT-1B")
        elif self.configs.model == 'dav3':
            from depth_anything_3.api import DepthAnything3
            self.model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
        elif self.configs.model == 'pi3':
            from Pi3.pi3.models.pi3 import Pi3
            self.model = Pi3.from_pretrained("yyfz233/Pi3")
        elif self.configs.model == 'pi3x':
            from pi3.models.pi3x import Pi3X
            self.model = Pi3X()
            from safetensors.torch import load_file
            weight = load_file('/iopsstor/scratch/cscs/cyutong/code/Pi3X/ckpt/model.safetensors')
            self.model.load_state_dict(weight, strict=False)
        elif self.configs.model == 'ma':
            from mapanything.models import MapAnything
            self.model = MapAnything.from_pretrained("facebook/map-anything")
        else:
            raise NotImplementedError(f"Model {self.configs.model} not implemented in FeedForward_Model.")
        self.model.eval()
    



    def forward(self, images, preprocessed=False, gt_dict=None):
        output_dict = {}
        device = images.device if isinstance(images, torch.Tensor) else 'cuda'
        output_width = 504 if self.configs.model == 'dav3' else 518
        images_ff = preprocess(images, output_width).to(device) if not preprocessed else images
        output_dict['images_ff'] = images_ff
        gt_intrinsics = None
        if gt_dict is not None and gt_dict.get('intrinsics', None) is not None:
            gt_intrinsics = gt_dict['intrinsics'].clone()
            _assert_intrinsics_hw(
                gt_intrinsics,
                images_ff.shape[1],
                images_ff.shape[2],
                name="FeedForward_Model gt_intrinsics",
            )
        if 'vggt' in self.configs.model:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.autocast(device_type='cuda', dtype=dtype):
                raw_outputs = self.model(images_ff.permute(0,3,1,2))  #(N,3,H,W)
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            output_dict['extrinsics'], intrinsics_native = pose_encoding_to_extri_intri(raw_outputs['pose_enc'], images_ff.shape[1:3])
            output_dict['extrinsics'], intrinsics_native = output_dict['extrinsics'][0], intrinsics_native[0] #squeeze the batch=1 dimension
            output_dict['intrinsics'] = _normalize_intrinsics_hw(
                intrinsics_native,
                images_ff.shape[1],
                images_ff.shape[2],
            )
            if 'depth' in self.configs.model:
                B,N,H,W,D = raw_outputs['depth'].shape
                # Keep VGGT's native pixel convention when turning depth into 3D.
                output_dict['points'] = unproject_depth_map_to_point_map_torch(
                        depth_map=raw_outputs['depth'].view(B*N,H,W),
                        extrinsics_cam=output_dict['extrinsics'].view(B*N,3,4),
                        intrinsics_cam=intrinsics_native.view(B*N,3,3)).view(B,N,H,W,3)[0]
                output_dict['points_conf'] = raw_outputs['depth_conf'][0] #squeeze the batch=1 dimension
            else:
                output_dict['points'] = raw_outputs['world_points'][0]  #(N,H,W,3)
                output_dict['points_conf'] = raw_outputs['world_points_conf'][0]  #(N,H,W)
        elif self.configs.model == 'dav3':
            if self.configs.dav3.get('input_pose', False):
                input_intrinsics = gt_intrinsics.clone()
                input_intrinsics[...,1,2] = input_intrinsics[...,1,2] + 0.5
                input_intrinsics[...,0,2] = input_intrinsics[...,0,2] + 0.5
                input_extrinsics = torch.eye(4,device=device)[None,:,:].repeat(images_ff.shape[0],1,1).to(device)
                input_extrinsics[:,:3,:4] = gt_dict['extrinsics'].clone()
            else:
                input_intrinsics = None
                input_extrinsics = None
            # The installed DepthAnything3 API expects a Python list of per-view images
            # for its internal parallel preprocessor, not a stacked torch tensor.
            dav3_images = [
                np.clip(img.detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8, copy=False)
                for img in images_ff
            ]
            dav3_results = self.model.inference(
                image=dav3_images,
                intrinsics=input_intrinsics.detach().cpu().numpy() if input_intrinsics is not None else None,
                extrinsics=input_extrinsics.detach().cpu().numpy() if input_extrinsics is not None else None,
                process_res=int(images_ff.shape[2]),
            )

            H, W = images_ff.shape[1:3]
            def _dav3_get(x, key):
                if isinstance(x, dict):
                    return x[key]
                return getattr(x, key)

            def _dav3_get_optional(x, *keys):
                if isinstance(x, dict):
                    for key in keys:
                        if key in x and x[key] is not None:
                            return x[key]
                    return None
                for key in keys:
                    value = getattr(x, key, None)
                    if value is not None:
                        return value
                return None

            def _to_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().float()
                return torch.from_numpy(np.asarray(x)).float()

            extrinsics = _to_tensor(_dav3_get(dav3_results, 'extrinsics'))
            intrinsics = _to_tensor(_dav3_get(dav3_results, 'intrinsics')).clone()
            depths = _to_tensor(_dav3_get(dav3_results, 'depth'))
            depth_conf_raw = _dav3_get_optional(dav3_results, 'depth_conf', 'conf')
            if depth_conf_raw is None:
                raise AssertionError("DepthAnything3 returned no confidence map; GGPT's DA3 path requires per-pixel confidence.")
            depth_conf = _to_tensor(depth_conf_raw)

            if extrinsics.ndim != 3 or intrinsics.ndim != 3 or depths.ndim != 3 or depth_conf.ndim != 3:
                raise AssertionError(
                    "Unexpected DepthAnything3 output shapes: "
                    f"extrinsics={tuple(extrinsics.shape)} "
                    f"intrinsics={tuple(intrinsics.shape)} "
                    f"depth={tuple(depths.shape)} "
                    f"conf={tuple(depth_conf.shape)}"
                )

            output_dict['extrinsics'] = extrinsics.to(device)
            output_dict['intrinsics'] = intrinsics.to(device)
            #Convert to opencv convention (cx = W-1/2, cy=H/2)
            output_dict['intrinsics'][...,1,2] = output_dict['intrinsics'][...,1,2] - 0.5
            output_dict['intrinsics'][...,0,2] = output_dict['intrinsics'][...,0,2] - 0.5
            output_dict['points'] = unproject_depth_map_to_point_map_torch(
                depth_map=depths.to(device).view(-1,H,W),
                extrinsics_cam=output_dict['extrinsics'][...,:3,:].view(-1,3,4),
                intrinsics_cam=output_dict['intrinsics'].view(-1,3,3))
            output_dict['points_conf'] = depth_conf.to(device).view(-1,H,W)
        elif self.configs.model in ['pi3', 'pi3x']:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            if self.configs.model == 'pi3x':
                if self.configs.pi3x.get('input_intrinsics', False):
                    input_intrinsics = gt_intrinsics.clone()
                    input_intrinsics[...,1,2] = input_intrinsics[...,1,2] + 0.5
                    input_intrinsics[...,0,2] = input_intrinsics[...,0,2] + 0.5 #Not sure here. But I guess it follows colmap-convention?
                    input_intrinsics = input_intrinsics.unsqueeze(0)
                else:
                    input_intrinsics = None

                if self.configs.pi3x.get('input_extrinsics', False):
                    input_extrinsics = torch.eye(4,device=device)[None,:,:].repeat(images_ff.shape[0],1,1).to(device)
                    input_extrinsics[:,:3,:4] = closed_form_inverse_se3(gt_dict['extrinsics'].clone())[:,:3,:4] #opencv c2w
                    input_extrinsics = input_extrinsics.unsqueeze(0)
                else:
                    input_extrinsics = None
                #TODO accept partial depth
                with torch.no_grad():
                        with torch.amp.autocast('cuda', dtype=dtype):
                            res = self.model(
                                imgs=images_ff.permute(0,3,1,2).unsqueeze(0), #B, N, C, H, W [0,1]
                                intrinsics=input_intrinsics,
                                poses=input_extrinsics,
                            )
            
            else:
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=dtype):
                        res = self.model(images_ff.permute(0,3,1,2).unsqueeze(0))
            from MoGe.moge.utils.geometry_torch import recover_focal_shift
            hh, ww = res['local_points'].shape[-3:-1]
            aspect_ratio = ww / hh
            focal, shift = recover_focal_shift(points=res['local_points'], mask = torch.sigmoid(res['conf'][..., 0])>0.1)
            f = (focal/2*(1+aspect_ratio**2)**0.5)*hh
            intrinsics = torch.eye(3,device=device)[None,:,:].repeat(res['camera_poses'].shape[1],1,1)
            intrinsics[:,0,0] = f[0]
            intrinsics[:,1,1] = f[0] #squeeze the batch=1 dimension
            intrinsics[:,0,2] = (ww-1)/2 
            intrinsics[:,1,2] = (hh-1)/2
            bb, nn = res['camera_poses'].shape[:2]
            extrinsics = closed_form_inverse_se3(res['camera_poses'].view(-1,4,4)).view(*res['camera_poses'].shape)
            output_dict['extrinsics'] = extrinsics[0]
            output_dict['intrinsics'] = intrinsics
            output_dict['points'] =  res['points'][0]
            output_dict['points_conf'] = res['conf'][0,...,0].exp()+1     
        elif self.configs.model == 'ma':
            from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
            import torchvision.transforms as tvf
            img_norm = IMAGE_NORMALIZATION_DICT['dinov2']
            N, H, W, C = images_ff.shape
            MA_RESOLUTION_LIST = [ #(w,h)
                    (518, 518),  # 1:1
                    (518, 392),  # 4:3
                    (518, 336),  # 3:2
                    (518, 294),  # 16:9
                    (518, 252),  # 2:1
                    (518, 168),  # 3.2:1
                    (392, 518),  # 3:4
                    (336, 518),  # 2:3
                    (294, 518),  # 9:16
                    (252, 518),  # 1:2

            ]
            assert (W,H) in MA_RESOLUTION_LIST, f'mapanything only supports {MA_RESOLUTION_LIST} now, but got {(W,H)}'
            images_ff = tvf.functional.normalize(images_ff.permute(0,3,1,2), mean=img_norm.mean, std=img_norm.std).reshape(1,N,C,H,W)
            input_list = [dict(img=images[:,ii], true_shape=np.int32([[H,W]]), idx=ii, instance=str(ii), data_norm_type=['dinov2']) for ii in range(N)]
            #add multi-modality 
            ma_results = self.model.infer(
                train_mode=False,
                views=input_list,
                memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
                use_amp=True,                     # Use mixed precision inference (recommended)
                amp_dtype="bf16",                # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
                apply_mask=False,              # Apply masking to dense geometry outputs (We set to False!)
                mask_edges=False,       # Remove edge artifacts by using normals and depth (We set to False!)
                apply_confidence_mask=True,      # Filter low-confidence regions
                confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels (Useless?)
            )
            output_dict['intrinsics'] = []
            output_dict['extrinsics'] = []
            output_dict['points'] = []
            output_dict['points_conf'] = []
            for i, pred in enumerate(ma_results):
                cam2world = pred["camera_poses"] #1,4,4
                output_dict['extrinsics'].append(closed_form_inverse_se3(cam2world))
                output_dict['intrinsics'].append(pred["intrinsics"])
                output_dict['points'].append(pred["pts3d"]) #1,H,W,3
                output_dict['points_conf'].append(pred["conf"])
            output_dict = {k:torch.stack(v, axis=1).squeeze(0) for k,v in output_dict.items()}
        else:
            raise NotImplementedError(f"Model {self.configs.model} not implemented in FeedForward_Model.")

        if 'intrinsics' in output_dict:
            output_dict['intrinsics'] = _normalize_intrinsics_hw(
                output_dict['intrinsics'],
                images_ff.shape[1],
                images_ff.shape[2],
            )
        return output_dict
