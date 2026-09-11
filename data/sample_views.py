import torch
import sys
import os
import hydra
from tqdm import tqdm
import torch.distributed as dist

# Make this repo's top-level packages (`data.*`, `utils.*`) importable regardless of the
# cwd this script is launched from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)

from data.composed_dataset import TupleConcatDataset
from utils.geometry import unproject_depth_map_to_point_map_torch, homo, closed_form_inverse_se3
import cv2
import numpy as np
from time import time
import math
from hydra.utils import instantiate


def init_DDP(cfg=None):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f'cuda:{local_rank}'
    print(f"Start running basic DDP example on rank {rank}, device {device}")
    return rank, local_rank, device


def downsample_depths(depth_nhw, downsample=4):
    depth = depth_nhw.unsqueeze(1)
    kernel_size = 3
    assert downsample in [1,2,4,8]
    if downsample ==1 :
        return depth_nhw
    niter = int(math.log(downsample,2))
    for i in range(niter):
        depth = torch.nn.functional.max_pool2d(depth, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)
    return depth.squeeze(1)


def fill_holes(depth_nhw):
    #x (N,H,W,3)
    kernel_size = 3
    initial_depth = depth_nhw.unsqueeze(1)
    initial_mask = (initial_depth!=0)
    depth = initial_depth.clone()
    while (depth!=0).float().mean()<0.9:
        depth = torch.nn.functional.max_pool2d(depth, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2) #Remove zeros
        depth[initial_mask] = initial_depth[initial_mask]
    depth = depth.squeeze(1)
    return depth


@hydra.main(version_base=None, config_path="../configs/train",config_name="sample-views")
def main(cfg):
    rank, local_rank, device = init_DDP(cfg)

    selected_scenes = None#['nrgbd_val/complete_kitchen-img248.png']

    base_dataset_list = []
    # Instantiate each base dataset with common configuration
    for baseset_dict in cfg.valdataset_configs:
        baseset = instantiate(baseset_dict, common_conf=cfg.common_config)
        base_dataset_list.append(baseset)

    # Use custom concatenation class that supports tuple indexing
    dataset = TupleConcatDataset(base_dataset_list, cfg.common_config)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=cfg.common_config.num_workers,
        sampler=sampler
    )

    setup_dataset = {}
    global_step = 0
    total_sequences = 0
    max_seq_num = cfg.sample_configs.get('max_seq_num', None)
    stop = False
    for data_unchunked in tqdm(dataloader, total=len(dataloader)):
        if stop:
            break
        global_step += 1
        #move everything to GPU
        data_unchunked = {k:v.to(device) if torch.is_tensor(v) else v for k,v in data_unchunked.items()}
        dataset_name, scene_name = data_unchunked['seq_name'].split('-')[0], '-'.join(data_unchunked['seq_name'].split('-')[1:])
        if selected_scenes is not None and not f'{dataset_name}/{scene_name}' in selected_scenes:
            continue
        N0 = data_unchunked['images'].shape[0]
        # split N into chunks of cfg.sample_configs.max_image
        num_chunk = int(math.ceil(N0/cfg.sample_configs.max_image))
        for ci in range(num_chunk):
            end = min((ci+1)*cfg.sample_configs.max_image, N0)
            start = max(0, end-cfg.sample_configs.max_image)
            #print(f'Processing {data_unchunked["seq_name"]} #Images{N0}: {start}-{end}...')
            data = {k:v[start:end] if torch.is_tensor(v) or type(v)==list else v for k,v in data_unchunked.items()}
            N = end-start
            downsample = 1
            depths = downsample_depths(data['depths'], downsample=downsample) #NHW
            depths = fill_holes(depths) #NHW

            images = torch.nn.functional.interpolate(data['images'], size=(depths.shape[1], depths.shape[2]), mode='bilinear', align_corners=False) #N3HW

            intrinsics_scaled = data['intrinsics'].clone()
            intrinsics_scaled[:,:2] = intrinsics_scaled[:,:2]/downsample
            points = unproject_depth_map_to_point_map_torch(depths, data['extrinsics'], intrinsics_scaled) #NHW3
            proj_mat = torch.einsum('nij,njk->nik', intrinsics_scaled, data['extrinsics'][:,:3]) #B34
            cam_points = torch.einsum('nij,mhwj->nmhwi', proj_mat, homo(points))
            reproj_depths = cam_points[...,2] #n-tgtm-srchw
            reproj_xy = cam_points[...,:2]/cam_points[...,2:]
            depth_masked_for_min = depths.masked_fill(depths==0, float('inf'))
            depth_min = depth_masked_for_min.view(N,-1).min(axis=1).values
            depth_masked_for_max = depths.masked_fill(depths==0, -1*float('inf'))
            depth_max = depth_masked_for_max.view(N,-1).max(axis=1).values
            h, w = images.shape[-2:]
            depth_min = depth_min[:,None,None,None]
            depth_max = depth_max[:,None,None,None]

            in_frustum = (reproj_depths>depth_min)&(reproj_depths<depth_max)&(reproj_xy[...,0]>0)&(reproj_xy[...,0]<w)&(reproj_xy[...,1]>0)&(reproj_xy[...,1]<h)
            in_frustum_ratio = in_frustum.sum(dim=(2,3))/(h*w)
            in_frustum_bidirect = torch.stack([in_frustum_ratio, in_frustum_ratio.T], axis=-1).min(-1).values
            reproj_xy_normed = reproj_xy.clone() #nmhw, n:tgt, m:src
            reproj_xy_normed[...,0] = (reproj_xy_normed[...,0]/(w-1)-0.5)*2
            reproj_xy_normed[...,1] = (reproj_xy_normed[...,1]/(h-1)-0.5)*2
            sampled_depth = torch.nn.functional.grid_sample(
                depths.unsqueeze(1), #N1HW
                reproj_xy_normed.view(N,N*h,w,2),
                mode='nearest', align_corners=True) #(N,1,N*h,w)
            sampled_depth = sampled_depth.squeeze(1).view(N,N,h,w) #tgt, src, h,w
            depth_diff = (sampled_depth-reproj_depths).abs()

            not_occluded = depth_diff < 0.05*reproj_depths
            V = not_occluded & in_frustum & (sampled_depth>0) #N,N,H,W
            valid_pixel_ratio1 = V.sum(dim=(2,3))/(h*w)
            valid_pixel_ratio2 = V.transpose(0,1).sum(dim=(2,3))/(h*w)
            #IoU = (valid_pixel_ratio1+valid_pixel_ratio2)/2 #N,N
            IoU = torch.stack([valid_pixel_ratio1,valid_pixel_ratio2], axis=0).min(axis=0).values
            '''
            V[tgti,srci,h,w] indicates whether the pixel (h,w) in the srci view is visible in the tgti view or not
            valid_pixel_ratio[tgti,srci] indicates the percentage of pixels in the srci view that are visible in the tgti view
            V = V.permute(1,0,2,3) #src,tgt,H,W
            When building the graph, we may need to consider both directions
            '''

            #Triangulation angles
            #TODO more efficient. Only compute the triangulation angle of the center of the visibliity region
            c2w = closed_form_inverse_se3(data['extrinsics']) #N44
            cam_center = c2w[:,:3,3] #N3
            rays_src = points - cam_center[:,None,None,:] #NHW3
            rays_src = rays_src.unsqueeze(0).repeat(N,1,1,1,1) #1N-HW3-1N-HW3=NNHW3
            rays_tgt = points[None,:,:,:,:]-cam_center[:,None,None,None,:] #1NHW3-N1113=N-tgtN-srcHW3
            rays_src, rays_tgt = torch.nn.functional.normalize(rays_src, dim=-1), torch.nn.functional.normalize(rays_tgt, dim=-1)
            cos_angle = (rays_src*rays_tgt).sum(dim=-1) #
            angle = torch.acos(cos_angle.clamp(-1+1e-6, 1-1e-6)) #NNHW (tgt, src, h, w)
            #convert it into degree
            Theta= angle/np.pi*180
            average_angles = (Theta*V).sum(dim=(2,3)) / V.sum(dim=(2,3)) #NN


            #start_mast3r = time()
            #mast3r_scores = mast3r_retrieval_scores(images, retriever)
            #print(f'  Time for mast3r retrieval: {time()-start_mast3r:.1f}s')

            if cfg.save_vis:
                output_dir = os.path.join(cfg.output_dir, f'statistics/{dataset_name}_{scene_name}-{start}-{end}')
                os.makedirs(output_dir, exist_ok=True)
                img_show = (images.permute(2,0,3,1).reshape(h,N*w,3)*255).cpu().numpy().astype(np.uint8)
                # cv2.imwrite(os.path.join(output_dir, f'all-views.png'), cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
                for i in range(N):
                    #i is the tgt view
                    valid_mask_i = V[i] #N,H,W
                    img_show = []
                    for j in range(N):
                        img_j = (images[j].permute(1,2,0).clone()*255).cpu()
                        if i==j:
                            img_j[:5,:] = torch.tensor([255.,0,0])
                            img_j[-5:,:] = torch.tensor([255.,0,0])
                        if i!=j:
                            img_j[valid_mask_i[j]] = torch.tensor([0,255.,0])
                        img_j = img_j.numpy().astype(np.uint8)
                        #add some text to show the IoU
                        angle_text = 'nan' if average_angles[i,j].isnan() else f'{int(average_angles[i,j])}'
                        cv2.putText(img_j, f'{int(IoU[i,j]*100)}%, {angle_text}', (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
                        img_show.append(img_j)
                    img_show = np.stack(img_show, axis=0) #N,H,W,3
                    num_row = 10
                    num_col = int(math.ceil(N/num_row))
                    img_show_grid = np.ones((num_col*h, num_row*w, 3), dtype=np.uint8)*255
                    for k in range(N):
                        row_id = k//num_row
                        col_id = k%num_row
                        img_show_grid[row_id*h:(row_id+1)*h, col_id*w:(col_id+1)*w] = img_show[k]
                    cv2.imwrite(os.path.join(output_dir, f'tgt-view{i:02d}.png'), cv2.cvtColor(img_show_grid, cv2.COLOR_RGB2BGR))


            def sample_sequence(num_sequences, N_range, V0, IoU0, Theta, alpha_range, r_range, seed_thresh=0.7):
                L, device = V0.shape[0], V0.device
                V = V0.clone()
                V[torch.arange(L),torch.arange(L)] = 0
                IoU = IoU0.clone()
                IoU[torch.arange(L),torch.arange(L)] = 0
                h, w = V.shape[-2:]
                results = []
                visible_area = V.max(dim=0).values.sum(dim=(1,2))/(h*w) #L.
                seed_thresh = min(seed_thresh, visible_area.quantile(0.8))
                candidate_seedviews = (visible_area>seed_thresh).nonzero().squeeze(1)
                for seed_i in candidate_seedviews:
                    Msub = torch.zeros([L,1,1,1], dtype=torch.bool).to(device)
                    # 1. sample seed view
                    #seed_i = candidate_seedviews[torch.randint(candidate_views.shape[0],[])]
                    selected_view = []
                    Msub[seed_i] = True
                    selected_view.append(seed_i.item())
                    Vis = (Msub*V).max(0).values #L,H,W
                    theta_min = (Msub*V*Theta + (~(Msub*V))*360).min(0).values #Lsrc,H,W
                    average_theta_min = (theta_min*Vis).sum(dim=(1,2))/Vis.sum(dim=(1,2))
                    IoU_current = (IoU*Msub[:,:,0,0]).max(axis=0).values #Ltgt, Lsrc -> L src
                    IoU_current[Msub[:,0,0,0]] = 0
                    # Exclude views that are already selected as IoU[i][i]=0
                    FLAG = True
                    N_planned = torch.randint(N_range[0], N_range[1]+1, []).item()
                    for i in range(1,N_planned):
                        candidate_views = ((IoU_current>r_range[0])&(IoU_current<r_range[1])&(average_theta_min>alpha_range[0])&(average_theta_min<alpha_range[1])).nonzero().squeeze(1)
                        if len(candidate_views) == 0:
                            break
                        next_i = candidate_views[torch.randint(candidate_views.shape[0],[])]
                        Msub[next_i] = True
                        Vis = (Msub*V).max(0).values #L,H,W
                        theta_min = (Msub*V*Theta + (~(Msub*V))*360).min(0).values
                        average_theta_min = (theta_min*Vis).sum(dim=(1,2))/Vis.sum(dim=(1,2))
                        IoU_current = (IoU*Msub[:,:,0,0]).max(axis=0).values #Ltgt, Lsrc -> L src
                        IoU_current[Msub[:,0,0,0]] = 0# Exclude views that are already selected as IoU[i][i]=0
                        selected_view.append(next_i.item())
                    if Msub.sum()>=N_range[0]:
                        results.append(selected_view)
                    if len(results) == num_sequences:
                        break
                return results


            for N_r_alpha in cfg.sample_configs.N_r_alpha:
                N_range, r_range, alpha_range = N_r_alpha.N_range, N_r_alpha.r_range, N_r_alpha.alpha_range
                results = sample_sequence(cfg.sample_configs.num_sequences_per_scene, N_range, V, IoU, Theta, alpha_range, r_range)
                setup_key = f'num{N_range[0]}-{N_range[1]}_r{r_range[0]}-{r_range[1]}_alpha{alpha_range[0]}-{alpha_range[1]}'
                #print(f'Setup key {setup_key} Sample {len(results)} sequences')
                if setup_key not in setup_dataset:
                    setup_dataset[setup_key] = {}
                if dataset_name not in setup_dataset[setup_key]:
                    setup_dataset[setup_key][dataset_name] = []
                new_entries = [(scene_name, [data['image_names'][ii] for ii in res]) for res in results]
                setup_dataset[setup_key][dataset_name].extend(new_entries)
                total_sequences += len(new_entries)
                if max_seq_num is not None and total_sequences > max_seq_num:
                    overflow = total_sequences - max_seq_num
                    setup_dataset[setup_key][dataset_name] = setup_dataset[setup_key][dataset_name][:-overflow]
                    total_sequences = max_seq_num
                if cfg.save_vis and global_step:
                    vis_dir = os.path.join(cfg.output_dir, 'vis', setup_key, dataset_name)
                    os.makedirs(vis_dir, exist_ok=True)
                    for res in results:
                        images_selected = images[res].permute(2,0,3,1).reshape(h,-1,3) #N,C,H,W->H,N,W,C
                        images_selected = (images_selected*255).cpu().numpy().astype(np.uint8)
                        vis_name = os.path.join(vis_dir, f'{scene_name}_{"-".join([str(ri) for ri in res])}.png')
                        cv2.imwrite(vis_name, images_selected[...,[2,1,0]])

            if max_seq_num is not None and total_sequences >= max_seq_num:
                stop = True
                break

    for setup_key in setup_dataset:
        for dataset_name in setup_dataset[setup_key]:
            save_file = os.path.join(cfg.output_dir, 'views/per_rank', f'{dataset_name}_sample{setup_key}.rank{rank}.json')
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            import json
            print(setup_key, dataset_name, len(setup_dataset[setup_key][dataset_name]))
            with open(save_file,'w') as f:
                json.dump(setup_dataset[setup_key][dataset_name], f, indent=4)
    torch.distributed.barrier()
    if rank==0:
        #merge all the jsonfiles
        for setup_key in setup_dataset:
            for dataset_name in setup_dataset[setup_key]:
                save_file = os.path.join(cfg.output_dir, 'views', f'{dataset_name}_sample{setup_key}.json')
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                import json
                all_data = []
                for r in range(int(os.environ['WORLD_SIZE'])):
                    rank_file = os.path.join(cfg.output_dir, 'views/per_rank', f'{dataset_name}_sample{setup_key}.rank{r}.json')
                    if os.path.exists(rank_file):
                        with open(rank_file,'r') as f:
                            data = json.load(f)
                        all_data.extend(data)
                print(f'Final save {setup_key} {dataset_name} #sequences: {len(all_data)} to {save_file}')
                with open(save_file,'w') as f:
                    json.dump(all_data, f, indent=4)


if __name__ == "__main__":
    main()
