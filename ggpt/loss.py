import torch 
# # from pytorch3d.loss import chamfer_distance
# import pytorch3d.ops as ops
from collections import defaultdict
def laplacian_loss(pred, conf, gt, msk, beta=1, gamma=0.2, reduce='mean'):
    """msk=False everywhere => 0 loss (grad-connected); conf.log uses clamp to avoid -inf."""
    if not msk.any():
        z = (pred * 0).sum()
        if conf is None:
            return None, z
        return z, z
    error = torch.norm(pred - gt, dim=-1) #(N,)
    if conf is None:
        if reduce == 'mean':
            return None, error[msk].mean()
        elif reduce == 'sum':
            return None, error[msk].sum()
        elif reduce == 'none':
            return None, error[msk]
        else:
            raise ValueError(f"Invalid reduce method: {reduce}")
    conf_log = torch.log(conf.clamp(min=1e-8))
    if reduce == 'mean':
        loss = (error * beta * conf - gamma * conf_log)[msk].mean()
        return loss, error[msk].mean()
    elif reduce == 'sum':
        loss = (error * beta * conf - gamma * conf_log)[msk].sum()
        return loss, error[msk].sum()
    elif reduce == 'none':
        loss = (error * beta * conf - gamma * conf_log)[msk]
        return loss, error[msk]
    else:
        raise ValueError(f"Invalid reduce method: {reduce}")

def chamfer_loss(pred, gt, conf, gamma=0.2):
    error, error_normals = chamfer_distance(pred.unsqueeze(0), gt.unsqueeze(0), point_reduction=None, batch_reduction=None) #(1,N_pred), (1,N_gt)
    error = (error[0]+error[1])/2
    error = error.squeeze(0) #(N_pred,)
    loss = (error * conf - 0.2 * torch.log(conf.clamp(min=1e-8))).mean()
    return loss, error.mean()


def _disambiguate_vector_directions(pcl, knns, vecs: torch.Tensor) -> torch.Tensor:
    """
    Disambiguates normal directions according to [1].

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """
    # parse out K from the shape of knns
    K = knns.shape[2]
    # the difference between the mean of each neighborhood and
    # each element of the neighborhood
    df = knns - pcl[:, :, None]
    # projection of the difference on the principal direction
    proj = (vecs[:, :, None] * df).sum(3)
    # check how many projections are positive
    n_pos = (proj > 0).type_as(knns).sum(2, keepdim=True)
    # flip the principal directions where number of positive correlations
    flip = (n_pos < (0.5 * K)).type_as(knns)
    vecs = (1.0 - 2.0 * flip) * vecs
    return vecs

def normal_consistency_loss_packed(pred, gt, msk, conf, offset,k):
    loss, error = 0, 0
    valib_b = 0
    for b in range(len(offset)):
        start_idx = 0 if b==0 else offset[b-1]
        end_idx = offset[b]
        pred_b = pred[start_idx:end_idx]
        gt_b = gt[start_idx:end_idx]
        msk_b = msk[start_idx:end_idx]
        conf_b = conf[start_idx:end_idx]
        if msk_b.sum()<k*1.5:
            continue
        loss_b, error_b = normal_consistency_loss(pred_b, gt_b, msk_b, conf_b, k)
        loss += loss_b
        error += error_b
        valib_b += 1
    if valib_b>0:
        loss /= valib_b
        error /= valib_b
    else:
        loss = torch.tensor(0.0, device=pred.device)
        error = torch.tensor(0.0, device=pred.device)
    return loss, error

def normal_consistency_loss(pred, gt, msk, conf,k):
    pred_normals = ops.estimate_pointcloud_normals(pred.unsqueeze(0), neighborhood_size=k)[0] #(N,3)
    # We need to msk GT points before estimating normals to avoid noisy normals
    gt_normals_masked = ops.estimate_pointcloud_normals(gt[msk].unsqueeze(0), neighborhood_size=k)[0] #(N,3)
    cos = torch.abs((pred_normals[msk] * gt_normals_masked).sum(dim=-1)) #(N,)
    if conf is None:
        return None, (1 - cos).mean()
    loss = ((1 - cos) * conf[msk]).mean()
    return loss, (1 - cos).mean()


def local_geometry_loss_packed(pred, gt, msk, init, conf, offset, k):
    loss_dict, error_dict = {'nc':0, 'lap':0, 'cov':0, 'curv':0, 'curv_id':0}, {'nc':0, 'lap':0, 'cov':0, 'curv':0,'curv_id':0}
    valib_b = 0
    for b in range(len(offset)):
        start_idx = 0 if b==0 else offset[b-1]
        end_idx = offset[b]
        pred_b = pred[start_idx:end_idx]
        gt_b = gt[start_idx:end_idx]
        init_b = init[start_idx:end_idx]
        msk_b = msk[start_idx:end_idx]
        conf_b = conf[start_idx:end_idx] if conf is not None else None
        if msk_b.sum()<k*1.5:
            continue
        #Note that different from the previously implemented normal consistenct loss, 
        # We apply mask to both pred and gt to compute local geometry
        init_dic = compute_local_geometry(init_b[msk_b], k)
        pred_dic = compute_local_geometry(pred_b[msk_b], k)
        gt_dic = compute_local_geometry(gt_b[msk_b], k)
        nc_error = 1 - (pred_dic['normals'] * gt_dic['normals']).sum(dim=-1).abs() #(N,)
        lap_error = torch.norm(pred_dic['laplacian'] - gt_dic['laplacian'], dim=-1) #(N,)
        cov_error = torch.norm(pred_dic['cov'] - gt_dic['cov'], dim=(1,2)) #(N,3,3) -> (N,)
        cur_error = torch.norm(pred_dic['curvatures'] - gt_dic['curvatures'], dim=-1) #(N,)
        error_dict['nc'] += nc_error.mean()
        error_dict['lap'] += lap_error.mean()
        error_dict['cov'] += cov_error.mean()
        error_dict['curv'] += cur_error.mean()
        curv_id_error = torch.norm(pred_dic['curvatures'] - init_dic['curvatures'], dim=-1) #(N,) #(N,)
        error_dict['curv_id'] += curv_id_error.mean()
        if conf_b is not None:
            loss_dict['nc'] += (nc_error * conf_b[msk_b]).mean()
            loss_dict['lap'] += (lap_error * conf_b[msk_b]).mean()
            loss_dict['cov'] += (cov_error * conf_b[msk_b]).mean()
            loss_dict['curv'] += (cur_error * conf_b[msk_b]).mean()
            loss_dict['curv_id'] += (curv_id_error * conf_b[msk_b]).mean()
        else:
            loss_dict['nc'] += nc_error.mean()
            loss_dict['lap'] += lap_error.mean()
            loss_dict['cov'] += cov_error.mean()
            loss_dict['curv'] += cur_error.mean()
            loss_dict['curv_id'] += curv_id_error.mean()
        valib_b += 1

    if valib_b>0:
        for key in loss_dict.keys():
            loss_dict[key] /= valib_b
            error_dict[key] /= valib_b
    #Add some prefix
    return loss_dict, error_dict


def compute_local_geometry(pts, k):
    pts_centered = pts - pts.mean(dim=0) #(N,3)
    pts_centered = pts_centered.unsqueeze(0)
    cov, knns = ops.get_point_covariances(pts_centered, torch.tensor([pts_centered.shape[1]]).to(pts.device), neighborhood_size=k) #(1,N,3,3), (1,N,k)    use_symeig_workaround, disambiguate_directions = True, True
    from pytorch3d.common.workaround import symeig3x3
    curvatures, local_coord_frames = symeig3x3(cov, eigenvectors=True)
    # disambiguate normal
    normals = _disambiguate_vector_directions(
        pts_centered, knns, local_coord_frames[:, :, :, 0]
    )
    laplacian = pts_centered-knns.mean(dim=-2) #(1,N,3)
    return {'cov':cov[0], 'knns':knns[0], 'normals':normals[0],'laplacian':laplacian[0], 'curvatures':curvatures[0]}



class LossComputer(object):
    def __init__(self, 
        ff_lap_weight,  dlt_lap_weight, id_weight,
        laplacian_beta=1, laplacian_gamma=0.2,
        dlt_part_weight=1.0, non_dlt_part_weight=1.0,
        nc_weight=None, local_geo_weights = {}, local_geo_use_conf=True,
        chamfer_weight=0.0, chamfer_max_num=10000,
        ):
        self.laplacian_beta = laplacian_beta
        self.laplacian_gamma = laplacian_gamma
        self.ff_lap_weight = ff_lap_weight
        self.dlt_lap_weight = dlt_lap_weight
        self.id_weight = id_weight
        self.dlt_part_weight = dlt_part_weight
        self.non_dlt_part_weight = non_dlt_part_weight
        self.chamfer_weight = chamfer_weight
        self.chamfer_max_num = chamfer_max_num
        self.nc_weight = nc_weight  
        self.local_geo_weights = local_geo_weights
        self.local_geo_use_conf = local_geo_use_conf
    def __call__(self, out):
        loss_dict = {}
        '''
        loss_dict['loss/ff_lap_loss'], loss_dict['metric/ff_error'] = laplacian_loss(pred=out['ff_pts_out'], conf=out['ff_pts_conf_out'], gt=out['gt_pts'], msk=out['gt_msks'], beta=self.laplacian_beta, gamma=self.laplacian_gamma)
        loss_dict['loss/dlt_lap_loss'], loss_dict['metric/dlt_error'] = laplacian_loss(pred=out['dlt_pts_out'], conf=out['dlt_pts_conf_out'], gt=out['gt_pts'], msk=out['gt_msks']&out['dlt_msks'], beta=self.laplacian_beta, gamma=self.laplacian_gamma)
        '''
        loss_dict['loss/ff_lap_loss(dlt-part)'], loss_dict['metric/ff_error(dlt-part)'] = laplacian_loss(pred=out['ff_pts_out'], conf=out['ff_pts_conf_out'], gt=out['gt_pts'], msk=out['gt_msks']&out['dlt_msks'], beta=self.laplacian_beta, gamma=self.laplacian_gamma, reduce='none')
        loss_dict['loss/ff_lap_loss(nondlt-part)'], loss_dict['metric/ff_error(nondlt-part)'] = laplacian_loss(pred=out['ff_pts_out'], conf=out['ff_pts_conf_out'], gt=out['gt_pts'], msk=out['gt_msks']&(~out['dlt_msks']), beta=self.laplacian_beta, gamma=self.laplacian_gamma, reduce='none')

        if out['gt_msks'].sum()>0:
            loss_dict['loss/ff_lap_loss'] = loss_dict['loss/ff_lap_loss(dlt-part)'].sum()*self.dlt_part_weight + loss_dict['loss/ff_lap_loss(nondlt-part)'].sum()*self.non_dlt_part_weight
            loss_dict['loss/ff_lap_loss'] /= out['gt_msks'].sum()
            loss_dict['metric/ff_error'] = (loss_dict['metric/ff_error(dlt-part)']).sum()*self.dlt_part_weight+(loss_dict['metric/ff_error(nondlt-part)']).sum()*self.non_dlt_part_weight #unweighted
            loss_dict['metric/ff_error'] /= out['gt_msks'].sum()
            loss_dict['metric/ff_error_(dlt-part)'] = loss_dict['metric/ff_error(dlt-part)'].mean()
            loss_dict['metric/ff_error_(nondlt-part)'] = loss_dict['metric/ff_error(nondlt-part)'].mean()
        else:
            loss_dict['loss/ff_lap_loss'] = torch.tensor(0.0, device=out['ff_pts_out'].device)
            loss_dict['metric/ff_error'] = torch.tensor(0.0, device=out['ff_pts_out'].device)
            loss_dict['metric/ff_error_(dlt-part)'] = torch.tensor(0.0, device=out['ff_pts_out'].device)
            loss_dict['metric/ff_error_(nondlt-part)'] = torch.tensor(0.0, device=out['ff_pts_out'].device)

        loss_dict['loss/dlt_lap_loss'], loss_dict['metric/dlt_error'] = laplacian_loss(pred=out['dlt_pts_out'], conf=out['dlt_pts_conf_out'], gt=out['gt_pts'], msk=out['gt_msks']&out['dlt_msks'], beta=self.laplacian_beta, gamma=self.laplacian_gamma, reduce='mean')
        #loss_dict['metric/ff_error_(dlt-part)'] = laplacian_loss(pred=out['ff_pts_out'], conf=None, gt=out['gt_pts'], msk=out['gt_msks']&out['dlt_msks'])[1]
        #loss_dict['metric/ff_error_(nondlt-part)'] = laplacian_loss(pred=out['ff_pts_out'], conf=None, gt=out['gt_pts'], msk=out['gt_msks']&(~out['dlt_msks']))[1]
        dlt_m = out['dlt_msks']
        if dlt_m.any():
            loss_dict['loss/id_loss'] = (out['ff_pts_out'] - out['dlt_pts_out'])[dlt_m].norm(dim=-1).mean()
        else:
            loss_dict['loss/id_loss'] = (out['ff_pts_out'] * 0).sum()
        
        loss_dict['loss/objective'] = self.ff_lap_weight*loss_dict['loss/ff_lap_loss'] + \
            self.dlt_lap_weight*loss_dict['loss/dlt_lap_loss'] + \
            self.id_weight*loss_dict['loss/id_loss']
        
        if self.chamfer_weight>0: # We only compute chamfer distance when its loss is > 0 as it is expensive
            loss_dict['loss/chamfer_lap_loss'], loss_dict['metric/chamfer_error'] = 0, 0
            valid_b = 0
            for b in range(len(out['offset'])):
                start_idx = 0 if b==0 else out['offset'][b-1]
                end_idx = out['offset'][b]
                #We sample only chamfer_max_num points to compute chamfer distance
                gt_msks = out['gt_msks'][start_idx:end_idx]
                if gt_msks.sum() > self.chamfer_max_num:
                    perm = torch.randperm(gt_msks.sum())[:self.chamfer_max_num]
                    gt_msks_idx = gt_msks.nonzero(as_tuple=False).squeeze(1)[perm]
                    gt_msks = torch.zeros_like(gt_msks, dtype=torch.bool)
                    gt_msks[gt_msks_idx] = True
                pred = out['ff_pts_out'][start_idx:end_idx][gt_msks]
                gt = out['gt_pts'][start_idx:end_idx][gt_msks]
                conf = out['ff_pts_conf_out'][start_idx:end_idx][gt_msks]
                if gt.shape[0]==0: #No gt points in this chunk
                    continue
                loss_b, error_b = chamfer_loss(pred, gt, conf)
                loss_dict['loss/chamfer_lap_loss'] += loss_b
                loss_dict['metric/chamfer_error'] += error_b
                valid_b += 1
            if valid_b>0:
                loss_dict['loss/chamfer_lap_loss'] /= valid_b
                loss_dict['metric/chamfer_error'] /= valid_b
            else:
                loss_dict['loss/chamfer_lap_loss'] = torch.tensor(0.0, device=out['ff_pts_out'].device)
                loss_dict['metric/chamfer_error'] = torch.tensor(0.0, device=out['ff_pts_out'].device)
            loss_dict['loss/objective'] += self.chamfer_weight*loss_dict['loss/chamfer_lap_loss']
        if self.nc_weight is not None:
            loss_dict['loss/nc_loss'], loss_dict['metric/nc_error'] = normal_consistency_loss_packed(
                pred=out['ff_pts_out'], gt=out['gt_pts'], msk=out['gt_msks'], conf=out['ff_pts_conf_out'], offset=out['offset'])
            loss_dict['loss/objective'] += self.nc_weight*loss_dict['loss/nc_loss']
        if self.local_geo_weights is not None and self.local_geo_weights != {}:
            loss_dict_local, error_dict_local = local_geometry_loss_packed(
                pred=out['ff_pts_out'], gt=out['gt_pts'], init=out['ff_pts_in'],
                msk=out['gt_msks'], conf=out['ff_pts_conf_out'] if self.local_geo_use_conf else None,
                offset=out['offset'], k=40)
            for key in self.local_geo_weights.keys():
                loss_dict[f'loss/local_geo_{key}_loss'] = loss_dict_local[key]
                loss_dict[f'metric/local_geo_{key}_error'] = error_dict_local[key]
                loss_dict['loss/objective'] += self.local_geo_weights[key]*loss_dict[f'loss/local_geo_{key}_loss']
        return loss_dict
