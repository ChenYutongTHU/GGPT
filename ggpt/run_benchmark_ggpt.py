import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from tqdm import tqdm

from evaluation import eval_points
from ggpt.dataloader import build_val_dataloader
from utils.common import init_DDP, move_to_device, set_seed
from utils.basic import Print
from utils.logger import EvalLogger
from utils.points import aggregate_chunks
from utils.io import save_xyzrgb_to_ply




def _disable_eval_tqdm() -> bool:
    # In DDP/non-interactive batch logs, tqdm causes excessive log flushing/noise.
    return dist.is_available() and dist.is_initialized()


def evaluate_ggpt(model, val_dataloader, hydra_output_dir, step, vis=False, log_suffix=""):
    log_dir = os.path.join(hydra_output_dir, "eval_logs", f"step{step:07d}{log_suffix}")
    eval_start_time = time.time()
    Print(
        "[Eval][BEGIN] "
        f"step={step} "
        f"log_suffix='{log_suffix}' "
        f"num_scenes={len(val_dataloader)} "
        f"vis={vis} "
        f"log_dir={log_dir}"
    )
    logger = EvalLogger(log_dir)
    device = next(model.parameters()).device
    unnormalize_func = val_dataloader.dataset.unnormalize_pts
    
    for batch in tqdm(
        val_dataloader,
        total=len(val_dataloader),
        disable=_disable_eval_tqdm(),
    ):
        assert len(batch) == 1, "Eval dataset should have only one scene!"
        scene_chunks, scene = batch[0]
        chunks_batch = [[chunk] for chunk in scene_chunks]
        to_collect = {"ff_pts": [], "ff_pts_conf": []}
        t0 = time.time()
        for chunk_idx, chunk_batch in enumerate(chunks_batch):
            chunk_batch = move_to_device(chunk_batch, device)
            with torch.no_grad():
                out = model(chunk_batch)
            to_collect["ff_pts"].append(unnormalize_func(chunk_batch[0], out["ff_pts_out"]))
            to_collect["ff_pts_conf"].append(out["ff_pts_conf_out"])

        if len(scene_chunks) == 0:
            print(f"{scene['dataset_name']}-{scene['scene_name']}: No valid chunks found!")
            pred_pts = scene["ff_pts"].clone().to(device)
        else:
            ff_pts_all = torch.cat(to_collect["ff_pts"], dim=0)
            ff_pts_conf_all = torch.cat(to_collect["ff_pts_conf"], dim=0)
            msks_in_scene = torch.stack(
                [chunk["msks_in_scene"] for chunk in scene_chunks], dim=0
            ).to(device)
            pred_pts, _, _ = aggregate_chunks(
                ff_pts_all, ff_pts_conf_all, msks_in_scene, scene
            )

        t1 = time.time()
        logger.write(
            {"runtime": t1 - t0},
            prefix="runtime",
            dataset_key=scene["dataset_name"],
            seq_key=scene["scene_name"],
        )
        logger.write(
            {"chunk_num": len(scene_chunks)},
            prefix="chunk_num",
            dataset_key=scene["dataset_name"],
            seq_key=scene["scene_name"],
        )
        eval_mask = scene["gt_msks"].to(device)
        gt_pts = scene["gt_pts_metric"].to(device)
        max_error = 0.03

        metric_init, ff_pts_aligned, _ = eval_points(
            gt_points=gt_pts,
            pred_points=scene["ff_pts_original"].to(device),
            eval_mask=eval_mask,
            umeyama=True,
            umeyama_max_error=max_error,
        )

        metric_pred, pred_pts_aligned, _ = eval_points(
            gt_points=gt_pts,
            pred_points=pred_pts,
            eval_mask=eval_mask,
            umeyama=True,
            umeyama_max_error=max_error,
        )

        metric_geo, geo_pts_aligned, _ = eval_points(
            gt_points=gt_pts,
            pred_points=scene["geo_pts"].to(device),
            eval_mask=eval_mask,
            align_mask=eval_mask&scene["geo_msks"].to(device),
            umeyama=True,
            umeyama_max_error=max_error,
        )
        imgs_flat = scene["images"].reshape(-1, 3)
        if vis:
            xyzs, rgbs = [], []
            for chunk_i in range(len(scene_chunks)):
                xyz = pred_pts_aligned[scene_chunks[chunk_i]['msks_in_scene']]
                rgb = torch.rand(1,3).expand(xyz.shape[0],3)
                xyzs.append(xyz)
                rgbs.append(rgb)
            xyzs = torch.cat(xyzs, dim=0)
            rgbs = torch.cat(rgbs, dim=0)
            if len(scene_chunks) > 0:
                save_xyzrgb_to_ply(xyzs[::10], rgbs[::10], os.path.join(logger.dirname, scene['dataset_name'],scene['scene_name'], "chunks.ply"))
            scene_dir = os.path.join(
                logger.dirname, scene["dataset_name"], scene["scene_name"]
            )
            save_xyzrgb_to_ply(
                pred_pts_aligned.reshape(-1, 3),
                imgs_flat,
                os.path.join(scene_dir, "ggpt.ply"),
            )
            gt_m = scene["gt_msks"].reshape(-1)
            save_xyzrgb_to_ply(
                gt_pts[scene["gt_msks"]],
                imgs_flat[gt_m],
                os.path.join(scene_dir, "gt.ply"),
            )
            geo_m = scene["geo_msks"].reshape(-1)
            save_xyzrgb_to_ply(
                geo_pts_aligned[scene["geo_msks"]],
                imgs_flat[geo_m],
                os.path.join(scene_dir, "geo.ply"),
            )
            save_xyzrgb_to_ply(
                ff_pts_aligned.reshape(-1, 3),
                imgs_flat,
                os.path.join(scene_dir, "ff.ply"),
            )
        logger.write(
            metric_init,
            prefix="init",
            dataset_key=scene["dataset_name"],
            seq_key=scene["scene_name"],
        )
        logger.write(
            metric_pred,
            prefix="pred",
            dataset_key=scene["dataset_name"],
            seq_key=scene["scene_name"],
        )
        logger.write(
            metric_geo,
            prefix="geo",
            dataset_key=scene["dataset_name"],
            seq_key=scene["scene_name"],
        )
        logger.save(False)

    logger.save(True)
    eval_elapsed = time.time() - eval_start_time
    Print(
        "[Eval][END] "
        f"step={step} "
        f"log_suffix='{log_suffix}' "
        f"num_scenes={len(val_dataloader)} "
        f"elapsed_sec={eval_elapsed:.2f} "
        f"log_dir={log_dir}"
    )
    return logger


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_ggpt")
def main(cfg):
    set_seed(cfg.common_config.seed)
    if not cfg.common_config.ddp:
        raise NotImplementedError("DDP is required for benchmark_ggpt.")
    _, device_id = init_DDP()
    device = f"cuda:{device_id}"
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model = instantiate(cfg.ggptmodel_config).eval()
    raw = torch.load(cfg.common_config.load_ckpt, map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device_id], find_unused_parameters=False
    )

    val_dataloader = build_val_dataloader(cfg)
    evaluate_ggpt(model, val_dataloader, output_dir, 0)


if __name__ == "__main__":
    main()
