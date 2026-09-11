import os
import sys

import hydra
import torch
import torch.distributed as dist
import wandb
from hydra.utils import instantiate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.basic import Print
from utils.common import move_to_device, set_seed

from ggpt.dataloader import (
    build_train_dataloader,
    build_train_eval_dataloader,
    build_val_dataloader,
)
from ggpt.run_benchmark_ggpt import evaluate_ggpt

def _hydra_output_dir(cfg):
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


def _checkpoint_dir(cfg):
    return os.path.join(_hydra_output_dir(cfg), "checkpoints")


def _has_resumable_checkpoint(ckptdir):
    if not os.path.isdir(ckptdir):
        return False
    if os.path.isfile(os.path.join(ckptdir, "last.bin")):
        return True
    return any(
        f.startswith("model.step") and f.endswith(".pth") for f in os.listdir(ckptdir)
    )


def init_DDP():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device_id = rank % torch.cuda.device_count()
    return rank, device_id


def load_latest_ckpt(cfg, ckptdir, model, optimizer=None, scaler=None, device="cuda"):
    if not os.path.exists(ckptdir):
        return 0
    last_path = os.path.join(ckptdir, "last.bin")
    if os.path.isfile(last_path):
        try:
            return load_ckpt(cfg, last_path, model, optimizer, scaler, device)
        except Exception as e:
            Print("Fail to load " + last_path + "   ")
            Print(e)
    candidates = [
        f for f in os.listdir(ckptdir) if f.startswith("model.step") and f.endswith(".pth")
    ]
    ckpts_list = sorted(
        candidates, key=lambda c: int(c.replace("model.step", "").replace(".pth", ""))
    )
    ci = len(ckpts_list) - 1
    while ci >= 0:
        try:
            global_step = load_ckpt(
                cfg,
                os.path.join(ckptdir, ckpts_list[ci]),
                model,
                optimizer,
                scaler,
                device,
            )
        except Exception as e:
            Print("Fail to load " + os.path.join(ckptdir, ckpts_list[ci]) + "   ")
            Print(e)
            ci -= 1
            continue
        break
    if ci >= 0:
        return global_step
    return 0


def load_ckpt(cfg, ckptfile, model, optimizer=None, scaler=None, device='cuda'):
    if ckptfile=='last':
        return load_latest_ckpt(cfg, _checkpoint_dir(cfg), model, optimizer, scaler, device)
    assert os.path.exists(ckptfile)
    state_dict = torch.load(ckptfile, map_location=device)
    model.load_state_dict(state_dict['model'], strict=True) # not DDP model
    if optimizer is not None and 'optimizer' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer'])
    if scaler is not None and 'scaler' in state_dict:
        scaler.load_state_dict(state_dict['scaler'])
    global_step = int(state_dict['steps']) if 'steps' in state_dict else 0
    Print(f'Successfully load checkpoint from {ckptfile}, global_step: {global_step}')
    return global_step

def _train_uses_octree_iterable(cfg) -> bool:
    """Iterable octree path has uneven steps per rank; needs cross-rank epoch sync on exhaustion."""
    tdc = cfg.traindataset_configs
    return (
        getattr(tdc, "chunk_sample", None) == "octree"
        and getattr(tdc, "octree_exhaustive", False)
    )


def _octree_iterable_set_epoch(dataloader, cfg, epoch: int) -> None:
    if not _train_uses_octree_iterable(cfg):
        return
    ds = dataloader.dataset
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(epoch)


def _next_train_batch_sync_iterable(
    dataiter, dataloader, device_id, cfg, octree_epoch_holder=None
):
    """
    If any rank hit StopIteration on its iterable, all ranks barrier and restart iterators together.
    Ranks that still had data drop the rest of that logical epoch (DDP-safe).
    """
    batchdata_list = None
    local_exhausted = False
    try:
        batchdata_list = next(dataiter)
    except StopIteration:
        local_exhausted = True

    if cfg.common_config.ddp and _train_uses_octree_iterable(cfg):
        flag = torch.tensor(
            [1 if local_exhausted else 0], device=device_id, dtype=torch.int32
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0:
            dist.barrier()
            if octree_epoch_holder is not None:
                octree_epoch_holder[0] += 1
                _octree_iterable_set_epoch(dataloader, cfg, octree_epoch_holder[0])
            dataiter = iter(dataloader)
            batchdata_list = next(dataiter)
    else:
        if local_exhausted:
            if octree_epoch_holder is not None:
                octree_epoch_holder[0] += 1
                _octree_iterable_set_epoch(dataloader, cfg, octree_epoch_holder[0])
            dataiter = iter(dataloader)
            batchdata_list = next(dataiter)

    return batchdata_list, dataiter


def init_wandb(cfg):
    if dist.get_rank() == 0:
        id_, resume = None , None
        wandb_dir = os.path.join(_hydra_output_dir(cfg), "wandb")
        if cfg.load_ckpt is not None and cfg.train_config.wandb_config.resume:
            latest_run_file = os.path.join(wandb_dir, "wandb", "latest-run")
            if os.path.exists(latest_run_file):
                realpath = os.path.basename(os.path.realpath(latest_run_file))
                id_ = realpath.split('-')[-1] 
                cfg.train_config.wandb_config.resume = 'allow'
            else:
                cfg.train_config.wandb_config.resume = False
        if cfg.eval_only == False:
            run_name = os.path.basename(_hydra_output_dir(cfg).rstrip("/"))
            wandb.init(**cfg.train_config.wandb_config, dir=wandb_dir, name=run_name, id=id_)
    dist.barrier()


def _should_log_progress(step: int, start_step: int, final_step: int, interval: int) -> bool:
    if step == start_step or step == final_step:
        return True
    return interval > 0 and ((step - start_step) % interval == 0)


def _sanitize_metric_key(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_")


def _extract_train_chunk_stats(raw_batch):
    """
    raw_batch: list of (chunk, scene) tuples from DataLoader.
    Returns:
      chunks_only: list[chunk]
      local_chunk_count: int
      local_ff_points_sum: int
      per_scene: dict[scene_key] -> {"scene_chunk_count": int, "scene_avg_ff_points_per_chunk": float}
    """
    chunks_only = []
    local_chunk_count = 0
    local_ff_points_sum = 0
    per_scene_counts = {}
    per_scene_points = {}
    per_scene_meta = {}

    for item in raw_batch:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            chunk = item[0]
            scene = item[1] if len(item) > 1 else None
        else:
            chunk = item
            scene = None

        chunks_only.append(chunk)
        local_chunk_count += 1
        ff_n = int(chunk["ff_pts"].shape[0]) if isinstance(chunk, dict) and "ff_pts" in chunk else 0
        local_ff_points_sum += ff_n

        if isinstance(scene, dict):
            dname = scene.get("dataset_name", "unknown")
            sname = scene.get("scene_name", "unknown")
            scene_key = f"{dname}/{sname}"
            per_scene_counts[scene_key] = per_scene_counts.get(scene_key, 0) + 1
            per_scene_points[scene_key] = per_scene_points.get(scene_key, 0) + ff_n
            if (
                isinstance(chunk, dict)
                and "scene_chunk_count" in chunk
                and "scene_avg_ff_points_per_chunk" in chunk
                and scene_key not in per_scene_meta
            ):
                per_scene_meta[scene_key] = {
                    "scene_chunk_count": int(chunk["scene_chunk_count"]),
                    "scene_avg_ff_points_per_chunk": float(
                        chunk["scene_avg_ff_points_per_chunk"]
                    ),
                }

    per_scene = {}
    for k, c in per_scene_counts.items():
        pts = per_scene_points.get(k, 0)
        if k in per_scene_meta:
            per_scene[k] = per_scene_meta[k]
        else:
            per_scene[k] = {
                "scene_chunk_count": int(c),
                "scene_avg_ff_points_per_chunk": float(pts) / float(max(c, 1)),
            }

    return chunks_only, local_chunk_count, local_ff_points_sum, per_scene


@hydra.main(version_base=None, config_path="../configs/train", config_name="train")
def main(cfg):
    rank, device_id = init_DDP()
    set_seed(cfg.common_config.seed)
    val_dataloader = build_val_dataloader(cfg)

    model = instantiate(cfg.model_config).to(device_id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], find_unused_parameters=False)

    if cfg.eval_only:
        if cfg.load_ckpt is None:
            raise ValueError("eval_only=True requires load_ckpt (checkpoint path or 'last').")
        ckpt_dir = _checkpoint_dir(cfg)
        if cfg.load_ckpt == "last" and not _has_resumable_checkpoint(ckpt_dir):
            raise FileNotFoundError(
                f"eval_only with load_ckpt=last found no checkpoint in {ckpt_dir}. "
                "Pass an explicit path, or set hydra.run.dir to a run that contains checkpoints."
            )
        eval_step = load_ckpt(
            cfg,
            cfg.load_ckpt,
            model,
            optimizer=None,
            scaler=None,
            device=f"cuda:{device_id}",
        )
        model.eval()
        init_wandb(cfg)
        evaluate_ggpt(model, val_dataloader, _hydra_output_dir(cfg), eval_step, vis=True)
        return

    train_eval_dataloader = build_train_eval_dataloader(cfg)
    dataloader = build_train_dataloader(cfg)
    octree_epoch_holder = [0] if _train_uses_octree_iterable(cfg) else None
    if octree_epoch_holder is not None:
        _octree_iterable_set_epoch(dataloader, cfg, 0)
    dataiter = iter(dataloader)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_config.optimizer_config.lr)
    scaler = torch.cuda.amp.GradScaler() if cfg.enable_amp else None

    if cfg.load_ckpt is not None:
        start_step = load_ckpt(cfg, cfg.load_ckpt, model, optimizer=optimizer, scaler=scaler, device=f'cuda:{device_id}')
    else:
        start_step = 0

    loss_computer = instantiate(cfg.train_config.loss)

    init_wandb(cfg)
    final_step = cfg.train_config.optimizer_config.global_step
    progress_log_every = int(getattr(cfg.train_config, "progress_log_every", 200))

    for step in range(start_step, final_step + 1):
        if (step == start_step or step % cfg.train_config.eval_every == 0):
            model.eval()
            logger = evaluate_ggpt(model, val_dataloader, _hydra_output_dir(cfg), step)
            if train_eval_dataloader is not None:
                dist.barrier()
                logger_train_eval = evaluate_ggpt(
                    model,
                    train_eval_dataloader,
                    _hydra_output_dir(cfg),
                    step,
                    log_suffix="_train_eval",
                )
            else:
                logger_train_eval = None
            model.train()
            if rank == 0 and wandb.run is not None:
                log_payload = {}
                avgs = getattr(logger, "dataset2average", None)
                if avgs:
                    for dataset_key, metrics in avgs.items():
                        for metric_name, value in metrics.items():
                            log_payload[f"Val/{dataset_key}/{metric_name}"] = value
                if logger_train_eval is not None:
                    avgs_te = getattr(logger_train_eval, "dataset2average", None)
                    if avgs_te:
                        for dataset_key, metrics in avgs_te.items():
                            for metric_name, value in metrics.items():
                                log_payload[
                                    f"TrainEval/{dataset_key}/{metric_name}"
                                ] = value
                if log_payload:
                    wandb.log(log_payload, step=step)
        if step % cfg.train_config.save_every == 0 and rank == 0:
            ckpt_dir = _checkpoint_dir(cfg)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "steps": step},
                os.path.join(ckpt_dir, "last.bin"),
            )
        batchdata_list, dataiter = _next_train_batch_sync_iterable(
            dataiter,
            dataloader,
            device_id,
            cfg,
            octree_epoch_holder=octree_epoch_holder,
        )
        # Each item from the dataloader is a (chunk, scene) tuple; collect chunk stats before dropping scene metadata.
        (
            batchdata_list,
            local_chunk_count,
            local_ff_points_sum,
            per_scene_chunk_stats,
        ) = _extract_train_chunk_stats(batchdata_list)

        # Global (cross-rank) chunk/point stats for DDP-friendly monitoring.
        if cfg.common_config.ddp:
            stats_tensor = torch.tensor(
                [local_chunk_count, local_ff_points_sum],
                device=device_id,
                dtype=torch.float64,
            )
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
            global_chunk_count = int(stats_tensor[0].item())
            global_ff_points_sum = float(stats_tensor[1].item())
        else:
            global_chunk_count = int(local_chunk_count)
            global_ff_points_sum = float(local_ff_points_sum)

        batchdata_list = move_to_device(batchdata_list, device_id)

        with torch.cuda.amp.autocast(enabled=cfg.enable_amp):
            out = model(batchdata_list)
        #Compute loss
        loss_dict = loss_computer(out)
        loss = loss_dict['loss/objective']
        if cfg.enable_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if rank==0:
            log_payload = {f"Train/{k}": v for k, v in loss_dict.items()}
            # Step-level aggregates (kept for quick monitoring).
            log_payload["Train/chunks_per_step"] = global_chunk_count
            log_payload["Train/avg_ff_points_per_chunk_step"] = (
                global_ff_points_sum / max(global_chunk_count, 1)
            )
            '''
            # Scene-level octree stats (from chunk_by_octree leaves), same for every chunk in a scene.
            if per_scene_chunk_stats:
                scene_chunk_counts = []
                scene_avg_pts = []
                for scene_key, st in per_scene_chunk_stats.items():
                    scene_metric_key = _sanitize_metric_key(scene_key)
                    scene_chunk_counts.append(float(st["scene_chunk_count"]))
                    scene_avg_pts.append(float(st["scene_avg_ff_points_per_chunk"]))
                log_payload["Train/scene_chunk_count_mean"] = float(
                    sum(scene_chunk_counts) / max(len(scene_chunk_counts), 1)
                )
                log_payload["Train/scene_avg_ff_points_per_chunk_mean"] = float(
                    sum(scene_avg_pts) / max(len(scene_avg_pts), 1)
                )
            '''
            wandb.log(log_payload, step=step)
            if _should_log_progress(step, start_step, final_step, progress_log_every):
                Print(
                    f"step={step}/{final_step} "
                    f"loss={float(loss.detach().item()):.6g} "
                    f"batch_size={len(batchdata_list)}"
                )
        if cfg.train_config.optimizer_config.grad_clip_norm>0:
            if cfg.enable_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train_config.optimizer_config.grad_clip_norm)
        if cfg.enable_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

if __name__ == "__main__":
    main()