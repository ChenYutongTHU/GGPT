import torch 
from torch.utils.data import DataLoader, DistributedSampler
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ggpt.dataloader.train_dataset import OctreeExhaustiveIterableTrain

def collate_fn(batch):
    #Every batch has different number of points, so we cannot stack them directly
    return batch


def grouped_octree_train_collate(batch):
    """Expand one dataset sample ([chunk_0, ...], scene) into a list of (chunk, scene)."""
    if len(batch) == 1 and isinstance(batch[0][0], list):
        chunks, scene = batch[0]
        return [(c, scene) for c in chunks]
    return batch


def build_train_eval_dataloader(cfg):
    """Optional train-data subset for eval; returns None if train_eval_subset_configs is unset."""
    te = OmegaConf.select(cfg, "train_eval_subset_configs", default=None)
    if te is None:
        return None
    te_dataset = instantiate(te)
    if cfg.common_config.ddp:
        te_sampler = DistributedSampler(te_dataset, shuffle=False)
        return DataLoader(
            te_dataset,
            batch_size=1,
            num_workers=cfg.common_config.num_workers,
            collate_fn=collate_fn,
            sampler=te_sampler,
            drop_last=False,
        )
    return DataLoader(
        te_dataset,
        batch_size=1,
        num_workers=cfg.common_config.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )


def build_val_dataloader(cfg):
    val_dataset = instantiate(cfg.valdataset_configs)
    if cfg.common_config.ddp:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=cfg.common_config.num_workers, collate_fn=collate_fn, sampler=val_sampler, drop_last=False)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=cfg.common_config.num_workers, collate_fn=collate_fn, shuffle=False, drop_last=False)
    return val_dataloader


def build_train_dataloader(cfg):
    dataset = instantiate(cfg.traindataset_configs)
    loader_kwargs = {k: v for k, v in cfg.train_config.dataloader_config.items()}
    num_workers = loader_kwargs.pop(
        "num_workers", cfg.common_config.num_workers
    )
    batch_size = int(loader_kwargs.get("batch_size", 1))
    train_collate = collate_fn
    if (
        getattr(dataset, "mode", None) == "train"
        and getattr(dataset, "chunk_sample", None) == "octree"
    ):
        if not getattr(dataset, "octree_exhaustive", False):
            raise ValueError(
                "Training with chunk_sample='octree' requires octree_exhaustive=True. "
                "Octree training uses OctreeExhaustiveIterableTrain (one global worker slot per scene)."
            )
        dataset.train_octree_chunks_per_batch = batch_size
        if batch_size > 1:
            loader_kwargs["batch_size"] = 1
            train_collate = grouped_octree_train_collate
        else:
            loader_kwargs["batch_size"] = 1
        loader_kwargs.pop("shuffle", None)
        return DataLoader(
            OctreeExhaustiveIterableTrain(dataset),
            collate_fn=train_collate,
            drop_last=False,
            num_workers=num_workers,
            **loader_kwargs,
        )
    ddp_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    return DataLoader(
        dataset,
        collate_fn=train_collate,
        sampler=ddp_sampler,
        drop_last=False,
        num_workers=num_workers,
        **loader_kwargs,
    )