python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv-endpoint=localhost:29510 \
ggpt/train.py \
hydra.run.dir=outputs/train/ggpt-romav2-scannet_voxoct-stt-bs1-200k \
train_config.dataloader_config.batch_size=1