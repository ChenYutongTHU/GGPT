export HYDRA_FULL_ERROR=1
num_gpus=4
python -m torch.distributed.run --nnodes=1 --nproc_per_node=$num_gpus --rdzv-endpoint=localhost:2026 \
    sfm/run_benchmark_sfm.py --config-name=train/run-save-sfm
# Intermediate results are saved in outputs/benchmark/sfm_vggt-point-romav2-base/save and will be used for GGPT.


