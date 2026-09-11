export SCANNETPP_ROOT=YOUR_PATH/scannetpp/data
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --rdzv-endpoint=localhost:2026 \
    data/sample_views.py \
    sample_configs.max_seq_num=16