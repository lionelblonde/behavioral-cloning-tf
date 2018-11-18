#!/usr/bin/env bash
# Example: ./mujoco_clone.sh <num_mpi_workers> <env_id> <expert_demos_path> <num_demos>

mpirun -np $1 --bind-to core python -m clone \
    --note="" \
    --env_id=$2 \
    --seed=0 \
    --checkpoint_dir="data/imitation_checkpoints" \
    --summary_dir="data/summaries" \
    --log_dir="data/logs" \
    --task="clone" \
    --expert_path=$3 \
    --num_demos=$4 \
    --rmsify_obs \
    --clip_norm=5. \
    --save_frequency=10 \
    --num_iters=10000000 \
    --batch_size=64 \
    --hid_widths 32 32 \
    --hid_nonlin="leaky_relu" \
    --hid_w_init="he_normal" \
    --lr=3e-4
