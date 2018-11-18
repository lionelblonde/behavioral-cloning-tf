#!/usr/bin/env bash
# Example: ./mujoco_evaluate.sh <env_id> <xpo_pol_ckpt_dir_path>

python -m clone \
    --note="" \
    --env_id=$1 \
    --seed=0 \
    --checkpoint_dir="data/imitation_checkpoints" \
    --summary_dir="data/summaries" \
    --log_dir="data/logs" \
    --task="evaluate_bc_policy" \
    --num_trajs=20 \
    --rmsify_obs \
    --hid_widths 32 32 \
    --hid_nonlin="leaky_relu" \
    --hid_w_init="he_normal" \
    --render \
    --model_ckpt_dir=$2
