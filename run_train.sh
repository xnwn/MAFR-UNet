#!/bin/bash

# Directories
save_dir="./model"
list_dir="./list/Dongyang"
data_dir="/mnt/2T/LJ/Dongyang/train_npz"

# Training Parameters
seed=1234
deterministic=1
batch_size=24
pretrain_ckpt_path="./pretrained_ckpt/new_cswin_tiny_224.pth"
base_lr=0.05
epochs=100

# Network Parameters
in_channels=3
embed_dim=64
img_size=224
drop_path_probability=0.2
block_num_list="[1, 2, 9, 1]"
dropout_probability=0.
qkv_bias=True
stripe_width_list="[1, 2, 7, 7]"
num_heads_list="[2, 4, 8, 16]"
qk_scale=None
mlp_ratio=4.
num_classes=2

# Run the training script with the specified parameters
python train.py \
    --save_dir "$save_dir" \
    --list_dir "$list_dir" \
    --data_dir "$data_dir" \
    --seed "$seed" \
    --deterministic "$deterministic" \
    --batch_size "$batch_size" \
    --pretrain_ckpt_path "$pretrain_ckpt_path" \
    --base_lr "$base_lr" \
    --epochs "$epochs" \
    --in_channels "$in_channels" \
    --embed_dim "$embed_dim" \
    --img_size "$img_size" \
    --drop_path_probability "$drop_path_probability" \
    --block_num_list "$block_num_list" \
    --dropout_probability "$dropout_probability" \
    --qkv_bias "$qkv_bias" \
    --stripe_width_list "$stripe_width_list" \
    --num_heads_list "$num_heads_list" \
    --qk_scale "$qk_scale" \
    --mlp_ratio "$mlp_ratio" \
    --num_classes "$num_classes"