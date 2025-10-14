#!/bin/bash
cuda=$1
gpu_ids=(${cuda//,/ })
gpu_count=${#gpu_ids[@]}
CUDA_VISIBLE_DEVICES=$cuda torchrun --nproc_per_node=$gpu_count --master_port=29502 main_pretrain.py \
    --SR 0 --mask_ratio 0.5 --lam 0.9 --T 0.03 --eval_batch_size 96 --prefix ssacl \
    --output_dir outputs/ --model ssacl \
    --resume pretrained_weights/ssacl-best.pth --mode test