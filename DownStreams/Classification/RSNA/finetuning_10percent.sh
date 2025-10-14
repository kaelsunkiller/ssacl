#!/bin/bash
export https_proxy="http://100.84.94.139:7890"
cuda=$1
gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python3 -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]}  --master_addr 127.0.0.1 --master_port 29610 train.py \
    --name ssacl_ft_10p --stage train --model vit_base_patch16 --model_type ViT-B_16 --num_classes 1 \
    --pretrained_path "/path/to/pretrained_weights/ssacl-best.pth" \
    --dataset_path '/path/to/rsna/images' \
    --dataset_split_path '/path/to/datasplits/RSNA_Pneumonia' \
    --output_dir "ssacl_ft_10/" --data_volume '10' --num_steps 30000  --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 500 --fp16 --fp16_opt_level O2 --train_batch_size 96
