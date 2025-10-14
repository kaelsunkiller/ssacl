#!/bin/bash
cuda=$1
gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python3 -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]}  --master_addr 127.0.0.1 --master_port 29602 train.py \
    --name mrm199_ft_1 --stage train --model vit_base_patch16 --model_type ViT-B_16 --num_classes 1 \
    --pretrained_path "/path/to/pretrained_weights/ssacl-best.pth" \
    --dataset_path '/path/to/rsna/images' \
    --dataset_split_path '/path/to/datasplits/RSNA_Pneumonia' \
    --output_dir "ssacl_ft_1p/" --data_volume '1' --num_steps 3000  --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-2 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 96
