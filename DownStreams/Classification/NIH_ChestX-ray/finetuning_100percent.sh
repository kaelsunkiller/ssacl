#!/bin/bash
cuda=$1
gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python3 -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]}  --master_addr 127.0.0.1 --master_port 29506 train.py \
    --pretrained_path "/path/to/pretrained_weights/ssacl-best.pth" \
    --dataset_path '/path/to/dataset/ChestXray8/' \
    --data_volume '100' --num_steps 200000  --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 5000 --fp16 --fp16_opt_level O2 --train_batch_size 96 \
    --freeze_backbone 0 \
    --last ssacl_ft_100p

