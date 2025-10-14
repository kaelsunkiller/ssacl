import os

# parameter
gpu = "0"
name = "ssacl_ft_1_test"
file_path = "ssacl_ft_1/ssacl_ft_1_bestauc_checkpoint.bin"
model = "ViT-B_16"
print(os.system('CUDA_VISIBLE_DEVICES=' + gpu +' python3 train.py --name ' + name + ' --stage test --model_type ' + model +' --model vit_base_patch16 --num_classes 1 --pretrained_path ' + file_path +' --eval_batch_size 512 --img_size 224 --dataset_path /path/to/rsna/images --dataset_split_path /path/to/datasplits/RSNA_Pneumonia'))
    
