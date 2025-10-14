# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MRM: https://github.com/RL4M/MRM-pytorch
# CheXzero: https://github.com/rajpurkarlab/CheXzero
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import model_ssacl
from engine_pretrain import train_one_epoch, eval
from util.pretrain_datasets import MultimodalBertDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def calculate_img_stats_full(dataset):
    imgs_ = torch.stack([img for img,_ in dataset],dim=3)
    imgs_ = imgs_.view(3,-1)
    imgs_mean = imgs_.mean(dim=1)
    imgs_std = imgs_.std(dim=1)
    return imgs_mean,imgs_std

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    decay2 = []
    no_decay2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if 'opt_model' in name:
                no_decay2.append(param)
            else:
                no_decay.append(param)
        else:
            if 'opt_model' in name:
                decay2.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay2, 'weight_decay': 0., 'lr_scale': 0.25},
        {'params': decay2, 'weight_decay': weight_decay, 'lr_scale': 0.25},
        ]

def get_args_parser():
    parser = argparse.ArgumentParser('SS-ACL pre-training', add_help=False)

    parser.add_argument('--epochs', default=32, type=int)
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--model', default='ssacl', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=2.5e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='initial warmup LR')

    parser.add_argument('--data_path', default='/path/to/mimic_cxr/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='mae_pretrain_vit_base.pth',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--batch_size', default=512, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--eval_batch_size', default=16, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--eval_freq', default=1, type=int, help='evaluation frequency, default to 1')
    parser.add_argument('--prefix', default='SS-ACL', type=str)
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='Masking ratio (percentage of removed patches).')  # 0.75
    parser.add_argument('--lam', default=0.9, type=float)
    parser.add_argument('--T', default=0.03, type=float)
    parser.add_argument('--SR', default=1.0, type=float)

    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])

    return parser
 


def main(args):

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # simple augmentation
    if args.SR == 0:
        transform_train = transforms.Compose([
            transforms.Resize(
                (224, 224), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),    
            transforms.RandomAffine(degrees=20, scale=(0.8, 1.2)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4978], std=[0.2449])
            ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),    
            transforms.RandomAffine(degrees=20, scale=(0.8, 1.2)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4978], std=[0.2449])
            ])
        
    transform_test = transforms.Compose([
            transforms.Resize(
                (224, 224), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4978], std=[0.2449])
            ])
                
    dataset_train = MultimodalBertDataset(os.path.join(args.data_path), transform=transform_train, SR=args.SR, split='train')
    dataset_val = MultimodalBertDataset(os.path.join(args.data_path), transform=transform_train, SR=args.SR, split='val')
    dataset_test = MultimodalBertDataset(os.path.join(args.data_path), transform=transform_test, SR=args.SR, split='test')


    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False
        )
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    args.log_dir = os.path.join(args.output_dir, args.prefix, timestamp, "logs")
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=dataset_train.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=dataset_train.collate_fn
    )

    # define the model
    model = model_ssacl.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, T=args.T, lam=args.lam, SR=args.SR, warmE = args.warmup_epochs)
    model.to(device)

    if args.mode == 'train':
        model_without_ddp = model

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        
        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        
        # following timm: set wd as 0 for bias and norm layers
        param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer),
        loss_scaler = NativeScaler()

        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        try:
            misc.mkdir(os.path.join(args.output_dir, args.prefix, timestamp, "model"))
            print(f"Start training for {args.epochs} epochs")
        except:
            print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        
        best_score = 0.
        best_epoch = 0
        
        for epoch in range(args.start_epoch, args.epochs):
            # if epoch == 0:
            #     eval(
            #         model, data_loader_test, epoch=-1, log_writer=log_writer,
            #     )

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
                data_loader_test.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
            if args.output_dir and ((epoch) % 5 == 0 or epoch + 1 == args.epochs):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, timestamp=timestamp)
                
            if args.output_dir and ((epoch) % args.eval_freq == 0 or epoch + 1 == args.epochs):
                current_score = eval(
                    model, data_loader_test, epoch, log_writer=log_writer,
                )
                if global_rank==0 and current_score >= best_score:
                    best_score = current_score
                    best_epoch = epoch
                    misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, best=True, timestamp=timestamp)

                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'best_score': best_score,
                            'best_epoch': best_epoch,
                            'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, args.prefix, timestamp, 'logs', "log.txt"), mode="a", encoding="utf-8") as f:
                f.write('"%s"' %  (args.prefix) + ',' + ": ")
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        if int(os.environ['LOCAL_RANK'])==0:
            with open(os.path.join(args.output_dir, args.prefix, timestamp, 'logs', "train_log.txt"), "a") as file:
                file.write('"%s"' %  (args.prefix) + ',' + "\n")
    else:
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict and (v.shape == model_dict[k].shape)}
            print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
        
        model_without_ddp = model
        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        eval(model, data_loader_test, 0, log_writer=log_writer)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
