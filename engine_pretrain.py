# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

from typing import Iterable
import os
import torch
import torch.distributed as dist
import util.misc as misc
import util.lr_sched as lr_sched
from util.language_metrics import compute_language_model_scores


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    mask_ratio = args.mask_ratio
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        with torch.cuda.amp.autocast():
            losses, _, _ = model(batch, mask_ratio=mask_ratio, epoch=epoch)

            loss = 0.
            loss_values = []
            for i, loss_i in enumerate(losses):
                loss_values.append(loss_i.item())
                if i == 1:
                    lam = 0.15
                else:
                    lam = 1
                loss += lam * loss_i
            loss = loss / accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            for i, loss_value in enumerate(loss_values):
                metric_logger.update(**{f'loss{i}':loss_value})

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_values_reduce = []
            for loss_value in loss_values:
                loss_value_reduce = misc.all_reduce_mean(loss_value)
                loss_values_reduce.append(loss_value_reduce)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                for i, loss_value_reduce in enumerate(loss_values_reduce):
                    log_writer.add_scalar(f'train_loss{i}', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def eval(model: torch.nn.Module,
                    data_loader: Iterable, epoch: int,
                    log_writer=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Eval Epoch: [{}]'.format(epoch)
    print_freq = 10

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    gens, refs = {}, {}
    # for data_iter_step, batch in enumerate(data_loader):
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        with torch.cuda.amp.autocast():
            gen_reports, ref_reports = model.module.generate(batch)
            gens.update(gen_reports)
            refs.update(ref_reports)

            torch.cuda.synchronize()
    gens = gather_objects(gens, world_size=dist.get_world_size())
    refs = gather_objects(refs, world_size=dist.get_world_size())
    image_ids = list(gens.keys())
    gens = [gens[iid].replace('\n', '') for iid in image_ids]
    refs = [refs[iid].replace('\n', '') for iid in image_ids]
    if get_rank() == 0:
        print(f'Generated [{len(gens)}/{len(refs)}] | {len(data_loader)}')
        gen_res = [x+'\n' for x in gens]
        test_gts = [x+'\n' for x in refs]
        res_dir = log_writer.log_dir.replace('logs', 'results')
        os.makedirs(res_dir, exist_ok=True)
        output_res_file = os.path.join(res_dir, f'gen_res_epoch{epoch}.txt')
        output_gts_file = os.path.join(res_dir, f'test_gts_epoch{epoch}.txt')
        with open(output_res_file, 'w') as f:
            f.writelines(gen_res)
        with open(output_gts_file, 'w') as f:
            f.writelines(test_gts)
        print('Write results to ', output_res_file, output_gts_file)

        gen_and_ref_reports = {"generated_reports": gens, "reference_reports": refs}
        test_met = compute_language_model_scores(gen_and_ref_reports)
        test_met = test_met['main_metrics']
        metlog = [f'eval_{k}: {v}\n' for k, v in test_met.items()]
        output_scores_file = os.path.join(res_dir, 'scores.txt')
        with open(output_scores_file, 'a') as f:
            f.write(f'Epoch {epoch}:\n')
            f.writelines(metlog)
        metric_logger.update(**test_met)
        print("Averaged stats:", metric_logger)
        return test_met['bleu_4']
    else:
        return 0

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def gather_objects(local_object, world_size: int):
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_object)
    merged_dict = {}
    for d in gathered:
        merged_dict.update(d)
    return merged_dict
