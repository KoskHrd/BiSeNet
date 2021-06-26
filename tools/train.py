#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
# import torch.distributed as dist
from torch.utils.data import DataLoader

from lib.models import model_factory
from configs import cfg_factory
# from lib.cityscapes_cv2 import get_data_loader
from lib.carla_cv2 import get_data_loader, prepare_data_loader
# from tools.evaluate import eval_model
from tools.evaluate import eval_model, test_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg

from tqdm import tqdm

# apex
has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False

print("has_apex: {}".format(has_apex))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count() if device.type == 'cuda' else None
print("device: {}".format(device))
print("device_count: {}".format(device_count))


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    # parse.add_argument('--port', dest='port', type=int, default=44554,)
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = cfg_factory[args.model]


## fix all random seeds
torch.manual_seed(cfg.random_seed)
torch.cuda.manual_seed(cfg.random_seed)
np.random.seed(cfg.random_seed)
random.seed(cfg.random_seed)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
# torch.multiprocessing.set_sharing_strategy('file_system')



def set_model():
    # net = model_factory[cfg.model_type](n_classes=19)
    net = model_factory[cfg.model_type](n_classes=cfg.n_classes)
    if not args.finetune_from is None:
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    if cfg.use_sync_bn: net = set_syncbn(net)
    net.to(device)
    net.train()
    #CHANGED: undo to use normal CrossEntropyLoss
    #FIXME: learn how to use OhemCrossEntropyLoss (Online Hard Example Mining)
    criteria_pre = OhemCELoss(0.7, cfg.anns_ignore)
    criteria_aux = [OhemCELoss(0.7, cfg.anns_ignore) for _ in range(cfg.num_aux_heads)]
    # criteria_pre = nn.CrossEntropyLoss(ignore_index=cfg.anns_ignore)
    # criteria_aux = [nn.CrossEntropyLoss(ignore_index=cfg.anns_ignore) for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux

def set_syncbn(net):
    if has_apex:
        net = parallel.convert_syncbn_model(net)
    else:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    return net


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:  # ex. batchnorm, bias, etc
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:  # ex. weight
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},  # ex. batchnorm, bias, etc
        ]
    if device_count >= 2:
        optim = torch.optim.SGD(
            params_list,  # weight decay (L2 penalty)
            lr=cfg.lr_start * device_count,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    else:
        optim = torch.optim.SGD(
            params_list,  # weight decay (L2 penalty)
            lr=cfg.lr_start,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    # optim = torch.optim.SGD(
    #     params_list,
    #     lr=cfg.lr_start,
    #     momentum=0.9,
    #     weight_decay=cfg.weight_decay,
    # )
    return optim


def set_model_dist(net):
    if has_apex:
        net = parallel.DistributedDataParallel(net, delay_allreduce=True)
    else:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank)
    return net


def set_meters(max_iter):
    time_meter = TimeMeter(max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def save_model(states, save_pth):
    logger = logging.getLogger()
    logger.info('\nsave models to {}'.format(save_pth))
    for name, state in states.items():
        save_name = 'model_final_{}.pth'.format(name)
        modelpth = os.path.join(save_pth, save_name)
        torch.save(state, modelpth)
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     torch.save(state, modelpth)


def train(loginfo):
    logger = logging.getLogger()
    # is_dist = dist.is_initialized()

    logger.info("config: \n{}".format([item for item in cfg.__dict__.items()]))

    # ## dataset
    # dl = get_data_loader(
    #         cfg.train_img_root, cfg.train_img_anns,
    #         cfg.imgs_per_gpu, cfg.scales, cfg.cropsize,
    #         cfg.max_iter, mode='train', distributed=is_dist)
    # dl = get_data_loader(
    #         cfg.train_img_root, cfg.train_img_anns,
    #         cfg.imgs_per_gpu, cfg.scales, cfg.cropsize,
    #         cfg.anns_ignore, cfg.max_iter, mode='train', distributed=False)
    dl = prepare_data_loader(
            cfg.train_img_root, cfg.train_img_anns, cfg.input_size,
            cfg.imgs_per_gpu, device_count, cfg.scales, cfg.cropsize,
            cfg.anns_ignore, mode='train', distributed=False)

    max_iter = cfg.max_epoch * len(dl.dataset) // (cfg.imgs_per_gpu * device_count) \
        if device == 'cuda' else cfg.max_epoch * len(dl.dataset) // cfg.imgs_per_gpu
    progress_iter = len(dl.dataset) / (cfg.imgs_per_gpu * device_count) // 5 \
        if device == 'cuda' else len(dl.dataset) / cfg.imgs_per_gpu // 5

    ## model
    net, criteria_pre, criteria_aux = set_model()
    net.to(device)
    if device_count >= 2:
        net = nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
        torch.multiprocessing.set_sharing_strategy('file_system')

    ## optimizer
    optim = set_optimizer(net)

    ## fp16
    if has_apex:
        opt_level = 'O1' if cfg.use_fp16 else 'O0'
        net, optim = amp.initialize(net, optim, opt_level=opt_level)

    ## ddp training
    # net = set_model_dist(net)
    # #CHANGED: normal training
    # #FIXME: GETTING STARTED WITH DISTRIBUTED DATA PARALLEL
    # #https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters(max_iter)

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    ## train loopx
    n_epoch = 0
    n_iter = 0
    best_valid_loss = np.inf
    while n_epoch < cfg.max_epoch:
        net.train()
        # for n_iter, (img, tar) in enumerate(dl):
        # for n_iter, (img, tar) in enumerate(tqdm(dl)):
        for (img, tar) in tqdm(dl, desc='train epoch {:d}/{:d}'.format(n_epoch+1, cfg.max_epoch)):
            img = img.to(device)
            tar = tar.to(device)

            tar = torch.squeeze(tar, 1)

            optim.zero_grad()
            logits, *logits_aux = net(img)
            loss_pre = criteria_pre(logits, tar)
            loss_aux = [crit(lgt, tar) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)
            if has_apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optim.step()
            torch.cuda.synchronize()
            lr_schdr.step()

            time_meter.update()
            loss_meter.update(loss.item())
            loss_pre_meter.update(loss_pre.item())
            _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

            ## print training log message
            # if (n_iter + 1) % 100 == 0:
            if (n_iter + 1) % progress_iter == 0:
                lr = lr_schdr.get_lr()
                lr = sum(lr) / len(lr)
                print_log_msg(
                    n_epoch, cfg.max_epoch, n_iter, max_iter, lr,
                    time_meter, loss_meter, loss_pre_meter, loss_aux_meters)

            n_iter = n_iter + 1

        #CHANGED: save weight with valid loss
        ## dump the final model and evaluate the result
        # save_pth = os.path.join(cfg.weight_path, 'model_final.pth')
        # logger.info('\nsave models to {}'.format(save_pth))
        # state = net.module.state_dict()
        # if dist.get_rank() == 0: torch.save(state, save_pth)
        logger.info('vaildating the {} epoch model'.format(n_epoch+1))
        valid_loss = valid(net, criteria_pre, criteria_aux, n_epoch, cfg, logger)
        if valid_loss < best_valid_loss:
            # save_path = os.path.join(cfg.weight_path,
            #     'epoch{:d}_valid_loss_{:.4f}.pth'.format(n_epoch, valid_loss))
            if not os.path.exists(cfg.weight_path): os.makedirs(cfg.weight_path)
            save_path = os.path.join(cfg.weight_path, 'model_bestValidLoss-{}.pth'.format(loginfo))
            logger.info('save models to {}'.format(save_path))
            torch.save(net.state_dict(), save_path)
            best_valid_loss = valid_loss

        # logger.info('\nevaluating the final model')
        logger.info('evaluating the {} epoch model'.format(n_epoch+1))
        torch.cuda.empty_cache()  ## For reset cuda memory used by cache
        # heads, mious = eval_model(net, 2, cfg.val_img_root, cfg.val_img_anns, cfg.n_classes)
        # logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
        # heads, mious, eious = eval_model(net, cfg, device_count, cfg.val_img_root, cfg.val_img_anns, cfg.n_classes, cfg.anns_ignore)
        heads, mious, eious = test_model(net, cfg, device_count, cfg.val_img_root, cfg.val_img_anns, cfg.n_classes, cfg.anns_ignore)
        logger.info('\n' + tabulate([mious, ], headers=heads, tablefmt='github', floatfmt=".8f"))
        logger.info('\n' + tabulate(np.array(eious).transpose(), headers=heads,
                    tablefmt='github', floatfmt=".8f", showindex=True))

        n_epoch = n_epoch + 1

    heads, mious, eious = eval_model(net, cfg, device_count, cfg.val_img_root, cfg.val_img_anns, cfg.n_classes, cfg.anns_ignore)
    logger.info('\n' + tabulate([mious, ], headers=heads, tablefmt='github', floatfmt=".8f"))
    logger.info('\n' + tabulate(np.array(eious).transpose(), headers=heads,
                tablefmt='github', floatfmt=".8f", showindex=True))

    return

def valid(net, criteria_pre, criteria_aux, epoch, cfg, logger):
    net.eval()
    valid_loss = 0.0
    dl = prepare_data_loader(
            cfg.val_img_root, cfg.val_img_anns, cfg.input_size,
            cfg.imgs_per_gpu, device_count, cfg.scales, cfg.cropsize,
            cfg.anns_ignore, mode='val', distributed=False)
    with torch.no_grad():
        for (img, tar) in tqdm(dl, desc='valid epoch {:d}/{:d}'.format(epoch+1, cfg.max_epoch)):
            img = img.to(device)
            tar = tar.to(device)
            tar = torch.squeeze(tar, 1)
            logits, *logits_aux = net(img)
            loss_pre = criteria_pre(logits, tar)
            loss_aux = [crit(lgt, tar) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)
            valid_loss = valid_loss + loss.item()
    valid_loss = valid_loss / len(dl.dataset) * cfg.imgs_per_gpu
    logger.info('epoch: {:d}/{:d}, valid loss: {:.4f}'.format(epoch+1, cfg.max_epoch, valid_loss))
    return valid_loss


def main():
    torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(
    #     backend='nccl',
    #     init_method='tcp://127.0.0.1:{}'.format(args.port),
    #     world_size=torch.cuda.device_count(),
    #     rank=args.local_rank
    # )
    if not os.path.exists(cfg.respth): os.makedirs(cfg.respth)
    np.set_printoptions(precision=8, floatmode='maxprec', suppress=True)
    # setup_logger('{}-train-{}'.format(cfg.model_type,cfg.log_level), cfg.respth, cfg.log_level)
    loginfo = setup_logger('{}-train-{}'.format(cfg.model_type,cfg.log_level),
                        cfg.respth, cfg.log_level)
    train(loginfo)


if __name__ == "__main__":
    main()
