#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import time
import logging

import torch.distributed as dist


def setup_logger(name, logpth, log_level):
    logtime = time.strftime('%Y-%m-%d-%H-%M-%S')
    logfile = '{}-{}.log'.format(name, logtime)
    # logfile = '{}-{}.log'.format(name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = os.path.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    # log_level = logging.INFO
    # if dist.is_initialized() and dist.get_rank() != 0:
    #     log_level = logging.WARNING
    level_dict = {
                'debug':logging.DEBUG,
                'info':logging.INFO,
                'warning':logging.WARNING,
                'error':logging.ERROR,
                'critical':logging.CRITICAL,
                }
    if log_level in level_dict:
        log_level = level_dict[log_level]
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())
    return logtime


def print_log_msg(epoch, max_epoch, it, max_iter, lr, time_meter, loss_meter, loss_pre_meter,
        loss_aux_meters):
    t_intv, eta = time_meter.get()
    loss_avg, _ = loss_meter.get()
    loss_pre_avg, _ = loss_pre_meter.get()
    loss_aux_avg = ', '.join(['{}: {:.4f}'.format(el.name, el.get()[0]) for el in loss_aux_meters])
    msg = ', '.join([
        'epoch: {epoch}/{max_epoch}',
        'iter: {it}/{max_it}',
        'lr: {lr:4f}',
        'eta: {eta}',
        'time: {time:.2f}',
        'loss: {loss:.4f}',
        'loss_pre: {loss_pre:.4f}',
    ]).format(
        epoch=epoch+1,
        max_epoch=max_epoch,
        it=it+1,
        max_it=max_iter,
        lr=lr,
        time=t_intv,
        eta=eta,
        loss=loss_avg,
        loss_pre=loss_pre_avg,
        )
    msg += ', ' + loss_aux_avg
    logger = logging.getLogger()
    logger.info(msg)
