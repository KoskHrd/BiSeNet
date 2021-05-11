#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


#  import ohem_cpp
#  class OhemCELoss(nn.Module):
#
#      def __init__(self, thresh, ignore_target=255):
#          super(OhemCELoss, self).__init__()
#          self.score_thresh = thresh
#          self.ignore_target = ignore_target
#          self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_target, reduction='mean')
#
#      def forward(self, logits, targets):
#          n_min = targets[targets != self.ignore_target].numel() // 16
#          targets = ohem_cpp.score_ohem_label(
#                  logits, targets, self.ignore_target, self.score_thresh, n_min).detach()
#          loss = self.criteria(logits, targets)
#          return loss


class OhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_target=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_target = ignore_target
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_target, reduction='none')

    def forward(self, logits, targets):
        #FIXME: learn how to use Online Hard Example Mining
        # n_min = targets[targets != self.ignore_target].numel() // 16
        n_min = targets[targets != self.ignore_target].numel()
        loss = self.criteria(logits, targets).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


if __name__ == '__main__':
    pass

