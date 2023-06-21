# This file contains code for different losses.
from typing import Any, Mapping, Text
import torch
import torch.nn.functional as F

def create_loss(loss_config: Mapping[Text, Any]) -> torch.nn.Module:
    """Creates a loss object based on the config.
    Args:
        loss_config (Mapping[Text, Any]): A mapping that specifies the loss.
    Returns:
        torch.nn.Module: The loss object.
    """
    loss_name = loss_config['TYPE']
    if loss_name == 'cross-entropy':
        return CrossEntropy(ignore_index=loss_config['IGNORE_INDEX'])
    elif loss_name == 'focal-loss':
        return FocalLoss(ignore_index=loss_config['IGNORE_INDEX'])
    else:
        raise ValueError(f'The specified loss {loss_name} is currently not supported.')


class CrossEntropy(torch.nn.Module):
    def __init__(self, ignore_index=-100):
        super(CrossEntropy, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        loss = F.cross_entropy(input, target, weight=None,
                               ignore_index=self.ignore_index, reduction='none')
        return torch.sum(loss)/torch.count_nonzero(target != self.ignore_index)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        loss = F.cross_entropy(input, target, weight=None,
                               ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * loss)
        return torch.sum(focal_loss)/torch.count_nonzero(target != self.ignore_index)
