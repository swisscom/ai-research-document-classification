"""
Copyright © 2018-present, Swisscom (Schweiz) AG.
All rights reserved.

Author: Khalil Mrini
"""

import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Custom CrossEntropyLoss for the case of target replication.

    Parameters
    ----------
    weight (Tensor, optional): a manual rescaling weight given to each class.
           If given, has to be a Tensor of size `C`
    size_average (bool, optional): By default, the losses are averaged over observations for each minibatch.
           However, if the field `size_average` is set to ``False``, the losses are
           instead summed for each minibatch. Ignored if reduce is ``False``.
    ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When `size_average` is
            ``True``, the loss is averaged over non-ignored targets.
    reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on `size_average`. When reduce
            is ``False``, returns a loss per batch instead and ignores
            size_average. Default: ``True``
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(CustomCrossEntropyLoss, self).__init__(weight, size_average, reduce)
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        losses = []
        index = -1
        for input, child_inputs in inputs:
            index += 1
            input_loss = F.cross_entropy(input, targets[index], self.weight, self.size_average,
                                   self.ignore_index, self.reduce)
            child_loss = sum([F.cross_entropy(child_input, targets[index], self.weight, self.size_average,
                                   self.ignore_index, self.reduce) for child_input in child_inputs])
            losses.append(input_loss + child_loss * (1.0/len(child_inputs)))
        return sum(losses)

class CustomBCELoss(nn.BCELoss):
    """
    Custom BCELoss for the case of target replication.

    Parameters
    ----------
    weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
    size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. However, if the field
            size_average is set to ``False``, the losses are instead summed for
            each minibatch. Default: ``True``
    reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is False, returns a loss per input/target element instead and ignores
            size_average. Default: True
    """

    def __init__(self, weight=None, size_average=True, reduce=True):
        super(CustomBCELoss, self).__init__(weight, size_average, reduce)

    def forward(self, inputs, targets):
        losses = []
        index = -1
        for input, child_inputs in inputs:
            index += 1
            input_loss = F.binary_cross_entropy(input, targets[index], self.weight, self.size_average,
                                                self.ignore_index, self.reduce)
            child_loss = sum([F.binary_cross_entropy(child_input, targets[index], self.weight, self.size_average,
                                                     self.reduce) for child_input in child_inputs])
            losses.append(input_loss + child_loss * (1.0/len(child_inputs)))
        return sum(losses)