import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import numpy as np

from pathlib import Path

from spikingjelly.datasets import n_mnist
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from tqdm import tqdm
from spikingjelly.activation_based.surrogate import SurrogateFunctionBase

@torch.jit.script
def power_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    
    if alpha<=0 or alpha>=1:
        raise ValueError("The value of alpha should in the range (0,1), {} is illegal!".format(alpha))
    # a True-of-False tensor signing whether x>a
    x_larger_than_1 = (x > 1)

    # a True-of-False tensor signing whether x<-a
    x_smaller_than_minus_1 = (x < -1)

    # ! ((x>1) or (x<-1))
    x_in_range = torch.bitwise_not(torch.bitwise_or(x_larger_than_1, x_smaller_than_minus_1))

    grad = x_in_range * (alpha / 2) * torch.pow(torch.abs(x), alpha)

    return grad_output * grad, None

class power_surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x >= 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return power_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class PowerSurrogate(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        super().__init__(alpha, spiking)
    @staticmethod
    def spiking_function(x, alpha):
        return power_surrogate.apply(x, alpha)
    @staticmethod
    def backward(grad_output, x, alpha):
        return power_backward(grad_output, x, alpha)[0]