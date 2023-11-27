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
def trigono_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    # a True-of-False tensor signing whether x>a
    x_larger_than_a = (x > alpha)

    # a True-of-False tensor signing whether x<-a
    x_smaller_than_minus_a = (x < -alpha)

    # ! ((x>a) or (x<-a))
    x_in_range = torch.bitwise_not(torch.bitwise_or(x_larger_than_a, x_smaller_than_minus_a))

    grad = x_in_range * (torch.pi / 4 / alpha) * torch.cos(torch.pi * x / 2 / alpha)

    return grad_output * grad, None

class trigono_surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x >= 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return trigono_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class TrigonoSurrogate(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        super().__init__(alpha, spiking)
    @staticmethod
    def spiking_function(x, alpha):
        return trigono_surrogate.apply(x, alpha)
    @staticmethod
    def backward(grad_output, x, alpha):
        return trigono_backward(grad_output, x, alpha)[0]