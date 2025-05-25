from torch.autograd import Function

import torch


class MyAutoGrad(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return torch.exp(x) + torch.cos(y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * torch.exp(x), grad_output * (-torch.sin(y))
