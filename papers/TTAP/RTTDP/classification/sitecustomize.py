import importlib

import torch


def zero_gradients(inputs):
    if isinstance(inputs, torch.Tensor):
        if inputs.grad is not None:
            inputs.grad.detach_()
            inputs.grad.zero_()
    elif isinstance(inputs, (list, tuple)):
        for item in inputs:
            zero_gradients(item)


gradcheck = importlib.import_module("torch.autograd.gradcheck")
if not hasattr(gradcheck, "zero_gradients"):
    gradcheck.zero_gradients = zero_gradients
