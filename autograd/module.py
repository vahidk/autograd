import pickle
from typing import Sequence

import numpy as np

from .init import kaiming
from .tensor import Tensor


class Module:
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Must implement forward method.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        for _, tensor in self.named_tensors():
            if tensor.requires_grad:
                yield tensor

    def named_tensors(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                for k, p in attr.named_tensors():
                    yield f"{attr_name}.{k}", p
            elif isinstance(attr, Tensor):
                yield attr_name, attr

    def state_dict(self) -> dict:
        state = {}
        for name, tensor in self.named_tensors():
            state[name] = tensor.data
        return state

    def load_state_dict(self, state: dict):
        for name, tensor in self.named_tensors():
            if name not in state:
                raise KeyError(f"Missing key {name} in state dict.")
            if tensor.shape != state[name].shape:
                raise ValueError(f"Shape mismatch for {name}: {tensor.shape} vs {state[name].shape}")
            tensor.data = state[name]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.load_state_dict(pickle.load(f))


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for module in modules:
            if not isinstance(module, Module):
                raise ValueError(f"Expected a Module, got {type(module)}")
        self.modules = modules

    def forward(self, *args, **kwargs):
        x = args
        for module in self.modules:
            if isinstance(x, tuple):
                x = module(*x)
            else:
                x = module(x)
        return x

    def named_tensors(self):
        for i, module in enumerate(self.modules):
            for name, tensor in module.named_tensors():
                yield f"{i}.{name}", tensor


class Dense(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = Tensor(kaiming([in_channels, out_channels]), requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_channels)), requires_grad=True)

    def forward(self, x):
        return x @ self.weight + self.bias


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != 2:
                raise ValueError("kernel_size should be a tuple of (height, width)")
        else:
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, Sequence):
            if len(stride) != 2:
                raise ValueError("stride should be a tuple of (height, width)")
        else:
            stride = (stride, stride)
        self.weight = Tensor(kaiming([*kernel_size, in_channels, out_channels]), requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_channels)), requires_grad=True)
        self.stride = stride

    def forward(self, x):
        b, h, w, c = x.shape
        kh, kw, cin, cout = self.weight.shape
        sh, sw = self.stride
        if c != cin:
            raise ValueError(f"Input channels {c} do not match weight channels {cin}.")
        y = x.im2col((kh, kw), (sh, sw)) @ self.weight.reshape((kh * kw * cin, cout)) + self.bias
        return y.reshape((b, (h - kh) // sh + 1, (w - kw) // sw + 1, cout))


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.relu()


class AvgPool2D(Module):
    def __init__(self, kernel_size):
        super().__init__()
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != 2:
                raise ValueError("kernel_size should be a tuple of (height, width)")
        else:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x):
        b, h, w, c = x.shape
        kh, kw = self.kernel_size
        return x.im2col((kh, kw), (kh, kw)).reshape((-1, kh * kw, c)).mean(axis=1).reshape((b, h // kh, w // kw, c))


class MaxPool2D(Module):
    def __init__(self, kernel_size):
        super().__init__()
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != 2:
                raise ValueError("kernel_size should be a tuple of (height, width)")
        else:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x):
        b, h, w, c = x.shape
        kh, kw = self.kernel_size
        return x.im2col((kh, kw), (kh, kw)).reshape((-1, kh * kw, c)).max(axis=1).reshape((b, h // kh, w // kw, c))


class Reshape(Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)
