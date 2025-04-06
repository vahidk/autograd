from typing import Sequence

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: Sequence[Tensor]):
        self.params = list(params)

    def step(self):
        raise NotImplementedError("Optimizer step method must be implemented in subclasses.")
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0)


class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float = 0.01, momentum: float = 0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [0] * len(self.params)

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * param.grad
            param.data += self.velocities[i]
