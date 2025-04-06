import numpy as np


class Tensor:
    def __init__(self, data, dtype=np.float32, requires_grad=False):
        self.data = np.asarray(data, dtype=dtype)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._prevs = set()
        self._backward = lambda: None
        self._name = ""

    @property
    def requires_grad(self):
        return self.grad is not None

    @property
    def shape(self):
        return self.data.shape

    def __str__(self):
        return str(self.data)
    
    def __len__(self):
        return len(self.data)

    def _unbroadcast(self, grad):
        padded_target = (1,) * (grad.ndim - len(self.data.shape)) + tuple(self.data.shape)
        reduce_axes = [
            i for i, (g_dim, t_dim) in enumerate(zip(grad.shape, padded_target))
            if t_dim == 1 and g_dim != 1
        ]
        if reduce_axes:
            grad = grad.sum(axis=tuple(reduce_axes), keepdims=True)
        return grad.reshape(self.data.shape)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad)
            if other.requires_grad:
                other.grad += other._unbroadcast(out.grad)

        out._prevs = {self, other}
        out._backward = _backward
        out._name = "add"
        return out

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad)
            if other.requires_grad:
                other.grad -= self._unbroadcast(out.grad)

        out._prevs = {self, other}
        out._backward = _backward
        out._name = "subtract"
        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad * other.data)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad * self.data)

        out._prevs = {self, other}
        out._backward = _backward
        out._name = "multiply"
        return out

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._prevs = {self, other}
        out._backward = _backward
        out._name = "matmul"
        return out

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad / other.data)
            if other.requires_grad:
                other.grad += self._unbroadcast(-out.grad * self.data / np.square(other.data))

        out._prevs = {self, other}
        out._backward = _backward
        out._name = "div"
        return out

    def square(self):
        out = Tensor(np.square(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * self.data * 2

        out._prevs = {self}
        out._backward = _backward
        out._name = "square"
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._prevs = {self}
        out._backward = _backward
        out._name = "log"
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * out.data
        
        out._prevs = {self}
        out._backward = _backward
        out._name = "exp"
        return out

    def softmax(self):
        e_x = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out = Tensor(e_x / np.sum(e_x, axis=-1, keepdims=True), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                s = out.data
                self.grad += out.grad * s * (1 - s)
                for i in range(s.shape[-1]):
                    for j in range(s.shape[-1]):
                        if i != j:
                            self.grad[..., i] += out.grad[..., j] * (-s[..., i] * s[..., j])
        
        out._prevs = {self}
        out._backward = _backward
        out._name = "softmax"
        return out

    def sparse_softmax_cross_entropy(self, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)

        e_x = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        s = e_x / np.sum(e_x, axis=-1, keepdims=True)

        loss = -np.log(s[np.arange(len(target)), target.data])
        out = Tensor(loss.mean(), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = s.copy()
                grad[np.arange(len(target)), target.data] -= 1
                grad /= len(target)
                self.grad += grad

        out._prevs = {self}
        out._backward = _backward
        out._name = "sparse_softmax_cross_entropy"
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0).astype(np.float32)

        out._prevs = {self}
        out._backward = _backward
        out._name = "relu"
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad

        out._prevs = {self}
        out._backward = _backward
        out._name = "sum"
        return out

    def mean(self, axis):
        out = Tensor(self.data.mean(axis=axis), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * (np.expand_dims(out.grad, axis=axis) / self.data.shape[axis])
        
        out._prevs = {self}
        out._backward = _backward
        out._name = "mean"
        return out

    def max(self, axis):
        out = Tensor(self.data.max(axis=axis, keepdims=False), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                mask = (self.data == np.max(self.data, axis=axis, keepdims=True)).astype(np.float32)
                self.grad += mask * (np.expand_dims(out.grad, axis=axis) / mask.sum(axis=axis, keepdims=True))
        
        out._prevs = {self}
        out._backward = _backward
        out._name = "max"
        return out
    
    def min(self, axis):
        out = Tensor(self.data.max(axis=axis, keepdims=False), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                mask = (self.data == np.min(self.data, axis=axis, keepdims=True)).astype(np.float32)
                self.grad += mask * (np.expand_dims(out.grad, axis=axis) / mask.sum(axis=axis, keepdims=True))
        
        out._prevs = {self}
        out._backward = _backward
        out._name = "min"
        return out
    
    def argmax(self, axis=0):
        out = Tensor(np.argmax(self.data, axis=axis), requires_grad=False)
        out._prevs = {self}
        out._name = "argmax"
        return out
    
    def argmin(self, axis=0):
        out = Tensor(np.argmin(self.data, axis=axis), requires_grad=False)
        out._prevs = {self}
        out._name = "argmin"
        return out

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)

        out._prevs = {self}
        out._backward = _backward
        out._name = "reshape"
        return out

    def im2col(self, kernel_size, stride):
        x = self.data
        kh, kw = kernel_size
        sh, sw = stride
        b, h, w, c = x.shape
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1

        y = np.empty((b, out_h, out_w, kh * kw * c), dtype=self.data.dtype)
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                y[:, i, j, :] = patch.reshape(b, kh * kw * c)
        out = Tensor(y.reshape(-1, kh * kw * c), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                out_grad = out.grad.reshape(b, out_h, out_w, kh * kw * c)
                for i in range(out_h):
                    for j in range(out_w):
                        grad_patch = out_grad[:, i, j, :].reshape(b, kh, kw, c)
                        self.grad[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :] += grad_patch

        out._prevs = {self}
        out._backward = _backward
        out._name = "im2col"
        return out

    def backward(self):
        if not self.requires_grad:
            raise ValueError("The tensor doesn't have gradients.")

        ordered = []
        visited = set()
        nexts = set()

        def _sort(node):
            if node in nexts:
                raise ValueError("There's a cyclic dependency.")
            if node in visited:
                return
            nexts.add(node)
            for prev in node._prevs:
                _sort(prev)
            nexts.remove(node)
            visited.add(node)
            ordered.append(node)

        _sort(self)

        self.grad = np.ones_like(self.data)
        for tensor in reversed(ordered):
            tensor._backward()
