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

    def _op(self, name, data, inputs, vjp_fn):
        out = Tensor(data, requires_grad=any(t.requires_grad for t in inputs))
        def _backward():
            grads = vjp_fn(out.grad)
            for t, g in zip(inputs, grads):
                if t.requires_grad and g is not None:
                    t.grad += g
        out._prevs = set(inputs)
        out._backward = _backward
        out._name = name
        return out

    def __add__(self, other):
        other = ensure_tensor(other)
        return self._op("add", self.data + other.data, [self, other],
            lambda g: (self._unbroadcast(g), other._unbroadcast(g)))

    def __sub__(self, other):
        other = ensure_tensor(other)
        return self._op("sub", self.data - other.data, [self, other],
            lambda g: (self._unbroadcast(g), other._unbroadcast(-g)))

    def __mul__(self, other):
        other = ensure_tensor(other)
        return self._op("mul", self.data * other.data, [self, other],
            lambda g: (self._unbroadcast(g * other.data),
                       other._unbroadcast(g * self.data)))

    def __matmul__(self, other):
        other = ensure_tensor(other)
        return self._op("matmul", self.data @ other.data, [self, other],
            lambda g: (g @ other.data.T, self.data.T @ g))

    def __truediv__(self, other):
        other = ensure_tensor(other)
        return self._op("div", self.data / other.data, [self, other],
            lambda g: (self._unbroadcast(g / other.data),
                       other._unbroadcast(-g * self.data / np.square(other.data))))

    def square(self):
        return self._op("square", np.square(self.data), [self],
            lambda g: (g * self.data * 2,))

    def log(self):
        return self._op("log", np.log(self.data), [self],
            lambda g: (g / self.data,))

    def exp(self):
        data = np.exp(self.data)
        return self._op("exp", data, [self], lambda g: (g * data,))

    def _softmax(self):
        e_x = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def softmax(self):
        s = self._softmax()
        return self._op("softmax", s, [self],
            lambda g: (s * (g - np.sum(g * s, axis=-1, keepdims=True)),))

    def sparse_softmax_cross_entropy(self, target):
        target = ensure_tensor(target)
        s = self._softmax()
        n = np.arange(len(target))
        def vjp(g):
            grad = s.copy()
            grad[n, target.data] -= 1
            grad *= g / len(target)
            return (grad,)
        return self._op("sparse_softmax_cross_entropy",
            -np.log(s[n, target.data]).mean(), [self], vjp)

    def relu(self):
        return self._op("relu", np.maximum(self.data, 0), [self],
            lambda g: (g * (self.data > 0).astype(np.float32),))

    def sum(self):
        return self._op("sum", np.sum(self.data), [self], lambda g: (g,))

    def mean(self, axis):
        return self._op("mean", self.data.mean(axis=axis), [self],
            lambda g: (np.expand_dims(g, axis=axis) / self.data.shape[axis],))

    def _extremum(self, name, axis, fn):
        def vjp(g):
            mask = (self.data == fn(self.data, axis=axis, keepdims=True)).astype(np.float32)
            return (mask * (np.expand_dims(g, axis=axis) / mask.sum(axis=axis, keepdims=True)),)
        return self._op(name, fn(self.data, axis=axis), [self], vjp)

    def max(self, axis):
        return self._extremum("max", axis, np.max)

    def min(self, axis):
        return self._extremum("min", axis, np.min)

    def argmax(self, axis=0):
        return Tensor(np.argmax(self.data, axis=axis))

    def argmin(self, axis=0):
        return Tensor(np.argmin(self.data, axis=axis))

    def reshape(self, shape):
        return self._op("reshape", self.data.reshape(shape), [self],
            lambda g: (g.reshape(self.shape),))

    def im2col(self, kernel_size, stride):
        x = self.data
        kh, kw = kernel_size
        sh, sw = stride
        b, h, w, c = x.shape
        out_h, out_w = (h - kh) // sh + 1, (w - kw) // sw + 1
        y = np.empty((b, out_h, out_w, kh * kw * c), dtype=x.dtype)
        for i in range(out_h):
            for j in range(out_w):
                y[:, i, j, :] = x[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :].reshape(b, -1)
        def vjp(g):
            grad = np.zeros_like(x)
            out_grad = g.reshape(b, out_h, out_w, kh, kw, c)
            for i in range(out_h):
                for j in range(out_w):
                    grad[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :] += out_grad[:, i, j]
            return (grad,)
        return self._op("im2col", y.reshape(-1, kh * kw * c), [self], vjp)

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


def ensure_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)
