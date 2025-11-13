import numpy as np
from math import sqrt

class Node:
    def __init__(self, inputs=None):
        self.inputs = inputs if inputs else []
        self.outputs = []
        self.value = None
        self.gradients = {}
        for n in self.inputs:
            n.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Input(Node):
    def __init__(self):
        super().__init__()

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


class Parameter(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


class Linear(Node):
    def __init__(self, x, input_dim, output_dim):
        W = Parameter(np.random.randn(input_dim, output_dim) * 0.01)
        b = Parameter(np.zeros((1, output_dim)))
        super().__init__([x, W, b])

    def forward(self):
        x, W, b = self.inputs
        self.value = np.dot(x.value, W.value) + b.value

    def backward(self):
        x, W, b = self.inputs
        grad = self.outputs[0].gradients[self]
        self.gradients[x] = np.dot(grad, W.value.T)
        self.gradients[W] = np.dot(x.value.T, grad)
        self.gradients[b] = np.sum(grad, axis=0, keepdims=True)


class Conv(Node):
    def __init__(self, x, Cin, Cout, Kh, Kw, kernel=None, bias=None, stride=1, padding=1):
        self.stride = stride
        self.padding = padding

        if kernel is None:
            limit = sqrt(6 / (Cin * Kh * Kw + Cout))
            kernel = np.random.uniform(-limit, limit, (Cout, Cin, Kh, Kw))

        if bias is None:
            bias = np.zeros((Cout,), dtype=np.float32)

        W = Parameter(kernel)
        b = Parameter(bias)
        super().__init__([x, W, b])

    def forward(self):
        x, W, b = self.inputs
        N, C, H, W_in = x.value.shape
        Cout, Cin, Kh, Kw = W.value.shape

        pad = self.padding
        if pad > 0:
            x_pad = np.pad(x.value, ((0,0),(0,0),(pad,pad),(pad,pad)))
        else:
            x_pad = x.value

        H_out = (x_pad.shape[2] - Kh) // self.stride + 1
        W_out = (x_pad.shape[3] - Kw) // self.stride + 1

        out = np.zeros((N, Cout, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                h0 = i * self.stride
                w0 = j * self.stride
                patch = x_pad[:, :, h0:h0+Kh, w0:w0+Kw]
                out[:, :, i, j] = np.tensordot(patch, W.value, axes=([1,2,3],[1,2,3])) + b.value

        self.x_pad = x_pad
        self.value = out

    def backward(self):
        x, W, b = self.inputs
        N, C, H, W_in = x.value.shape
        Cout, Cin, Kh, Kw = W.value.shape
        grad = self.outputs[0].gradients[self]

        pad = self.padding
        x_pad = self.x_pad
        dx_pad = np.zeros_like(x_pad)

        self.gradients[W] = np.zeros_like(W.value)
        self.gradients[b] = np.zeros_like(b.value)

        H_out = grad.shape[2]
        W_out = grad.shape[3]

        for i in range(H_out):
            for j in range(W_out):
                h0 = i * self.stride
                w0 = j * self.stride

                patch = x_pad[:, :, h0:h0+Kh, w0:w0+Kw]

                self.gradients[W] += np.tensordot(grad[:, :, i, j], patch, axes=([0],[0]))
                self.gradients[b] += np.sum(grad[:, :, i, j], axis=0)
                dx_pad[:, :, h0:h0+Kh, w0:w0+Kw] += np.tensordot(grad[:, :, i, j], W.value, axes=([1],[0]))

        if pad > 0:
            self.gradients[x] = dx_pad[:, :, pad:-pad, pad:-pad]
        else:
            self.gradients[x] = dx_pad


class MaxPooling(Node):
    def __init__(self, x, ph, pw):
        self.ph = ph
        self.pw = pw
        super().__init__([x])

    def forward(self):
        x = self.inputs[0].value
        N, C, H, W = x.shape
        ph, pw = self.ph, self.pw
        H_out = H // ph
        W_out = W // pw

        out = np.zeros((N, C, H_out, W_out))
        self.max_idx = np.zeros((N, C, H_out, W_out, 2), dtype=np.int32)

        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                flat = patch.reshape(N, C, -1)
                idx = flat.argmax(axis=2)

                out[:, :, i, j] = flat.max(axis=2)
                self.max_idx[:, :, i, j, 0] = idx // pw + i*ph
                self.max_idx[:, :, i, j, 1] = idx % pw + j*pw

        self.value = out

    def backward(self):
        x = self.inputs[0].value
        grad = self.outputs[0].gradients[self]
        N, C, H, W = x.shape

        dx = np.zeros_like(x)
        idx = self.max_idx

        for i in range(grad.shape[2]):
            for j in range(grad.shape[3]):
                dx[np.arange(N)[:,None], np.arange(C)[None,:], idx[:, :, i, j, 0], idx[:, :, i, j, 1]] += grad[:, :, i, j]

        self.gradients[self.inputs[0]] = dx


class Flatten(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        x = self.inputs[0].value
        self.shape = x.shape
        self.value = x.reshape(x.shape[0], -1)

    def backward(self):
        grad = self.outputs[0].gradients[self]
        self.gradients[self.inputs[0]] = grad.reshape(self.shape)


class ReLU(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.value = np.maximum(0, self.inputs[0].value)

    def backward(self):
        grad = self.outputs[0].gradients[self]
        mask = (self.inputs[0].value > 0).astype(float)
        self.gradients[self.inputs[0]] = grad * mask


class Softmax(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        z = self.inputs[0].value
        expz = np.exp(z - z.max(axis=1, keepdims=True))
        self.value = expz / expz.sum(axis=1, keepdims=True)

    def backward(self):
        S = self.value
        grad = self.outputs[0].gradients[self]
        dot = np.sum(grad * S, axis=1, keepdims=True)
        self.gradients[self.inputs[0]] = S * (grad - dot)


class CE(Node):
    def __init__(self, y, y_pred):
        super().__init__([y, y_pred])

    def forward(self):
        y, p = self.inputs
        self.value = -np.sum(y.value * np.log(p.value + 1e-15)) / y.value.shape[0]

    def backward(self):
        y, p = self.inputs
        self.gradients[p] = -(y.value / (p.value + 1e-15)) / y.value.shape[0]
        self.gradients[y] = np.zeros_like(y.value)
