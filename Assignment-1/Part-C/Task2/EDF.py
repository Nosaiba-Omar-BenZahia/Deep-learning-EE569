import numpy as np
from math import sqrt

class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}
        for node in inputs:
            node.outputs.append(self)
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
    def __init__(self, x, A, B):
        super().__init__([x, A, B])
    def forward(self):
        x, A, B = self.inputs
        b_val = B.value.reshape(1, -1)
        self.value = np.dot(x.value, A.value) + b_val
    def backward(self):
        x, A, B = self.inputs
        upstream = self.outputs[0].gradients[self]
        self.gradients = {}
        self.gradients[A] = np.dot(x.value.T, upstream)
        self.gradients[x] = np.dot(upstream, A.value.T)
        self.gradients[B] = np.sum(upstream, axis=0, keepdims=True)

class Sigmoid(Node):
    def __init__(self, node):
        super().__init__([node])
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    def forward(self):
        self.value = self._sigmoid(self.inputs[0].value)
    def backward(self):
        upstream = self.outputs[0].gradients[self]
        self.gradients = {}
        self.gradients[self.inputs[0]] = upstream * self.value * (1 - self.value)

class ReLU(Node):
    def __init__(self, node):
        super().__init__([node])
    def forward(self):
        self.value = np.maximum(0, self.inputs[0].value)
    def backward(self):
        upstream = self.outputs[0].gradients[self]
        mask = (self.inputs[0].value > 0).astype(float)
        self.gradients = {}
        self.gradients[self.inputs[0]] = upstream * mask

class Softmax(Node):
    def __init__(self, node):
        super().__init__([node])
    def forward(self):
        z = self.inputs[0].value
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.value = exps / np.sum(exps, axis=1, keepdims=True)
    def backward(self):
        S = self.value
        upstream = self.outputs[0].gradients[self]
        dot = np.sum(upstream * S, axis=1, keepdims=True)
        self.gradients = {}
        self.gradients[self.inputs[0]] = S * (upstream - dot)

class BCE(Node):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])
    def forward(self):
        y_true, y_pred = self.inputs
        eps = 1e-12
        y_pred_safe = np.clip(y_pred.value, eps, 1 - eps)
        self.value = np.mean(
            -y_true.value * np.log(y_pred_safe) -
            (1 - y_true.value) * np.log(1 - y_pred_safe)
        )
    def backward(self):
        y_true, y_pred = self.inputs
        eps = 1e-12
        y_pred_safe = np.clip(y_pred.value, eps, 1 - eps)
        batch_size = y_true.value.shape[0]
        self.gradients = {}
        self.gradients[y_pred] = (y_pred_safe - y_true.value) / (
            y_pred_safe * (1 - y_pred_safe)
        ) / batch_size
        self.gradients[y_true] = np.zeros_like(y_true.value)

class CE(Node):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])
    def forward(self):
        y_true, y_pred = self.inputs
        eps = 1e-15
        y_pred_safe = np.clip(y_pred.value, eps, 1 - eps)
        self.value = -np.mean(np.sum(y_true.value * np.log(y_pred_safe), axis=1))
    def backward(self):
        y_true, y_pred = self.inputs
        eps = 1e-15
        y_pred_safe = np.clip(y_pred.value, eps, 1 - eps)
        self.gradients = {}
        self.gradients[y_pred] = -(y_true.value / y_pred_safe) / y_true.value.shape[0]
        self.gradients[y_true] = np.zeros_like(y_true.value)

class Conv(Node):
    def __init__(self, x, input_channels, output_channels, kernel_height, kernel_width,
                 kernel=None, bias=None, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        if kernel is not None:
            W = Parameter(kernel)
        else:
            W = Parameter(np.random.randn(output_channels, input_channels, kernel_height, kernel_width))
        if bias is not None:
            b = Parameter(bias)
        else:
            b = Parameter(np.zeros((output_channels,)))
        super().__init__([x, W, b])
    def forward(self):
        x, W, b = self.inputs
        N, Cin, H, W_in = x.value.shape
        Cout, _, Kh, Kw = W.value.shape
        if self.padding > 0:
            pad = self.padding
            padded = np.pad(x.value, ((0,0),(0,0),(pad,pad),(pad,pad)), mode="constant")
        else:
            padded = x.value
        H_out = (padded.shape[2] - Kh) // self.stride + 1
        W_out = (padded.shape[3] - Kw) // self.stride + 1
        out = np.zeros((N, Cout, H_out, W_out), dtype=float)
        for i in range(H_out):
            for j in range(W_out):
                patch = padded[:, :, i*self.stride:i*self.stride+Kh,
                                     j*self.stride:j*self.stride+Kw]
                out[:,:,i,j] = np.tensordot(patch, W.value, axes=([1,2,3],[1,2,3])) + b.value
        self.value = out
    def backward(self):
        pass

class MaxPooling(Node):
    def __init__(self, x, pool_height=2, pool_width=2):
        self.pool_height = pool_height
        self.pool_width = pool_width
        super().__init__([x])
    def forward(self):
        x = self.inputs[0].value
        N, C, H, W = x.shape
        ph, pw = self.pool_height, self.pool_width
        H_out = H // ph
        W_out = W // pw
        out = np.zeros((N, C, H_out, W_out), dtype=float)
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, :, i*ph:i*ph+ph, j*pw:j*pw+pw]
                out[:,:,i,j] = np.max(patch, axis=(2,3))
        self.value = out
    def backward(self):
        pass
