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
        self.gradients = {}
        upstream = self.outputs[0].gradients[self]
        self.gradients[A] = np.dot(x.value.T, upstream)
        self.gradients[x] = np.dot(upstream, A.value.T)
        self.gradients[B] = np.sum(upstream, axis=0, keepdims=True)

class Sigmoid(Node):
    def __init__(self, node):
        super().__init__([node])
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    def forward(self):
        z = self.inputs[0].value
        self.value = self._sigmoid(z)
    def backward(self):
        self.gradients = {}
        upstream = self.outputs[0].gradients[self]
        local_grad = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = local_grad * upstream

class ReLU(Node):
    def __init__(self, node):
        super().__init__([node])
    def forward(self):
        self.value = np.maximum(0, self.inputs[0].value)
    def backward(self):
        self.gradients = {}
        upstream = self.outputs[0].gradients[self]
        mask = (self.inputs[0].value > 0).astype(float)
        self.gradients[self.inputs[0]] = upstream * mask

class Softmax(Node):
    def __init__(self, node):
        super().__init__([node])
    def forward(self):
        z = self.inputs[0].value
        exp_vals = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.value = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    def backward(self):
        self.gradients = {}
        S = self.value
        upstream = self.outputs[0].gradients[self]
        dot = np.sum(upstream * S, axis=1, keepdims=True)
        self.gradients[self.inputs[0]] = S * (upstream - dot)

class BCE(Node):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])
    def forward(self):
        y_true, y_pred = self.inputs
        eps = 1e-12
        y_pred_safe = np.clip(y_pred.value, eps, 1 - eps)
        batch_size = y_true.value.shape[0]
        self.value = np.sum(-y_true.value * np.log(y_pred_safe) -
                            (1 - y_true.value) * np.log(1 - y_pred_safe)) / batch_size
    def backward(self):
        self.gradients = {}
        y_true, y_pred = self.inputs
        eps = 1e-12
        y_pred_safe = np.clip(y_pred.value, eps, 1 - eps)
        batch_size = y_true.value.shape[0]
        self.gradients[y_pred] = (y_pred_safe - y_true.value) / (y_pred_safe * (1 - y_pred_safe)) / batch_size
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
        self.gradients[y_pred] = -(y_true.value / y_pred_safe) / y_true.value.shape[0]
        self.gradients[y_true] = np.zeros_like(y_true.value)
