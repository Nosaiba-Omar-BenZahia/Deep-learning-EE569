import numpy as np

# Base Node class
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


# Input Node
class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

class Linear(Node):
    def __init__(self, x, A, B):
        Node.__init__(self, [x, A, B])

    def forward(self):
        x = self.inputs[0]
        A = self.inputs[1]
        B = self.inputs[2]
        y = np.dot(x.value, A.value)
        y += B.value
        self.value = y

    def backward(self):
        x = self.inputs[0]
        A = self.inputs[1]
        B = self.inputs[2]
        GRAD = self.outputs[0].gradients[self]

        dA = np.dot(x.value.T, GRAD)
        dx = np.dot(GRAD, A.value.T)
        dB = np.sum(GRAD, axis=0, keepdims=True)

        self.gradients[A] = dA
        self.gradients[x] = dx
        self.gradients[B] = dB

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]

class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = np.sum(-y_true.value*np.log(y_pred.value)-(1-y_true.value)*np.log(1-y_pred.value))

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (1 / y_true.value.shape[0]) * (y_pred.value - y_true.value)/(y_pred.value*(1-y_pred.value))
        self.gradients[y_true] = (1 / y_true.value.shape[0]) * (np.log(y_pred.value) - np.log(1-y_pred.value))
