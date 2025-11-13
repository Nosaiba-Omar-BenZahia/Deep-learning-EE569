import numpy as np
from EDF import *

def build_cnn():
    x = Input()

    conv1 = Conv(x, 3, 16, 3, 3, stride=1, padding=1)
    relu1 = ReLU(conv1)
    pool1 = MaxPooling(relu1, 2, 2)

    conv2 = Conv(pool1, 16, 32, 3, 3, stride=1, padding=1)
    relu2 = ReLU(conv2)
    pool2 = MaxPooling(relu2, 2, 2)

    conv3 = Conv(pool2, 32, 64, 3, 3, stride=1, padding=1)
    relu3 = ReLU(conv3)
    pool3 = MaxPooling(relu3, 2, 2)

    conv4 = Conv(pool3, 64, 128, 3, 3, stride=1, padding=1)
    relu4 = ReLU(conv4)

    flat = Flatten(relu4)

    flatten_dim = 128 * (32 // 8) * (32 // 8)

    linear = Linear(flat, flatten_dim, 10)
    softmax = Softmax(linear)

    y = Input()
    loss = CE(y, softmax)

    return x, y, loss, softmax



def gather_graph(node):
    visited = set()
    order = []

    def dfs(n):
        if n not in visited:
            visited.add(n)
            for i in n.inputs:
                dfs(i)
            order.append(n)

    dfs(node)
    return order


def gather_params(graph):
    return [n for n in graph if isinstance(n, Parameter)]


def forward(graph):
    for n in graph:
        n.forward()


def backward(graph):
    for n in reversed(graph):
        n.backward()


def numerical_grad(f, w, eps=1e-4):
    g = np.zeros_like(w)
    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old = w[idx]

        w[idx] = old + eps
        p1 = f(w)

        w[idx] = old - eps
        p2 = f(w)

        g[idx] = (p1 - p2) / (2 * eps)
        w[idx] = old
        it.iternext()

    return g


def test_cnn():
    x_node, y_node, loss_node, softmax_node = build_cnn()
    X = np.random.randn(1, 3, 32, 32)
    Y = np.zeros((1, 10))
    Y[0, np.random.randint(0, 10)] = 1

    x_node.value = X
    y_node.value = Y

    graph = gather_graph(loss_node)
    params = gather_params(graph)

    forward(graph)
    print("Loss:", loss_node.value)

    backward(graph)

    target_W = None
    for p in params:
        if p.value.ndim == 4:
            target_W = p
            break

    def f(w):
        target_W.value = w
        forward(graph)
        return loss_node.value

    numeric = numerical_grad(f, target_W.value.copy())
    analytic = target_W.gradients[target_W]
    diff = np.linalg.norm(numeric - analytic)

    print("Performing gradient check...")
    print("Difference:", diff)


test_cnn()
