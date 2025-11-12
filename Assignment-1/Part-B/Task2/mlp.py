from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy
from math import sqrt

LR = 0.01
EPOCHS = 3000
BATCH_SIZE = 32
TEST_SIZE = 0.25
ACTIVATION = "sigmoid"
def generate_xor_data(n_per_cluster=50):
    cov = np.array([[1, 0], [0, 1]])
    X0_1 = multivariate_normal.rvs(mean=[-3, 3], cov=cov, size=n_per_cluster)
    X0_2 = multivariate_normal.rvs(mean=[3, -3], cov=cov, size=n_per_cluster)
    X1_1 = multivariate_normal.rvs(mean=[3, 3], cov=cov, size=n_per_cluster)
    X1_2 = multivariate_normal.rvs(mean=[-3, -3], cov=cov, size=n_per_cluster)
    X0 = np.vstack((X0_1, X0_2))
    X1 = np.vstack((X1_1, X1_2))
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(X0.shape[0]), np.ones(X1.shape[0])))
    return X, y

X, y = generate_xor_data()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("XOR Dataset for MLP")
plt.show()


indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]


def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in reversed(graph):
        n.backward()

def sgd_update(params, lr):
    for p in params:
        p.value -= lr * p.gradients[p]

def get_batches(X, y, batch_size):
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        sel = idx[i:j]
        yield X[sel], y[sel].reshape(-1, 1)


def make_layer(input_node, input_dim, output_dim, activation="sigmoid"):
    limit = sqrt(6 / (input_dim + output_dim))
    W = Parameter(np.random.uniform(-limit, limit, (input_dim, output_dim)))
    b = Parameter(np.zeros((1, output_dim)))
    linear = Linear(input_node, W, b)
    if activation == "sigmoid":
        act = Sigmoid(linear)
    elif activation == "relu":
        act = ReLU(linear)
    else:
        act = linear
    return [W, b], linear, act


x_node = Input()
y_node = Input()

param1, l1, a1 = make_layer(x_node, 2, 20, ACTIVATION)
param2, l2, a2 = make_layer(a1, 20, 20, ACTIVATION)
param3, l3, out = make_layer(a2, 20, 1, "sigmoid")

loss = BCE(y_node, out)

graph = [x_node, y_node] + param1 + [l1, a1] + param2 + [l2, a2] + param3 + [l3, out, loss]
trainables = param1 + param2 + param3


losses = []
for epoch in range(EPOCHS):
    total = 0
    for Xb, yb in get_batches(X_train, y_train, BATCH_SIZE):
        x_node.value = Xb
        y_node.value = yb
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainables, LR)
        total += loss.value
    losses.append(total / len(X_train))
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1} | Loss: {losses[-1]:.4f}")


correct = 0
for i in range(len(X_test)):
    x_node.value = X_test[i].reshape(1, -1)
    forward_pass(graph)
    pred = (out.value > 0.5).astype(int)
    correct += (pred == y_test[i])
acc = correct / len(X_test)




x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = []
for xi, yi in zip(xx.ravel(), yy.ravel()):
    x_node.value = np.array([[xi, yi]])
    forward_pass(graph)
    Z.append(out.value)
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
plt.title("MLP Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
