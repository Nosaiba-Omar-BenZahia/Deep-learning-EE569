from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy

CLASS_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.01
EPOCHS = 100
TEST_SIZE = 0.25
BATCH_SIZES = [4, 16, 64]

def generate_xor_data(n_per_cluster=50):
    cov = np.array([[1, 0], [0, 1]])


    X0_1 = multivariate_normal.rvs(mean=[-3, 3], cov=cov, size=n_per_cluster)
    X0_2 = multivariate_normal.rvs(mean=[3, -3], cov=cov, size=n_per_cluster)
    X0 = np.vstack((X0_1, X0_2))

    X1_1 = multivariate_normal.rvs(mean=[3, 3], cov=cov, size=n_per_cluster)
    X1_2 = multivariate_normal.rvs(mean=[-3, -3], cov=cov, size=n_per_cluster)
    X1 = np.vstack((X1_1, X1_2))

    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(X0.shape[0]), np.ones(X1.shape[0])))

    return X, y

X, y = generate_xor_data(CLASS_SIZE)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("XOR Dataset")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.show()

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
test_size = int(len(X) * TEST_SIZE)
test_idx = indices[:test_size]
train_idx = indices[test_size:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in reversed(graph):
        n.backward()

def sgd_update(trainables, lr=LEARNING_RATE):
    for t in trainables:
        t.value -= lr * t.gradients[t]

def get_batches(X, y, batch_size):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx].reshape(-1, 1)


for BATCH_SIZE in BATCH_SIZES:
    print(f"\n🔹 Training Logistic Regression on XOR | Batch = {BATCH_SIZE}")


    W = np.random.randn(N_FEATURES, N_OUTPUT) * 0.1
    B = np.zeros((1, N_OUTPUT))

    x_node = Input()
    y_node = Input()
    W_node = Parameter(copy.deepcopy(W))
    B_node = Parameter(copy.deepcopy(B))
    linear = Linear(x_node, W_node, B_node)
    sigmoid = Sigmoid(linear)
    loss = BCE(y_node, sigmoid)

    graph = [x_node, W_node, B_node, linear, sigmoid, loss]
    trainables = [W_node, B_node]

    losses = []
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in get_batches(X_train, y_train, BATCH_SIZE):
            x_node.value = X_batch
            y_node.value = y_batch
            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainables, LEARNING_RATE)
            total_loss += loss.value

        avg_loss = total_loss / len(X_train)
        losses.append(avg_loss)

    correct = 0
    for i in range(X_test.shape[0]):
        x_node.value = X_test[i].reshape(1, -1)
        forward_pass(graph)
        pred = (sigmoid.value > 0.5).astype(int)
        if pred == y_test[i]:
            correct += 1
    acc = correct / X_test.shape[0]
    print(f"Final Accuracy: {acc * 100:.2f}%")


    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))
    Z = []
    for i, j in zip(xx.ravel(), yy.ravel()):
        x_node.value = np.array([[i, j]])
        forward_pass(graph)
        Z.append(sigmoid.value)
    Z = np.array(Z).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(f"Decision Boundary (Batch = {BATCH_SIZE})")
    plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
    plt.show()
