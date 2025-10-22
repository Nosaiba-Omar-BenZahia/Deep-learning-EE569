from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy


CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.01
EPOCHS = 100
TEST_SIZE = 0.25
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 200]

MEAN1 = np.array([-1, -1])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([1, 1])
COV2 = np.array([[1, 0], [0, 1]])

def generate_data(CLASS1, CLASS2):
    MEAN1 = np.array([-1, -1])
    COV1 = np.array([[1, 0], [0, 1]])
    MEAN2 = np.array([1, 1])
    COV2 = np.array([[1, 0], [0, 1]])

    X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
    X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(CLASS1), np.ones(CLASS2)))
    return X, y
X, y = generate_data(CLASS1_SIZE, CLASS2_SIZE)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Data')
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
    print(f"\n🔹 Training with batch size = {BATCH_SIZE}")

    W0 = np.zeros((1, 1))
    W = np.random.randn(2, 1) * 0.1

    x_node = Input()
    y_node = Input()
    W_node = Parameter(copy.deepcopy(W))
    B_node = Parameter(copy.deepcopy(W0))


    u_node = Linear(x_node, W_node, B_node)
    sigmoid = Sigmoid(u_node)
    loss = BCE(y_node, sigmoid)

    graph = [x_node, W_node, B_node, u_node, sigmoid, loss]
    trainable = [W_node, B_node]

    losses = []
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in get_batches(X_train, y_train, BATCH_SIZE):
            x_node.value = X_batch
            y_node.value = y_batch

            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, LEARNING_RATE)
            total_loss += loss.value

        avg_loss = total_loss / len(X_train)
        losses.append(avg_loss)

    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    correct = 0
    for i in range(X_test.shape[0]):
        x_node.value = X_test[i].reshape(1, -1)
        forward_pass(graph)
        pred = (sigmoid.value > 0.5).astype(int)
        if pred == y_test[i]:
            correct += 1
    acc = correct / X_test.shape[0]
    print(f"Accuracy: {acc * 100:.2f}%")

    plt.plot(losses)
    plt.title(f"Loss for Batch Size = {BATCH_SIZE}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
    Z = []
    for i, j in zip(xx.ravel(), yy.ravel()):
        x_node.value = np.array([[i, j]])
        forward_pass(graph)
        Z.append(sigmoid.value)
    Z = np.array(Z).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(f"Decision Boundary (Batch = {BATCH_SIZE})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
