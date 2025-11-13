import numpy as np
from EDF import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
from time import time

LEARNING_RATE = 0.05
EPOCHS = 2
BATCH_SIZE = 128
NUM_CLASSES = 10
LR_DECAY = 1
DECAY_STEPS = 1

def build_cnn():
    x_node = Input()
    conv1 = Conv(x_node, 1, 8, 3, 3, stride=1, padding=1)
    relu1 = ReLU(conv1)
    pool1 = MaxPooling(relu1, 2, 2)

    conv2 = Conv(pool1, 8, 16, 3, 3, stride=1, padding=1)
    relu2 = ReLU(conv2)
    pool2 = MaxPooling(relu2, 2, 2)

    conv3 = Conv(pool2, 16, 32, 3, 3, stride=1, padding=1)
    relu3 = ReLU(conv3)
    pool3 = MaxPooling(relu3, 2, 2)

    flatten_node = Flatten(pool3)
    linear = Linear(flatten_node, 288, 10)
    softmax = Softmax(linear)

    y_node = Input()
    loss = CE(y_node, softmax)

    return x_node, y_node, loss, softmax

def gather_graph(output_node):
    visited = set()
    ordering = []
    def dfs(node):
        if node not in visited:
            visited.add(node)
            for i in node.inputs:
                dfs(i)
            ordering.append(node)
    dfs(output_node)
    return ordering

def gather_trainables(graph):
    return [n for n in graph if isinstance(n, Parameter)]

def forward(graph):
    for n in graph:
        n.forward()

def backward(graph):
    for n in reversed(graph):
        n.backward()

def sgd(trainables, lr):
    for t in trainables:
        t.value -= lr * t.gradients[t]

def get_batches(X, y, batch_size):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        batch = idx[i:i+batch_size]
        yield X[batch], y[batch]

def evaluate(X, Y, x_node, y_node, graph, softmax_node, batch_size):
    correct = 0
    total = 0
    for Xb, Yb in get_batches(X, Y, batch_size):
        x_node.value = Xb
        y_node.value = Yb
        forward(graph)
        pred = softmax_node.value.argmax(axis=1)
        lab = Yb.argmax(axis=1)
        correct += np.sum(pred == lab)
        total += len(Xb)
    return correct / total

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

def one_hot(y):
    o = np.zeros((len(y), NUM_CLASSES))
    o[np.arange(len(y)), y] = 1
    return o

y_train = one_hot(y_train)
y_test = one_hot(y_test)

def train_cnn(X_train, y_train):
    x_node, y_node, loss_node, softmax_node = build_cnn()
    graph = gather_graph(loss_node)
    trainables = gather_trainables(graph)
    losses = []
    lr = LEARNING_RATE
    for epoch in range(EPOCHS):
        if epoch % DECAY_STEPS == 0 and epoch != 0:
            lr *= LR_DECAY
        epoch_loss = 0
        seen = 0
        for Xb, Yb in get_batches(X_train, y_train, BATCH_SIZE):
            x_node.value = Xb
            y_node.value = Yb
            forward(graph)
            backward(graph)
            sgd(trainables, lr)
            epoch_loss += loss_node.value
            seen += len(Xb)
            print(f"Epoch {epoch+1}/{EPOCHS}, samples:{seen}/{len(X_train)}, Loss={epoch_loss/seen:.5f}, LR={lr}")
            losses.append(epoch_loss / seen)
    return x_node, y_node, graph, softmax_node, losses

start = time()
x_node, y_node, graph, softmax_node, losses = train_cnn(X_train, y_train)
print("Training Time:", time() - start)

acc = evaluate(X_test, y_test, x_node, y_node, graph, softmax_node, BATCH_SIZE)
print("Test Accuracy:", acc * 100, "%")

plt.plot(losses)
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Loss")
plt.show()
