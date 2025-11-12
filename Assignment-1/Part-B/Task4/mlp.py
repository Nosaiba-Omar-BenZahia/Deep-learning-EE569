from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from time import time

N_FEATURES = 64
NUM_CLASSES = 10
LEARNING_RATE = 0.01
EPOCHS = 2000
BATCH_SIZE = 64
TEST_SIZE = 0.4
LAYERS = [(64, "sigmoid"), (NUM_CLASSES, "softmax")]

def get_batches(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]

def create_mlp_layer(input_node, input_size, output_size, activation_function="sigmoid"):
    W = Parameter(np.random.randn(input_size, output_size) * 0.01)
    b = Parameter(np.zeros((1, output_size)))
    linear = Linear(input_node, W, b)
    if activation_function == "sigmoid":
        activation = Sigmoid(linear)
    elif activation_function == "relu":
        activation = ReLU(linear)
    elif activation_function == "tanh":
        activation = Tanh(linear)
    elif activation_function == "softmax":
        activation = Softmax(linear)
    return activation

def build_mlp():
    x_node = Input()
    input_node = x_node
    output_dim = N_FEATURES
    for layer, activation_function in LAYERS:
        input_node = create_mlp_layer(input_node, output_dim, layer, activation_function)
        output_dim = layer
    y_node = Input()
    loss = CE(y_node, input_node)
    return x_node, y_node, input_node, loss

def gather_graph(output_node):
    visited, ordering = set(), []
    def dfs(node):
        if node not in visited:
            visited.add(node)
            for inp in node.inputs:
                dfs(inp)
            ordering.append(node)
    dfs(output_node)
    return ordering

def gather_trainables(graph):
    return [node for node in graph if isinstance(node, Parameter)]

def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in reversed(graph):
        n.backward()

def sgd_update(trainables, learning_rate):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]

mnist = datasets.load_digits()
X, y = mnist['data'], mnist['target'].astype(int)

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

y = one_hot_encode(y, NUM_CLASSES)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

x_node, y_node, output_node, loss = build_mlp()
graph = gather_graph(loss)
trainables = gather_trainables(graph)

losses = []
start = time()
for epoch in range(EPOCHS):
    epoch_loss = 0
    num_samples = 0
    for X_batch, y_batch in get_batches(X_train, y_train, BATCH_SIZE):
        x_node.value = X_batch
        y_node.value = y_batch
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainables, LEARNING_RATE)
        num_samples += X_batch.shape[0]
        epoch_loss += loss.value
    losses.append(epoch_loss / num_samples)
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}, Loss: {losses[-1]:.6f}")
print(f"Training Time: {time()-start:.4f}")

correct_predictions = 0
for i in range(X_test.shape[0]):
    x_node.value = X_test[i:i+1]
    forward_pass(graph)
    prediction = np.argmax(output_node.value, axis=1)
    true_label = np.argmax(y_test[i])
    if prediction == true_label:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training Loss ({LAYERS[0][1].upper()} Activation)')
plt.show()
