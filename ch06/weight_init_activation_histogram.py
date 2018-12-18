import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
  return  1 / (1 + np.exp(-x))

def ReLU(x):
  return np.maximum(0, x)

def tanh(x):
  return np.tanh(x)

input_data = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

x = input_data

for i in range(hidden_layer_size):
  if i != 0:
    x = activations[i - 1]
  
  # 初期値の値をいろいろ変えてみる
  w = np.random.randn(node_num, node_num) * 1

  a = np.dot(x, w)

  # 活性化関数も変えて実験する
  # z = sigmoid(a)
  # z = ReLU(a)
  z = tanh(a)

  activations[i] = z

for i, a in activations.items():
  plt.subplot(1, len(activations), i+1)
  plt.title(str(i + 1) + "-layer")
  if i != 0:
    plt.yticks([], [])
  
  plt.hist(a.flatten(), 30, range=(0,1))
plt.show()