import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle
from common.functions import sigmoid, softmax


def get_data():
  (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
  return x_test, t_test


def init_network():
  # pickleファイルに保存された学習済みの重みパラメータを読み込む
  with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)
  
  return network

def predict(network, x):
  w1, w2, w3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, w1) + b1
  z1 = sigmoid(a1)

  a2 = np.dot(z1, w2) + b2
  z2 = sigmoid(a2)

  a3 = np.dot(z2, w3) + b3
  y = softmax(a3)

  return y


x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
  y = predict(network, x[i])

  # 配列の中の最大値 = ネットワークが推測した答え
  p = np.argmax(y)
  if p == t[i]:
    accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))