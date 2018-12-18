# ミニバッチ学習
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
  load_mnist(normalize=True, one_hot_label = True)

print(x_train.shape)
print(t_train.shape)

# この中から ランダムに10枚抜き出す
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

print(batch_mask)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 交差エントロピー誤差 ミニバッチ対応版
def cross_entropy_error(y, t):
  # yの次元数が1の場合
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  
  batch_size = y.shape[0]
  return -np.sum(t * np.log(y + 1e-7)) / batch_size