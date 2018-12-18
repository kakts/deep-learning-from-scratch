import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error, numerical_gradient

class simpleNet:
  def __init__(self):
    self.W = np.random.randn(2, 3) # ガウス分布で正規化
  
  def predict(self, x):
    return np.dot(x, self.W)

  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)

    return loss
  

net = simpleNet()
print(net.W) # 重みパラメータを表示

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print(np.argmax(p))

t = np.array([0,0,1]) #正解ラベル
loss = net.loss(x, t)
print(loss)

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)