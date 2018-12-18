import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# 最後の層用の活性化関数
def identity_function(x):
  return x

x = np.array([1.0, 0.5])
w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])

print(w1.shape)
print(x.shape)
print(b1.shape)

a1 = np.dot(x, w1) + b1
z1 = sigmoid(a1)

print(a1)
print(z1)


#2層目
w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])

print(z1.shape)
print(w2.shape)
print(b2.shape)

a2 = np.dot(z1, w2) + b2
z2 = sigmoid(a2)

# 3層
w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])

a3 = np.dot(z2, w3) + b3
y = identity_function(a3)

print(y)