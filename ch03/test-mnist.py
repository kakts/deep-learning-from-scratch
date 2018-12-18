import sys, os

sys.path.append(os.pardir)

from dataset.mnist import load_mnist

# normalize 入力画像を0.0-1.0に正規化する flatten 入力画像を平ら（１次元配列）にする
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)