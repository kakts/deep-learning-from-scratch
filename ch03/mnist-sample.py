import sys, os
sys.path.append(os.pardir) #親ディレクトリのファイルをいんぽーとするための設定
from dataset.mnist import load_mnist

# 数分かかる
#(訓練画像, 訓練ラベル), (テスト画像, テストラベル)
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

# それぞれのデータの形状を出力
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
