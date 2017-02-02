import sys, os
sys.path.append(os.pardir) #親ディレクトリのファイルをいんぽーとするための設定
from dataset.mnist import load_mnist

# 数分かかる
#(訓練画像, 訓練ラベル), (テスト画像, テストラベル)
(x_train, t_train), (x_test, t_test) = \
    # normalize: 入力画像を0.0-1.0の値に正規化するか
    # flatten: 入力画像を１次元配列にする
    
    load_mnist(flatten=True, normalize=False)

# それぞれのデータの形状を出力
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
