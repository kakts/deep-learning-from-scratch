#c3.5.1 softmax
import numpy as np
a = np.array([0.3, 2.9, 4.0])

# aの各要素にexponentialを適用する
exp_a = np.exp(a)
print(exp_a)

# exp_aの各要素の和を求める
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)


#ソフトマックス関数
y = exp_a / sum_exp_a
print(y)

# 上記の処理を関数にまとめたもの
# ソフトマックス関数をコンピュータで扱ううえでの欠点
# 指数関数の桁の増加速度が非常に大きくてオーバーフローになる
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 上記の関数の改良版
# オーバーフローを解消する
def softmax_imp(a):
    # 入力値の中で最大値を取得
    c = np.max(a)
    # オーバーフロー対策として、最大値cを引く。こうすることで値が小さくなる
    exp_a = np.exp(a - c);
    sum_exp_a = np.sum(exp_a)

    y = exp_a / sum_exp_a
    return y
