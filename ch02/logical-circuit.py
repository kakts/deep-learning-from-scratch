import numpy as np

# 重みによる実装
def AND(x1, x2):
    # 重みと閾値設定
    w1, w2, theta = 1.0, 1.0, 1.0
    tmp = w1 * x1 + w2 * x2
    if tmp <= theta:
        return 0
    else:
        return 1

# 重みとバイアスによるAND実装
# バイアス = 発火のしやすさを表す
# b = -0.1のとき、w * x1 + w * x2が0.1をこえただけで発火する
def ANDBias(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# 重みとバイアスによるNAND実装
def NANDBias(x1, x2):
    x = np.array([x1, x2])
    # 重みとバイアスだけがANDと異なる
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 重みとバイアスによるOR実装
def ORBias(x1, x2):
    x = np.array([x1, x2])
    # 重みとバイアスだけがANDと異なる
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 重みとバイアスによるOR実装
def XORBias(x1, x2):
    s1 = ORBias(x1, x2)
    s2 = NANDBias(x1, x2)
    y = ANDBias(s1, s2)
    return y

def printAND():
    print(XORBias(0, 0))
    print(XORBias(1, 0))
    print(XORBias(0, 1))
    print(XORBias(1, 1))

printAND()
