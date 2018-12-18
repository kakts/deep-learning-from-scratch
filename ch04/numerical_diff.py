import numpy as np
import matplotlib.pylab as plt
# 数値微分 hが小さくて丸め誤差が発生する
# そのため (x+h) (x-h)でのｆの差分を計算する 中心差分という
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def func1(x):
  return 0.01 * x ** 2 + 0.1 * x

x = np.arange(0.0, 20.0, 0.1) # 0から20まで0.1刻みのx配列
y = func1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff(func1, 5))
print(numerical_diff(func1, 10))