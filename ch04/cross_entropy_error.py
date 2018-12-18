# 交差エントロピー誤差
import numpy as np

def cross_entropy_error(y, t):

  # 微小値を足す
  # 理由は np.log(0)の値が発生した場合 -infとなり、それ以上計算を進めれなくなrう
  delta = 1e-7
  return -np.sum(t * np.log(y + delta))

# 2を正解とする
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
