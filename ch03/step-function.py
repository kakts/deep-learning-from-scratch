import numpy as np
import matplotlib.pylab as plt
# num_pyのnp.arrayに対応していない
def step_function_old(x):
  if x > 0:
    return 1
  else:
    return 0

def step_function(x):

  # boolean to int
  return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
print(y)

plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y軸の範囲を指定
plt.show()


