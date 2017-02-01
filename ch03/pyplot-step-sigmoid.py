import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-1, 1, 0.1)

def stepFunNormal(x):
    if x > 0:
        return 1
    else:
        return 0

# NumPy対応版
def step_function(x):
    y = x > 0
    # yがTrue: 1  False:0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plt.plot(x, stepFun, label = "stepFun")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()
