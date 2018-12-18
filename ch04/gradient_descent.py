import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import numerical_gradient
# 勾配降下法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad
  return x

def func_question_1(x):
  return x[0] ** 2 + x[1] ** 2

init_x = np.array([-3.0, 4.0])
print(gradient_descent(func_question_1, init_x = init_x, lr=0.1, step_num=100))