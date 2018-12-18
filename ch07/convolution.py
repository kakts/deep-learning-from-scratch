import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.util import im2col, col2im

class Convolution:
  def __init__(self, w, b, stride=1, pad=0):
    self.w = w
    self.b = b
    self.stride = stride
    self.pad = pad
  
  def forward(self, x):
    FN, C, FH, FW = self.w.shape
    N, C, H, W = x.shape
    out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
    out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

    col = im2col(x, FH, FW, self.stride, self.pad)
    col_w = self.w.reshape(FN, -1).T # フィルターの展開
    out = np.dot(col, col_w) + self.b
    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

    return out


class Pooling:
  def __init__(self, pool_h, pool_w, stride=1, pad=0):
    self.pool_h = pool_h
    self.pool_w = pool_w
    self.stride = stride
    self.pad = pad

  def forward(self, x):
    N, C, H, W = x.shape
    out_h = int(1 + (H - self.pool_h) / self.stride)
    out_w = int(1 + (W - self.pool_w) / self.stride)

    # 展開
    col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
    col = col.reshape(-1, self.pool_h * self.pool_w)

    # 最大値
    out = np.max(col, axis = 1)
    # 整形
    out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
    return out
