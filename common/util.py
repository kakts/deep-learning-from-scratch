# coding: utf-8
import numpy as np


def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる
    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

# input_data: 入力データ(データ数, チャンネル, 高さ, 幅)
# filter_h: フィルターの高さ
# filter_w: フィルターの横幅
# stride: ストライド
# pad: パディング
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
  N, C, H, W = input_data.shape
  out_h = (H + (2 * pad) - filter_h) // stride + 1
  out_w = (W + (2 * pad) - filter_w) // stride + 1

  img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
  col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

  for y in range(filter_h):
    y_max = y + stride * out_h
    for x in range(filter_w):
      x_max = x + stride * out_w
      col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
  col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
  return col

# 逆伝播用
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
  N, C, H, W = input_shape
  out_h = (H + 2 * pad - filter_h) // stride + 1
  out_w = (W + 2 * pad - filter_w) // stride + 1
  col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

  img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
  for y in range(filter_h):
    y_max = y + stride * out_h
    for x in range(filter_w):
      x_max = x + stride * out_w
      img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
  
  return img[:, :, pad:H + pad, pad:W + pad]