#mnist画像表示プログラム
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

# 画像表示
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 画像データ読み込み
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]

print(label)

print(img.shape)
img = img.reshape(28, 28) # 形状を元の画像サイズに変形
print(img.shape)

img_show(img)
