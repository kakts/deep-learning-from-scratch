import sys, os
sys.path.append(os.pardir)

from layer_naive import MulLayer, AddLayer

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_total_layer = MulLayer()

# forward

# りんごの価格
apple_price = mul_apple_layer.forward(apple, apple_num)

# オレンジの価格
orange_price = mul_orange_layer.forward(orange, orange_num)

# りんごとオレンジの総額
apple_orange_price = add_apple_orange_layer.forward(apple_price, orange_price)

# 消費税を含めた総額
total_price = mul_total_layer.forward(apple_orange_price, tax)


# backward
d_price = 1
d_apple_orange_price, d_tax = mul_total_layer.backward(d_price)

d_apple_price, d_orange_price = add_apple_orange_layer.backward(d_apple_orange_price)

d_apple, d_apple_num = mul_apple_layer.backward(d_apple_price)

d_orange, d_orange_num = mul_orange_layer.backward(d_orange_price)

print(d_tax, d_apple_orange_price)
print(d_apple_price, d_apple, d_apple_num)
print(d_orange_price, d_orange, d_orange_num)



