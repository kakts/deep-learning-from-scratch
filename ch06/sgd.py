class SGD:
  def __init__(self, lr=0.01):
    self.lr = lr
  
  def update(self, params, grads):
    for key in params.keys():
      # 学習率に傾きをかけたものを引いていく
      params[key] -= self.lr * grads[key]