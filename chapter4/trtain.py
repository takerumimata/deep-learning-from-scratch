# p.92: ミニバッチ学習
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

# P.93
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choise(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

def coross_entropy_error(y, t): 
    # yがNNの出力, tが訓練データ
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1. y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

def numerical_diff(f, x):
    # 前方差分と中心差分について理解すること: f(x+h) - f(x)の差分は真の微分、真の接戦ではないことに留意する
    h = 1e-4
    return (f(x+h) -f(x-h)) / (2*h)

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = x_init

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
