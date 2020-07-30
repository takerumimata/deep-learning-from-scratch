## 3.2.3 ステップ関数のグラフ
## ステップ関数をグラフとして表現する

import numpy as np
import matplotlib.pylab as plt
import sys

print(sys.version)

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y軸の範囲を指定
plt.show()