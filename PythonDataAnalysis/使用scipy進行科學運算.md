# 科學運算的python:scipy

```
功能:
內插法
統計分析
Fast fourier transform
```
### 統計分析之rayleigh機率分布

```
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# (1)統計分佈函式的設定（預先Freeze）
rv = sp.stats.rayleigh(loc=1)

# (2)以上述統計分佈函式生成的亂數變數
r = rv.rvs(size=3000)

# (3)機率密度函式繪製用的百分點資料列
x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)

# 將取樣資料的分佈與原本的機率密度函數一同繪製
plt.figure(1)
plt.clf()
plt.plot(x, rv.pdf(x), 'k-', lw=2, label='機率密度函數')
plt.hist(r, normed=True, histtype='barstacked', alpha=0.5)
plt.xlabel('值')
plt.ylabel('分佈度')
plt.show()

```
### 內插法

```

import numpy as np
import matplotlib.pyplot as plt

# (1)由於名稱略長附加別名
import scipy.interpolate as ipl


# (2)原本的函數定義
def f(x):
    return (x-7) * (x-2) * (x+0.2) * (x-4)

# (3)生成原始資料（正解的值）
x = np.linspace(0, 8, 81)
y = np.array(list(map(f, x)))

# (4)補值前的寬刻度資料
x0 = np.arange(9)
y0 = np.array(list(map(f, x0)))

# (5)設定補值函式（線性補值）
#  設定補值函式（線性補值／3次樣條）
f_linear = ipl.interp1d(x0, y0, bounds_error=False)
f_cubic = ipl.interp1d(x0, y0, kind='cubic', bounds_error=False)
#  補值處理的執行
y1 = f_linear(x)  # 線性補值
y2 = f_cubic(x)  # 3次樣條補值

# (6)補值資料與原始資料的比較繪製
plt.figure(1)
plt.clf()
plt.plot(x, y, 'k-', label='原始函數')
plt.plot(x0, y0, 'ko', label='補值前資料', markersize=10)
plt.plot(x, y1, 'k:', label='線性補值', linewidth=4)
plt.plot(x, y2, 'k--', label='3次樣條補值', linewidth=4, alpha=0.7)
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.grid('on')
plt.show()
```
