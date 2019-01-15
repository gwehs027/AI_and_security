```
Python 資料運算與分析實戰：一次搞懂 NumPy•SciPy•Matplotlib•pandas 最強套件
作者： 中久喜健司  譯者： 莊永裕 出版社：旗標   出版日期：2018/02/05
```

```
第 1 章 資料科學運算與 Python
第 2 章 從實際專案看程式的開發重點
第 3 章 IPython 與 Spyder 開發環境
第 4 章 紮穩 Python 基礎
第 5 章 類別與物件的基礎
第 6 章 輸入與輸出
第 7 章 NumPy
第 8 章 SciPy
第 9 章 Matplotlib
第 10 章 pandas
第 11 章 程式效能最佳化(一)
第 12 章 程式效能最佳化(二)
(Cython、Numba、Numexpr)
附錄 A 參考文獻與學習資源
附錄 B 內建函式與標準函式庫
附錄 C NumPy 的函式庫參考文件
```
### 第 1 章 資料科學運算與 Python
```
import numpy as np
import time


def mult_basic(N, M, L, a, x, y):
    """ 不使用矩陣運算而以for迴圈來計算的函式
        但由於要建立所需大小的非ndarray有困難，
        輸入的變數使用NumPy的ndarray傳入 """
    r = np.empty((N, L))
    for i in range(N):
        for j in range(L):
            tmp = 0.0
            for k in range(M):
                tmp = tmp + a[k]*x[i][k]*y[k][j]
            r[i][j] = tmp
    return r


def mult_fast(N, M, L, a, x, y):
    """ 使用NumPy的函式來進行高速計算的函式
        和函式mult_basic得到完全相同的結果 """
    return np.dot(x*a, y)  # 處理的記述僅需1行


if __name__ == '__main__':
    # 產生計算用的陣列
    np.random.seed(0)
    N = 1000
    M = 100
    L = 1000
    a = np.random.random(M) - 0.5
    x = np.random.random((N, M)) - 0.5
    y = np.random.random((M, L)) - 0.5

    # 不使用矩陣運算而以for迴圈來計算
    ts = time.time()
    r1 = mult_basic(N, M, L, a, x, y)
    te = time.time()
    print("Basic method : %.3f [ms]" % (1000*(te - ts)))

    # 使用NumPy的函式來進行高速計算
    ts = time.time()
    r2 = mult_fast(N, M, L, a, x, y)
    te = time.time()
    print("Fast method  : %.3f [ms]" % (1000*(te - ts)))
```
Google Colab保出來的結果
```
Basic method : 87106.822 [ms]
Fast method  : 16.535 [ms]
```
### 第 2 章 從實際專案看程式的開發重點
```
第 2 章模擬程式的程式原始碼請參考底下的 GitHub repository。
https://github.com/pyjbooks/PyRockSim.git
```
