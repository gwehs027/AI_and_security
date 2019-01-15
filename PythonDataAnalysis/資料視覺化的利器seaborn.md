
# seaborn: statistical data visualization

```
可直接在Google Colab上跑

http://seaborn.pydata.org/index.html

https://ithelp.ithome.com.tw/articles/10186624
https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC2-5%E8%AC%9B-%E8%B3%87%E6%96%99%E8%A6%96%E8%A6%BA%E5%8C%96-matplotlib-seaborn-plotly-75cd353d6d3f
```

```
%matplotlib inline

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
normal_samples = np.random.normal(size = 100000) 

sns.distplot(normal_samples)
```

### 帶有X 軸變數與 Y 軸變數的直方圖
```
%matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

sns.jointplot(x = "speed", y = "dist", data = cars_df)
```


### 線圖（Line plot）==>使用factorplot()

```
%matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

sns.factorplot(data = cars_df, x="speed", y="dist", ci = None)
```

### 長條圖（Bar plot）==>使用 seaborn 套件的 countplot() 方法。
```
%matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

cyl = [6 ,6 ,4 ,6 ,8 ,6 ,8 ,4 ,4 ,6 ,6 ,8 ,8 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,4 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,8 ,6 ,8 ,4]
cyl_df = pd.DataFrame({"cyl": cyl})

sns.countplot(x = "cyl", data=cyl_df)
```

### 盒鬚圖（Box plot）==>使用 seaborn 套件的 boxplot() 方法。
```
%matplotlib inline

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
normal_samples = np.random.normal(size = 100000) 

sns.boxplot(normal_samples)
```
