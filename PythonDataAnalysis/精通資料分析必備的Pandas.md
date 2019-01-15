# Pandas資料型態:series, dataframe,Panel
```
lab1:建立series的三種方法
lab2:series的運算
lab3:建立dataframe的種方法
lab4:dataframe的運算

```
#### lab1:建立series的三種方法
```
#先建立一個list再用Series轉成series

from pandas import Series

a = [1, 2, 3, 4]
s = Series(a)
s.index
s.values
s1 = Series(a, index=['A','B','C','D'])
```

```
import numpy as np
s2 = Series(np.arange(5))
```

```
d = {'A':1,'B':2,'C':3,'D':4}
s3 = Series(d)
s3.to_dict()
```

https://pandas.pydata.org/pandas-docs/stable/io.html
