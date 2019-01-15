# matplotlib
```
# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../dataset/lena.png') 
plt.imshow(img)

plt.show()
#https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch01/img_show.py
# 更多參數說明,請參閱底下無範例
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
```
```
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1) 
y = np.sin(x)

plt.plot(x, y)
plt.show()
```
```
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 6, 0.1) 
y1 = np.sin(x)
y2 = np.cos(x)


plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos")
plt.xlabel("x") 
plt.ylabel("y") 
plt.title('sin & cos')
plt.legend()
plt.show()
```

```
import numpy as np
import matplotlib.pyplot as plt

a = [1, 2, 3]
b = [4, 5, 6]
plt.plot(a, b)
plt.plot(a, b, '*')
```

#### 3. Histogram and KDE Plot
```

```

```
```

