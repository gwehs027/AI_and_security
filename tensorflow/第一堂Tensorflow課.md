# 第一堂Tensorflow課
```
1.Tensor
2.Tensor的屬性
3.Tensor的運算
```

# 1.張量:constant常數與Variable變數

張量的意義==>詳見簡報

### 利用tf.constant建立張量
```
Google tf.constant()的用法
https://www.tensorflow.org/api_docs/python/tf/constant
```
```
tf.constant(
    value,
    dtype=None,
    shape=None,
    name='Const',
    verify_shape=False
)
```
### 建立一個常數Creates a constant tensor
```
import tensorflow as tf
print(tf.__version__)
```

```
import tensorflow as tf

t=tf.constant([1,3,5,7,9],tf.float32,name='t')
print(t)
```
### Tensor與numpy的ndarray互換

Tensor==>ndarray
```
# -*- coding: utf-8 -*-
import tensorflow as tf
#"一維張量"
t=tf.constant([1,2,3],tf.float32)

#"創建會話"
session=tf.Session()
#"張量轉換為ndarray"
array=session.run(t)

#"列印其資料結構類型及對應值"
print(type(t))
print(t)
print(type(array))
print(array)
```

ndarray==>Tensor
```
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#"一維的ndarray"
array=np.array([1,2,3],np.float32)

#"ndarray轉換為tensor"
t=tf.convert_to_tensor(array,tf.float32,name='t')

#"列印張量"
print(array)
print(t)
```
# 2.Tensor的屬性==>維數(階)[dim/rank]、形狀(shape)和資料類型

### 維數(階)[dim/rank]形狀(shape)
```
# -*- coding: utf-8 -*-
import tensorflow as tf
#"張量"
t=tf.constant(
        [
        [1,2,3],
        [4,5,6]
        ]
        ,tf.float32)
#"張量的形狀"
s=tf.shape(t)
r=tf.rank(t)

session=tf.Session()
print('張量的內容是:',session.run(t))
print('張量的形狀是:',session.run(s))
print('張量的維度是:',session.run(r))
```

### 將圖像資料轉為Tensor
```
https://colab.research.google.com/drive/1AtKW1Xx3TlACI-Z_6WUarWRnN_v08nC6#scrollTo=DJWqkRc0pkEZ
```
```
# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

#"讀取圖片檔"
image=tf.read_file("test.jpg",'r')

#"將圖片檔解碼為Tensor"
image_tensor=tf.image.decode_jpeg(image)

#"圖像張量的形狀"
shape=tf.shape(image_tensor)
session=tf.Session()
print('圖像的形狀:')
print(session.run(shape))

#"Tensor 轉換為 ndarray"
image_ndarray=image_tensor.eval(session=session)

#"顯示圖片"
plt.imshow(image_ndarray)
plt.show()
```

# 張量的運算
```
單一Tensor的運算:
多Tensor的各種運算
```
### 單一Tensor的運算:型態轉換
```
# -*- coding: utf-8 -*-
import tensorflow as tf
#"張量"
t=tf.constant(
        [
        [1,0,3],
        [0,0,0]
        ]
        ,tf.float32)
#"數值型轉換為bool類型"
r=tf.cast(t,tf.bool)    


session=tf.Session()
print(session.run(r))
```

```
# -*- coding: utf-8 -*-
import tensorflow as tf
#"張量"
t=tf.constant(
        [
        [False,True,False],
        [False,False,True]
        ]
        ,tf.bool)
#"bool型轉換為數值型"
r=tf.cast(t,tf.float32)        


session=tf.Session()
print(session.run(r))
```
### 多Tensor的加法運算
```
import tensorflow as tf

#"2行2列深度是2的二維張量"
x=tf.constant(
        [
        [[2,5],[4,3]],
        [[6,1],[1,2]]
        ],tf.float32
        )

#"2行2列深度是2的二維張量"
y=tf.constant(
        [
        [[1,2],[2,3]],
        [[3,2],[5,3]]
        ],tf.float32
        )
#待執行的運算:"x與y的和"
result=x+y

# 定義Model==>執行Model運算
session=tf.Session()
print(session.run(result))
```

### 多Tensor的乘法運算

```
# -*- coding: utf-8 -*-
import tensorflow as tf

#"2行2列的矩陣"
x=tf.constant(
        [[1,2],[3,4]]
        ,tf.float32
        )

#"2行1列的矩陣"
w=tf.constant([[-1],[-2]],tf.float32)

#"矩陣的乘法"
y=tf.matmul(x,w)

#"創建會話"
session=tf.Session()

#"列印矩陣相乘後的結果"
print(session.run(y))

```
