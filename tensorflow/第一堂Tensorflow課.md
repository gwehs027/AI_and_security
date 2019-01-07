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
### 單一Tensor的運算:型態轉換==>使用tf.cast()
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

### 單一Tensor的運算:slice存取某區段資料==>使用tf.cast()
```
# -*- coding: utf-8 -*-
import tensorflow as tf

#"長度為5的一維張量"
t1=tf.constant([1,3,4,6,9],tf.float32)

#"從t1的第1個位置開始,取長度為3的區域"
t=tf.slice(t1,[1],[3])

#"創建會話"
session=tf.Session()
#"列印結果"
print(session.run(t))
```

```
# -*- coding: utf-8 -*-
import tensorflow as tf

#"3行4列的二維張量"
t2=tf.constant(
        [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]
        ],tf.float32
        )
#"從[0,1]位置開始,取高2寬2的區域"
t=tf.slice(t2,[0,1],[2,2])

#"創建會話"
session=tf.Session()
#"列印結果"
print(session.run(t))
```
```
# -*- coding: utf-8 -*-
import tensorflow as tf

#"3行3列2深度的三維張量"
t3d=tf.constant(
        [
        [[2,5],[3,3],[8,2]],
        [[6,1],[1,2],[5,4]],
        [[7,9],[2,-3],[-1,3]]
        ],tf.float32
        )
#"從[1,0,1]位置處,取高2寬2深度1的區域"
t=tf.slice(t3d,[1,0,1],[2,2,1])

#"創建會話"
session=tf.Session()
#"列印結果"
print(session.run(t))

```
### 單一Tensor的轉置運算==>使用tf.transpose()
```
# -*- coding: utf-8 -*-
import tensorflow as tf
#"2行3列的二維張量"
x=tf.constant(
        [
        [1,2,3],
        [4,5,6]
        ],tf.float32
        )
#"轉置"
r=tf.transpose(x,perm=[1,0])
rr=tf.transpose(x,perm=[0,1])

#"創建會話"
session = tf.Session()
print(session.run(r))
print(session.run(rr))

```
```
更多案例與說明請參看
https://www.tensorflow.org/api_docs/python/tf/transpose
```
### 單一Tensor的shape改變運算==>使用tf.reshape()
```
# -*- coding: utf-8 -*-
import tensorflow as tf
#"2行3列2深度的三維張量"
t3d=tf.constant(
        [
        [[1,2],[4,5],[6,7]],
        [[8,9],[10,11],[12,13]]
        ],tf.float32
        )

#"改變形狀為4行1列3深度的三維張量"
t1 = tf.reshape(t3d,[4,1,3])
# 也可以寫成t1 = tf.reshape(t3d,[4,1,-1])

session=tf.Session()
print(session.run(t1))
print(session.run(t3d))
```

```
# -*- coding: utf-8 -*-
import tensorflow as tf
#"四維張量"
t4d=tf.constant(
        [
        #"第1個 高2寬3深度2的三維張量"
        [
        [[2,5],[3,3],[8,2]],
        [[6,1],[1,2],[5,4]]
        ],
        #"第2個 高2寬3深度2的三維張量"
        [
        [[1,2],[3,6],[1,2]],
        [[3,1],[1,2],[2,1]]
        ]
        ],tf.float32
        )

#"轉換為高為2的二維張量"
t2d=tf.reshape(t4d,[2,-1])
#t2d=tf.reshape(t4d,[-1,3*3*2])

#"創建會話"
session=tf.Session()
print(session.run(t2d))
```
### 單一Tensor的Reduction化約運算==>求和、平均值、最大（小）值所在的索引直
```
# -*- coding: utf-8 -*-
import tensorflow as tf


t1d=tf.constant([3,4,1,5],tf.float32)

sum0=tf.reduce_sum(t1d)
#sum0=tf.reduce_sum(t1d,axis=0)

session=tf.Session()
print(session.run(sum0))
```
```
import tensorflow as tf
#"二維張量"
value2d=tf.constant(
        [
        [5,1,4,2],
        [3,9,5,7]
                ],tf.float32
        )

#"創建會話"
session=tf.Session()

#"計算沿0軸方向上的和"
sum0=tf.reduce_sum(value2d,axis=0)
print("沿 0 軸方向上的和:")
print(session.run(sum0))

#"計算沿1軸方向上的和"
sum1=tf.reduce_sum(value2d,axis=1)
print("沿 1 軸方向上的和:")
print(session.run(sum1))

#"計算沿(0,1)平面上的和"
sum01=tf.reduce_sum(value2d,axis=(0,1))
print("沿 (0,1) 平面上的和:")
print(session.run(sum01))

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
# Variable變數與placeholder預留位置
```
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#"預留位置"
x=tf.placeholder(tf.float32,[2,None],name='x')

# "3行2列矩陣"
w=tf.constant(
        [
        [1,2],
        [3,4],
        [5,6]
        ],tf.float32
        )

#"矩陣相乘"
y=tf.matmul(w,x)


#"創建會話"
session=tf.Session()

#"令x為2行2列的矩陣"
result1=session.run(y,feed_dict={x:np.array([[2,1],[1,2]],np.float32)})
print(result1)

#"令x為2行1列的矩陣"
result2=session.run(y,feed_dict={x:np.array([[-1],[2]],np.float32)})
print(result2)
```
