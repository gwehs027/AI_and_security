# 第一堂Tensorflow課
```
Tensor
Tensor的屬性
Tensor的運算
```

# 張量:

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
# 張量的屬性

```

```

# 張量的運算

### 張量的乘法運算
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
