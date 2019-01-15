# 梯度與梯度下降法
### 梯度
```

```
### 優化器 (Optimizers)與梯度下降法

```
建立好模型之後，接著就是要使用既有的資料對模型進行訓練

TensorFlow 所提供的 optimizers 可以對模型的 variable 進行微調，讓 loss function 達到最小

最簡單的 optimizer 就是 gradient descent，他會依照 loss function 對個變數的 gradient 方向調整變數

TensorFlow 的 tf.gradients 可幫助我們計算函數的微分，而 optimizers 也會自動幫我們處理這部分的問題。
```
```
Optimizer 
GradientDescentOptimizer 
AdagradOptimizer 
AdagradDAOptimizer 
MomentumOptimizer 
AdamOptimizer 
FtrlOptimizer 
RMSPropOptimizer
```
```
https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer

https://blog.gtwang.org/statistics/tensorflow-google-machine-learning-software-library-tutorial/2/

http://ruder.io/optimizing-gradient-descent/
```

```
# -*- coding: utf-8 -*-
import tensorflow as tf

#"首先將變數初始化:梯度下降的初始點"
x=tf.Variable(4.0,dtype=tf.float32)

#"函數" y=(x-1)**2
y=tf.pow(x-1,2.0)

#"梯度下降,學習率為0.25"
opti=tf.train.GradientDescentOptimizer(0.25).minimize(y)

#"創建會話"
session=tf.Session()
session.run(tf.global_variables_initializer())

#"三次反覆運算"
for i in range(3):
    session.run(opti)
    #"列印每次反覆運算的值"
    print(session.run(x))
```

```
# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

#"首先將變數初始化:梯度下降的初始點"
x=tf.Variable(15.0,dtype=tf.float32)

#"函數"
y=tf.pow(x-1,2.0)

#"梯度下降,設置學習率為0.25"
opti=tf.train.GradientDescentOptimizer(0.05).minimize(y)


#"畫曲線"
value=np.arange(-15,17,0.01)
y_value=np.power(value-1,2.0)
plt.plot(value,y_value)

#"創建會話"
session=tf.Session()
session.run(tf.global_variables_initializer())

#"三次反覆運算"
for i in range(100):
    session.run(opti)
    if(i%10==0):
        v=session.run(x)
        plt.plot(v,math.pow(v-1,2.0),'go')
        print('第 %d 次的 x 的反覆運算值: %f'%(i+1,v))
plt.show()
```
```
https://colab.research.google.com/drive/1Wm_obDjaDCbXk1rCAQBwHowu7-oZGsVE#scrollTo=2o1_7mxa4ers
```
### 多元函數的SGD
```
# -*- coding: utf-8 -*-
import tensorflow as tf#"梯度下降的初始點"
x1=tf.Variable(-4.0,dtype=tf.float32)
x2=tf.Variable(4.0,dtype=tf.float32)

#"函數"
y=tf.square(x1)+tf.square(x2)

#"創建會話"
session=tf.Session()
session.run(tf.global_variables_initializer())

#"梯度下降,設置步長為0.25"
opti=tf.train.GradientDescentOptimizer(0.25).minimize(y)

#"2次反覆運算"
for i in range(2):
    session.run(opti)
    #"列印每次反覆運算的值"
    print((session.run(x1),session.run(x2)))
```
# 線性迴歸分析

```
https://blog.csdn.net/xierhacker/article/details/53257748

https://blog.csdn.net/xierhacker/article/details/53261008
```
```
# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#"6個點的橫坐標"
x=tf.constant([1,2,3,4,5,6],tf.float32)

#"6個點的縱坐標"
y=tf.constant([3,4,7,8,11,14],tf.float32)

#"初始化直線的斜率"
w=tf.Variable(1.0,dtype=tf.float32)

#"初始化直線的截距"
b=tf.Variable(1.0,dtype=tf.float32)

#"6個點到直線數值方向上距離的平方和"
loss=tf.reduce_sum(tf.square(y-(w*x+b)))

#創建會話
session=tf.Session()
session.run(tf.global_variables_initializer())

#"梯度下降法"
opti=tf.train.GradientDescentOptimizer(0.005).minimize(loss)

#"記錄每一次反覆運算後的平均平方誤差(Mean Squared Error)"
MSE=[]

#"迴圈500次"
for i in range(500):
    session.run(opti)
    MSE.append(session.run(loss))
    #"每隔50次列印直線的斜率和截距"
    if i%50==0:
        print((session.run(w),session.run(b)))

#"畫出損失函數的值"
plt.figure(1)
plt.plot(MSE)
plt.show()

#"畫出6個點及最後計算出的直線"
plt.figure(2)
x_array,y_array=session.run([x,y])
plt.plot(x_array,y_array,'o')
xx=np.arange(0,10,0.05)
yy=session.run(w)*xx+session.run(b)
plt.plot(xx,yy)
plt.show()

```

```
https://colab.research.google.com/drive/1hWvr_wY4lzoiYzQZhcbQu6AsdOLaOSAd
```

```
# -*- coding: utf-8 -*-
import tensorflow as tf

#"第1個Variable，初始化為一個長度為3的一維張量"
v1=tf.Variable(tf.constant([1,2,3],tf.float32),dtype=tf.float32,name='v1')

#"第2個Variable，初始化為一個長度為2的一維張量"
v2=tf.Variable(tf.constant([4,5],tf.float32),dtype=tf.float32,name='v2')

#"宣告一個tf.train.Saver物件"
saver =tf.train.Saver()


#"創建會話"
session=tf.Session()
#"初始化變數"
session.run(tf.global_variables_initializer())

#"將變數 v1 和 v2 保存到當前資料夾下的model.ckpt文件中"
save_path=saver.save(session,'./model.ckpt')
session.close()
```

```
# -*- coding: utf-8 -*-
import tensorflow as tf
#"初始化變數2個變數，形狀與model.cpkt檔中變數是必須相等的"
v1=tf.Variable([11,12,13],dtype=tf.float32,name='v1')
v2=tf.Variable([15,16],dtype=tf.float32,name='v2')
#"聲明一個tf.train.Saver類"
saver =tf.train.Saver()
with tf.Session() as sess:
    #"載入model.ckpt文件"
    saver.restore(sess,'./model.ckpt')
    #"列印兩個變數的值"
    print(sess.run(v1))
    print(sess.run(v2))
sess.close()
```
