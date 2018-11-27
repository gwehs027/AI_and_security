### 上傳與下載檔案:

```
https://medium.com/pyradise/%E4%B8%8A%E5%82%B3%E6%AA%94%E6%A1%88%E5%88%B0google-colab-dd5369a0bbfd
如何才能將本機的檔案放上Colab去使用
Colab存取檔案的方式非常多元，可以存取Google Drive的檔案，
也能夠存取GCP上的檔案，甚至可以存取Google sheet，最後當然也能夠存取runtime的vm中的檔案
```
上傳檔案
```
from google.colab import files
uploaded = files.upload()
```
下載檔案
```
from google.colab import files
files.download('serviceAccount.json')
```

https://ithelp.ithome.com.tw/questions/10189177
```
#上載 2.csv
from google.colab import files
uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

# 確認 2.csv 是否已上載
!ls *.* -l

# use Pandas to read 2.csv
import pandas as pd
df = pd.read_csv('2.csv')
print(df)
```
```
from google.colab import files
import pandas as pd
files.upload()              # 上傳檔案
print(files.os.listdir())   # 查看裡面有什麼檔案
pd.read_csv('test.csv')     # 開啟已經上傳之檔案
```

# Colaboratory 指定 Google Drive 資料夾

```
https://itw01.com/VWFSKEN.html
https://codertw.com/%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7/818/
```
```
https://www.jianshu.com/p/ce2e63d1c10c

1.將所需文件上傳至Google Drive
2.新建或上傳ipnb文件，並用Colaboratory打開
3.在notebook中運行下方代碼進行授權綁定

運行下方代碼（傻瓜式粘貼運行即可）：

!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

運行完，過一會兒會要求兩次點進連結登陸google帳號並完成相關授權。
出現以下提示，算是完成授權。

P.S: 穀歌最近可能對 Colaboratory 進行了更新，上述代碼可能會報錯。報錯的話，可以嘗試以下代碼：

!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
#!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
#!apt-get update -qq 2>&1 > /dev/null
#!apt-get -y install -qq google-drive-ocamlfuse fuse
!wget [https://launchpad.net/](https://launchpad.net/)~alessandro-strada/+archive/ubuntu/google-drive-ocamlfuse-beta/+build/15331130/+files/google-drive-ocamlfuse_0.7.0-0ubuntu1_amd64.deb
!dpkg -i google-drive-ocamlfuse_0.7.0-0ubuntu1_amd64.deb
!apt-get install -f
!apt-get -y install -qq fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

4. 指定工作目錄
在指定之前先用!ls命令查看一下雲端自動分配的預設檔目錄，雲端預設的檔根目錄是datalab

運行下方代碼，指定檔根目錄：

# 指定Google Drive雲端硬碟的根目錄，名為drive
!mkdir -p drive
!google-drive-ocamlfuse drive
指定完之後，再用!ls命令查看綁定的檔根目錄，根目錄變為drive。

5. 指定當前工作資料夾
# 指定當前的工作資料夾
import os

# 此處為google drive中的檔路徑,drive為之前指定的工作根目錄，要加上
os.chdir("drive/Colab Notebooks/dog_project") 
再次用!ls查看當前的檔目錄

需要注意的是，Colaboratory是完全基於雲端運行的，每次登陸操作，後臺分配的機子都是隨機的，所以如果notebook運行需要額外的檔，
那麼在運行之前都要將檔先上傳至Google Drive，然後對Colaboratory指定所需的工作目錄。

以下是每次綁定都需要運行的代碼，將其添加在notebook前面運行。

# 授權綁定Google Drive
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
# 指定Google Drive雲端硬碟的根目錄，名為drive
!mkdir -p drive
!google-drive-ocamlfuse drive
# 指定當前的工作目錄
import os

# 此處為google drive中的檔路徑,drive為之前指定的工作根目錄，要加上
os.chdir("drive/.../...") 
# 查看檔目錄，是否包含所需的檔
!ls

作者：caoqi95
連結：https://www.jianshu.com/p/ce2e63d1c10c
來源：簡書
```

# 將檔案存到Github

# 執行網路的ipynb檔案
```
若要開啟的是一份存放在GitHub上的Notebook，可以直接更改網址便能自動以Colab開啟。

例如Notebook連結為:https://github.com/<一大串東西>.ipynb

能用以下網址開啟：https://colab.research.google.com/github/<一大串東西>.ipynb

而若是要複製整個repo，也可在Colab上使用!git clone指令，執行後也是存放到虛擬機上。
不過要注意，若將東西從虛擬機移到自己雲端所要花費的時間頗長。


```
# Tensorflow
```
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

```

### 確認GPU運行正常
```
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

### 安裝package

```
Keras

!pip install -q keras
import keras


PyTorch

from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'
!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.3.0.post4-{platform}-linux_x86_64.whl torchvision

import torch

或
!pip3 install torch torchvision

MxNet

!apt install libnvrtc8.0
!pip install mxnet-cu80
import mxnet as mx


OpenCV

!apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python
import cv2


XGBoost

!pip install -q xgboost==0.4a30
import xgboost


GraphViz

!apt-get -qq install -y graphviz && pip install -q pydot
import pydot
```

# 安裝 PyTorch
```
!pip install -q http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl torchvision

import torch

!apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg- dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

!pip install pyvirtualdisplay
!pip install piglet
!apt-get install xvfb
```
```
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
```
```
Deep Learning Adventures with PyTorch [Video]
Jakub Konczyk
Wednesday, October 31, 2018 
https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-adventures-pytorch-video

https://github.com/PacktPublishing/Deep-Learning-Adventures-with-PyTorch

FIRST STOP: A QUICK INTRODUCTION TO PYTORCH
SLEEPING UNDER THE STARS: IT'S A BIRD...IT'S A PLANE...IT’S A CNN?
GOING ABROAD: LANGUAGE DETECTION FOR FUN AND PROFIT WITH RNN
MAKING FRIENDS: LOST IN TRANSLATION WITH LSTM
GETTING SOME CULTURE: BECOMING A DEEP NEURAL PICASSO WITH DNN


```

# Reinforcement Learning強化學習
```
An introduction to Reinforcement Learning
https://www.youtube.com/watch?v=JgvyzIkgxF0
```
# 安裝 gym
```
See AI_and_security/gym_into.ipynb

!pip install gym

!apt-get install python-opengl -y

!pip install pyvirtualdisplay

!pip install piglet

!apt install xvfb -y
```
```
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
```
```
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# This code creates a virtual display to draw game images on. 
# If you are running locally, just ignore it
import os
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY"))==0:
    !bash ../xvfb start
    %env DISPLAY=:1
```
```
import gym
env = gym.make("MountainCar-v0")

plt.imshow(env.render('rgb_array'))
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
```
### ChainerRL Quickstart Guide

```
ChainerRL is a deep reinforcement learning library that 
implements various state-of-the-art deep reinforcement algorithms in Python using Chainer, a flexible deep learning framework.

Evolution of chainerrl (Gource Visualization) [11-09-2018]
https://www.youtube.com/watch?v=cWJY7XCpzUk
```
```
AI_and_security/quickstart.ipynb
```
