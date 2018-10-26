# Google CoLaboratory
```
Colaboratory 是一個 Google 研究項目，旨在幫助傳播機器學習培訓和研究成果。
它是一個 Jupyter 筆記本環境，不需要進行任何設置就可以使用，並且完全在雲端運行。

Colaboratory 筆記本存儲在 Google 雲端硬碟 (https://drive.google.com/) 中，並且可以共用，
就如同您使用 Google 文檔或表格一樣。

Colaboratory 可免費使用。

免費使用 Nvidia Tesla K80 GPU
```

```
https://blog.csdn.net/qq_29592829/article/details/79444466
http://www.sohu.com/a/213845910_465975
https://makerpro.cc/2018/06/learn-ai-by-google-colaboratory/
★http://bangqu.com/t3y76W.html
★https://eventil.com/events/python-class-deep-reinforcement-learning-with-openai-gym[漂亮的學習環境]
```



# 打造你的GPU-based AI platform

```
Colaboratory自帶的框架為Tensorflow，也可以自己安裝別的函式庫。
為了import 不在Colab上的庫，可以直接使用!pip install <package name> 
或者!apt-get install <package name> 來做安裝
```

### 安裝matplotlib：
```
!pip install -q matplotlib-venn
```
### 調整TensorFlow的版本：
```
# To determine which version you're using:
!pip show tensorflow
# For the current version: 
!pip install --upgrade tensorflow
# For a specific version:
!pip install tensorflow==1.2
# For the latest nightly build:
!pip install tf-nightly
```
### 安裝Keras：
```
!pip install -q keras
import keras
```

### 安裝PyTorch：
```
!pip install -q http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl torchvision
import torch
```

### 安裝OpenCV：
```
!apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python
import cv2
```
### 安裝OpenAI gym

### 第一次測試gym==>成功2018.10.26
```
https://colab.research.google.com/github/MyDearGreatTeacher/AI_and_security/blob/master/gym_into.ipynb#scrollTo=F2TFUMJ_UyJc
```
### 第2次測試ChainerRL Quickstart Guide==>成功2018.10.26
```
https://colab.research.google.com/github/MyDearGreatTeacher/google_colab/blob/master/quickstart.ipynb#scrollTo=Mg8j6iCiwMVg
```
