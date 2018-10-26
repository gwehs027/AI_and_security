# Google CoLaboratory
```
Colaboratory 是一個 Google 研究項目，旨在幫助傳播機器學習培訓和研究成果。
它是一個 Jupyter 筆記本環境，不需要進行任何設置就可以使用，並且完全在雲端運行。

Colaboratory 筆記本存儲在 Google 雲端硬碟 (https://drive.google.com/) 中，並且可以共用，就如同您使用 Google 文檔或表格一樣。

Colaboratory 可免費使用。
```

Colaboratory自帶的框架為Tensorflow，也可以自己安裝別的函式庫。

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

# 安裝OpenCV：
```
!apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python
import cv2
```
# 第一次測試gym==>成功2018.10.26
```
https://colab.research.google.com/github/MyDearGreatTeacher/AI_and_security/blob/master/gym_into.ipynb#scrollTo=F2TFUMJ_UyJc
```
