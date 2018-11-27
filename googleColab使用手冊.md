#
```
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

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

