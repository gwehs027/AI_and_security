```
深度学习框架PyTorch快速开发与实战
邢梦来 (作者)　 
书　　号：978-7-121-34564-7
出版日期：2017-08-01
http://www.broadview.com.cn/book/5273

下载资源
资源下载-深度学习框架Pytorch快速开发与实践.zip
```
```
[已買]Deep Learning with PyTorch
Vishnu Subramanian
February 2018
```

```
[已買]Deep Learning with PyTorch [Video]
Anand Saha
Monday, April 30, 2018
```


```
[已買]Deep Learning and Neural Networks in PyTorch for Beginners [Video]
Daniel We
Thursday, June 21, 2018
```

```
[已買]Deep Reinforcement Learning Hands-On
Maxim Lapan
June 2018
```


```
[已買]Deep Learning Projects with PyTorch [Video]
AshishSingh Bhatia
Wednesday, June 27, 2018

GETTING READY WITH PYTORCH
CONVOLUTIONAL NEURAL NETWORK
UNDERSTANDING RNN AND LSTM
USING AUTOENCODERS FOR FRAUD DETECTION
RECOMMENDING A MOVIE WITH BOLTZMANN MACHINES<====重要
MOVIE RATING USING A AUTOENCODERS
MAKING MODEL FOR OBJECT RECOGNITION
```


```
[已買]Hands-On Intelligent Agents with OpenAI Gym
Praveen Palanisamy
July 2018

1: INTRODUCTION TO INTELLIGENT AGENTS AND LEARNING ENVIRONMENTS
2: REINFORCEMENT LEARNING AND DEEP REINFORCEMENT LEARNING
3: GETTING STARTED WITH OPENAI GYM AND DEEP REINFORCEMENT LEARNING
4: EXPLORING THE GYM AND ITS FEATURES
5: IMPLEMENTING YOUR FIRST LEARNING AGENT - SOLVING THE MOUNTAIN CAR PROBLEM
6: IMPLEMENTING AN INTELLIGENT AGENT FOR OPTIMAL CONTROL USING DEEP Q-LEARNING
7: CREATING CUSTOM OPENAI GYM ENVIRONMENTS - CARLA DRIVING SIMULATOR
8: IMPLEMENTING AN INTELLIGENT - AUTONOMOUS CAR DRIVING AGENT USING DEEP ACTOR-CRITIC ALGORITHM
9: EXPLORING THE LEARNING ENVIRONMENT LANDSCAPE - ROBOSCHOOL, GYM-RETRO, STARCRAFT-II, DEEPMINDLAB
10: EXPLORING THE LEARNING ALGORITHM LANDSCAPE - DDPG (ACTOR-CRITIC), PPO (POLICY-GRADIENT), RAINBOW (VALUE-BASED)
```

```
!pip install gym

!apt-get install python-opengl -y

!apt install xvfb -y

!pip install pyvirtualdisplay

!pip install piglet

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

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
3: GETTING STARTED WITH OPENAI GYM AND DEEP REINFORCEMENT LEARNING
```
#!/usr/bin/env python
import gym
env = gym.make('BipedalWalker-v2')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

```



```
[已買]Deep Learning Adventures with PyTorch [Video]
Jakub Konczyk
Wednesday, October 31, 2018
https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-adventures-pytorch-video
https://github.com/PacktPublishing/Deep-Learning-Adventures-with-PyTorch



FIRST STOP: A QUICK INTRODUCTION TO PYTORCH
SLEEPING UNDER THE STARS: IT'S A BIRD...IT'S A PLANE...IT’S A CNN?<====使用pre-trained model
GOING ABROAD: LANGUAGE DETECTION FOR FUN AND PROFIT WITH RNN
MAKING FRIENDS: LOST IN TRANSLATION WITH LSTM
GETTING SOME CULTURE: BECOMING A DEEP NEURAL PICASSO WITH DNN
```
Google Colab實測
```
!git clone https://github.com/PacktPublishing/Deep-Learning-Adventures-with-PyTorch.git

!ls Deep-Learning-Adventures-with-PyTorch/'Section 2'/train.py

!python3 Deep-Learning-Adventures-with-PyTorch/'Section 2'/train.py
```
2.SLEEPING UNDER THE STARS: IT'S A BIRD...IT'S A PLANE...IT’S A CNN?<====使用pre-trained model
```


```



```


```


