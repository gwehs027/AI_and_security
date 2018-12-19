
# Keras

```
TensorFlow+Keras深度學習人工智慧實務應用  書號：MP21710
作者： 林大貴  出版社：博碩
http://www.drmaster.com.tw/Bookinfo.asp?BookID=MP21710


Python深度學習實作：Keras快速上手      書號：MP11807
Keras快速上手-基于Python的深度学习实战 
作者： 謝梁, 魯穎, 勞虹嵐   譯者： 廖信彥   出版社：博碩
http://www.drmaster.com.tw/Bookinfo.asp?BookID=MP11807


Keras深度学习实战
安东尼奥·古利  人民邮电出版社


深度学习：Keras快速开发入门
乐毅  
https://github.com/yanchao727
https://github.com/yanchao727/keras_book

深度学习基于Keras的Python实践
魏贞原    电子工业出版社 
https://detail.tmall.com/item.htm?spm=a230r.1.14.74.19b476cdk9La3Q&id=572593310220&ns=1&abbucket=3


Python機器學習(第二版)  中文書 , Sebastian Raschka   Vahid Mirjalili   劉立民   吳建華 , 博碩 , 出版日期: 2018-08-30

Python深度學習  中文書 , Daniel Slater   Gianmario Spacagna   Valentino Zocca   劉立民   吳建華   陳開煇   Peter Roelants , 博碩 , 出版日期: 2018-01-05

```

```
深度学习：Keras快速开发入门
乐毅  
https://github.com/yanchao727


第1章 Keras概述 1
1.1 Keras簡介 1
1.1.1 Keras 2 1
1.1.2 Keras功能構成 4
1.2 Keras特點 6
1.3 主要深度學習框架 8
1.3.1 Caffe 8
1.3.2 Torch 10
1.3.3 Keras 12
1.3.4 MXNet 12
1.3.5 TensorFlow 13
1.3.5 CNTK 14
1.3.6 Theano 14

第2章 Keras的安裝與配置 16
2.1 Windows環境下安裝Keras 16
2.1.1 硬體設定 16
2.1.2 Windows版本 18
2.1.3 Microsoft Visual Studio版本 18
2.1.4 Python環境 18
2.1.5 CUDA 18
2.1.6 加速庫CuDNN 19
2.1.7 Keras框架的安裝 19
2.2 Linux環境下的安裝 20
2.2.1 硬體設定 20
2.2.2 Linux版本 21
2.2.3 Ubuntu環境的設置 22
2.2.4 CUDA開發環境 22
2.2.5 加速庫cuDNN 23
2.2.6 Keras框架安裝 24

第3章 Keras快速上手 25
3.1 基本概念 25
3.2 初識Sequential模型 29
3.3 一個MNIST手寫數位實例 30
3.3.1 MNIST數據準備 30
3.3.2 建立模型 31
3.3.3 訓練模型 32

第4章 Keras模型的定義 36
4.1 Keras模型 36
4.2 Sequential模型 38
4.2.1 Sequential模型介面 38
4.2.2 Sequential模型的資料登錄 48
4.2.3 模型編譯 49
4.2.4 模型訓練 50
4.3 函數式模型 51
4.3.1 全連接網路 52
4.3.2 函數模型介面 53
4.3.3 多輸入和多輸出模型 63
4.3.4 共用層模型 67


第5章 Keras網路結構 71
5.1 Keras層物件方法 71
5.2 常用層 72
5.2.1 Dense層 72
5.2.2 Activation層 74
5.2.3 Dropout層 75
5.2.4 Flatten層 75
5.2.5 Reshape層 76
5.2.6 Permute層 77
5.2.7 RepeatVector層 78
5.2.8 Lambda層 79
5.2.9 ActivityRegularizer層 80
5.2.10 Masking層 81
5.3 卷積層 82
5.3.1 Conv1D層 82
5.3.2 Conv2D層 84
5.3.3 SeparableConv2D層 87
5.3.4 Conv2DTranspose層 91
5.3.5 Conv3D層 94
5.3.6 Cropping1D層 97
5.3.6 Cropping2D層 97
5.3.7 Cropping3D層 98
5.3.8 UpSampling1D層 99
5.3.9 UpSampling2D層 100
5.3.10 UpSampling3D層 101
5.3.11 ZeroPadding1D層 102
5.3.12 ZeroPadding2D層 103
5.3.13 ZeroPadding3D層 104
5.4 池化層 105
5.4.1 MaxPooling1D層 105
5.4.2 MaxPooling2D層 106
5.4.3 MaxPooling3D層 108
5.4.4 AveragePooling1D層 109
5.4.5 AveragePooling2D層 110
5.4.6 AveragePooling3D層 111
5.4.7 GlobalMaxPooling1D層 112
5.4.8 GlobalAveragePooling1D層 113
5.4.9 GlobalMaxPooling2D層 113
5.4.10 GlobalAveragePooling2D層 114
5.5 局部連接層 115
5.5.1 LocallyConnected1D層 115
5.5.2 LocallyConnected2D層 117
5.6 迴圈層 120
5.6.1 Recurrent層 120
5.6.2 SimpleRNN層 124
5.6.3 GRU層 126
5.6.4 LSTM層 127
5.7 嵌入層 129
5.8 融合層 131
5.9 啟動層 134
5.9.1 LeakyReLU層 134
5.9.2 PReLU層 134
5.9.3 ELU層 135
5.9.4 ThresholdedReLU層 136
5.10 規範層 137
5.11 雜訊層 139
5.11.1 GaussianNoise層 139
5.11.2 GaussianDropout層 139
5.12 包裝器Wrapper 140
5.12.1 TimeDistributed層 140
5.12.2 Bidirectional層 141
5.13 自訂層 142


第6章 Keras數據預處理 144
6.1 序列數據預處理 145
6.1.1 序列數據填充 145
6.1.2 提取序列跳字樣本 148
6.1.3 生成序列抽樣概率表 151
6.2 文本預處理 153
6.2.1 分割句子獲得單詞序列 153
6.2.2 OneHot序列編碼器 154
6.2.3 單詞向量化 155
6.3 圖像預處理 159


第7章 Keras內置網路配置 167
7.1 模型性能評估模組 168
7.1.1 Keras內置性能評估方法 168
7.1.2 使用Keras內置性能評估 170
7.1.3 自訂性能評估函數 171
7.2 損失函數 171
7.3 優化器函數 174
7.3.1 Keras優化器使用 174
7.3.2 Keras內置優化器 176
7.4 啟動函數 180
7.4.1 添加啟動函數方法 180
7.4.2 Keras內置啟動函數 181
7.4.3 Keras高級啟動函數 185
7.5 初始化參數 189
7.5.1 使用初始化方法 189
7.5.2 Keras內置初始化方法 190
7.5.3 自訂Keras初始化方法 196
7.6 正則項 196
7.6.1 使用正則項 197
7.6.2 Keras內置正則項 198
7.6.3 自訂Keras正則項 198
7.7 參數約束項 199
7.7.1 使用參數約束項 199
7.7.2 Keras內置參數約束項 200


第8章 Keras實用技巧和視覺化 202
8.1 Keras調試與排錯 202
8.1.1 Keras Callback回呼函數與調試技巧 202
8.1.2 備份和還原Keras模型 215
8.2 Keras內置Scikit-Learn介面包裝器 217
8.3 Keras內置視覺化工具 224


第9章 Keras實戰 227
9.1 訓練一個準確率高於90%的Cifar-10預測模型 227
9.1.1 數據預處理 232
9.1.2 訓練 233
9.2 在Keras模型中使用預訓練詞向量判定文本類別 239
9.2.1 資料下載和實驗方法 240
9.2.2 數據預處理 241
9.2.3 訓練 245
9.3 用Keras實現DCGAN生成對抗網路還原MNIST樣本 247
9.3.1 DCGAN網路拓撲結構 250
9.3.2 訓練 254

```

# pyTorch

```
一直學不會Tensorflow?PyTorch更好用更強大更易懂! 廖星宇 , 深石 , 出版日期: 2018-09-25
深度學習入門之PyTorch 簡體書 , 廖星宇 , 電子工業出版社 , 出版日期: 2017-10-01
https://github.com/Elin24/learning_pyTorch_with_SherlockLiao


AI視覺大全：用最好用的PyTorch實作  唐進民 , 佳魁資訊 , 出版日期: 2018-10-09
深度學習之PyTorch實戰計算機視覺  唐進民 , 電子工業出版社 , 出版日期: 2018-06-01



PyTorch深度學習與自然語言中文處理   孫洋洋   王碩   邢夢來   廖信彥 , 博碩 , 出版日期: 2018-12-12

比Tensorflow還精美的人工智慧套件：PyTorch讓你愛不釋手  陳雲 , 佳魁資訊 , 出版日期: 2018-07-06
深度學習框架PyTorch：入門與實踐  簡體書 , 陳雲（編） , 電子工業出版社 , 出版日期: 2018-01-01


神經網路與PyTorch實戰  簡體書 , 肖智清 , 機械工業出版社 , 出版日期: 2018-08-01

深度學習框架PyTorch快速開發與實戰   簡體書 , 邢夢來等 , 電子工業出版社 , 出版日期: 2018-08-01




```
