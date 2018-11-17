## Webshell detection

### WebShell


### WebShell 檢測方法
```
主要有以下幾種：

[1]靜態檢測，通過匹配特徵碼，特徵值，危險函數函數來查找 WebShell 的方法，只能查找已知的 WebShell，並且誤報率漏報率會比較高，
但是如果規則完善，可以減低誤報率，但是漏報率必定會有所提高。

[2]動態檢測，執行時刻表現出來的特徵，比如資料庫操作、敏感檔讀取等。

[3]語法檢測，根據 PHP 語言掃描編譯的實現方式，進行剝離代碼、注釋，分析變數、函數、字串、語言結構的分析方式，來實現關鍵危險函數的捕捉方式。
這樣可以完美解決漏報的情況。但誤報上，仍存在問題。

[4]統計學檢測，通過資訊熵、最長單詞、重合指數、壓縮比等檢測

樸素貝葉斯

深度學習的MLP、

CNN
```

### WebShell Database
```

```
## 實戰Webshell detection
```
https://mp.weixin.qq.com/s/1V0xcjH-6V5qJoJILP0pJQ
```

```
初探機器學習檢測 PHP Webshell
 2018年02月02日
https://paper.seebug.org/526/
```

```

```

```

```
