```
Numpy資料型態:ndarray
ndarray的各種運算
```
```
Array
lab1:建立1D array
lab2:建立2D array3*3
lab3:array屬性shape/size/dtype
list_1 = [1,2,3,4]
list_2 = [2,2,6,4]
array_1 = np.array([list_1,list_2])
array屬性shape/size/dtype===>
array_1.shape
array_1.size
array_1.dtype

lab4:
array_4 = 
np.zeros(5)
lab5:slice運算
c = np.array([[1,2,3],[4,5,6],[7,8,9]])
c
c[1:,0:2]
c[1:2,]
c[:,1:2]

Array and matrix manipulation
lab1:快速創建array(下列何者為誤??)
    np.random.randint(10,size = 20)==>產生大小20的array
    np.random.randint(10,size = 20).reshape(4,5)
    np.random.randint(10,size = 20).reshape[4,5]
    randdn和randint有何差異??
    
lab2:array運算a{+-*/}b
lab3:建立matrix:np.mat(a)====> array to matrix
     a = np.mat(np.random.randint(10,size = 20).reshape[4,5])
     b = np.mat(np.random.randint(10,size = 20).reshape[5,4])
     a*b==>?*?的matrix

lab4:matrix運算
lab5:array常用函式

universal function
nda = np.arange(12).reshape(2,6)
nda
np.square(nda)

broadcasting機制
nda = np.array([1,2,3])
nda+1
https://docs.scipy.org/doc/numpy/reference/index.html
```
