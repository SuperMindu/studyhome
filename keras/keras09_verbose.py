import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66)

print(x)
print(y)
print(x.shape, y.shape)   # (506, 13) (506,)  

print(datasets.feature_names)   #sklearn 에서만 가능
print(datasets.DESCR)


#2.  모델 구성
model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


import time


#3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')

start_time = time.time()
print(start_time)    # 1656032956.615129 
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0) #verbose = 매개변수 (중간 훈련과정 스킵, 시간 단축 가능)
end_time = time.time() - start_time


print("걸린시간 : ", end_time)


'''

verbose 0 걸린시간 :  11.444794416427612 / 출력없음
verbose 1 걸린시간 :  12.67059874534607 / 잔소리 많음
verbose 2 걸린시간 :  11.591004848480225 / 프로그래스바 없음
verbose 3,4,5... 걸린시간 :  11.47945523262024 / epochs만 나옴 

'''



