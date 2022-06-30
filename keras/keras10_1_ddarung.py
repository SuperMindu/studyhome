# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd   # 엑셀 데이터 불러올 때 사용
from pandas import DataFrame 
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = './_data/ddarung/'  # .은 현재폴더라는 뜻
train_set = pd.read_csv(path + 'train.csv',  # train.csv의 데이터들이 train_set에 수치화 돼서 들어간다 
                        index_col=0)  # index_col=n. n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape)  # (1459, 10)


test_set = pd.read_csv(path + 'test.csv',  #예측에서 씀
                       index_col=0)
# submission = pd.read_csv(path + 'submission.csv') # 일단 이거를 읽어와야 함
submission_set = pd.read_csv('./_data/ddarung/submission.csv', index_col=0)
print(test_set)
print(test_set.shape)  # (715, 9)

print(train_set.columns)
print(train_set.info())  # 결측치 = 이빨 빠진 데이터

'''
(numpy의 주요 자료형)
정수 (int), 실수(float), 복소수(complex), 논리(bool)
numpy는 수치해석을 위한 라이브러리인 만큼, 숫자형 자료형에 대해서는 
파이썬 내장 숫자자료형에 비해 더욱 더 자세히 나누어놓은 자료형이 존재함
https://kongdols-room.tistory.com/53 <- 여기 보고 공부해보자
https://numpy.org/doc/stable/user/basics.types.html <- numpy공식 홈피
'''

print(train_set.describe())  # 

#### 결측치 처리 1. 제거 ####
# 결측치를 처리하는 방법은 여러가지가 있는데 일단은 제거
print(train_set.isnull().sum()) #null의 합계를 구함
train_set = train_set.dropna()
test_set = test_set.fillna(test_set.mean())
print(train_set.isnull().sum()) 
print(train_set.shape)  # (1328, 10)


x = train_set.drop(['count'], axis=1) # .drop - 데이터에서 ''사이 값 빼기, # axis=1 (열을 날리겠다), axis=0 (행을 날리겠다)
print(x)
print(x.columns)
print(x.shape)  # (1459, 9)

y = train_set['count']
print(y)
print(y.shape)  # (1459,) 

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.75, shuffle=True, random_state=31)


#2. 모델 구성
model=Sequential()
model.add(Dense(32, input_dim=9))
model.add(Dense(64, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(512, activation='selu'))
model.add(Dense(1024, activation='selu'))
model.add(Dense(2048, activation='selu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=800, batch_size=100)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict): #괄호 안의 변수를 받아들인다 :다음부터 적용 
    return np.sqrt(mean_squared_error(y_test, y_predict)) #루트를 씌워서 돌려줌 

rmse = RMSE(y_test, y_predict)  #y_test와 y_predict를 비교해서 rmse로 출력 (원래 데이터와 예측 데이터를 비교) 
print("RMSE : ", rmse)



y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape)  # (715, 1)  # 이거를 submission.csv 파일에 쳐박아야 한다

submission_set['count'] = y_summit
print(submission_set)
submission_set.to_csv('test1.csv', index=True)

end_time = time.time() - start_time
print("걸린시간 : ", end_time)


'''
y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape) # (715, 1)

submission_set = pd.read_csv(path + 'submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식


submission_set['count'] = y_summit
submission_set.to_csv('./_data/ddarung/submission.csv', index = True)



#### .to_csv() 를 사용해서 
### submission.csv를 완성하시오 



end_time = time.time() - start_time
print("걸린시간 : ", end_time)



# loss :  2360.45361328125
# RMSE :  48.58450201649001
# 걸린시간 :  15.192554950714111



x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.989, shuffle=True, random_state=100)


#2. 모델 구성
model=Sequential()
model.add(Dense(100, input_dim=9))
model.add(Dense(100, activation='swish'))
model.add(Dense(100, activation='swish'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=600, batch_size=6)
# loss :  838.1409301757812
# RMSE :  28.950666294450354
# 걸린시간 :  100.56669855117798 
'''

# 로스 0.001 r2 98 
# 로스 0.0001 r2 97 
# 이럴 경우 통상적으로 loss를 더 신뢰함
# 리더보드와 나의 데이터가 다를 때는 내꺼를 믿자
# 



#과제1. 함수에 대해서 공부하기 
#과제2. 낮은 로스값 스샷

