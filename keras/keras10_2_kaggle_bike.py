# kaggle bike sharing
import numpy as np
import pandas as pd   # 엑셀 데이터 불러올 때 사용
from pandas import DataFrame 
import time
from tensorflow.keras.models import Sequential # 함수는 소문자로 시작, Class는 대문자로 시작 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 6/28 과제 - 함수와 Class 비교해서 조사
#             tensorflow certificate 사연 제출

#1. 데이터
path = './_data/kaggle_bike/'  # .은 현재폴더라는 뜻
train_set = pd.read_csv(path + 'train.csv')  # train.csv의 데이터들이 train_set에 수치화 돼서 들어간다 

# index_col=n. n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape)  # (10886, 8)


test_set = pd.read_csv(path + 'test.csv',  #예측에서 씀
                       index_col=0)
# submission = pd.read_csv(path + 'submission.csv') # 일단 이거를 읽어와야 함
submission_set = pd.read_csv('./_data/kaggle_bike/sampleSubmission.csv', index_col=0)
print(test_set)
print(test_set.shape)  # (6493, 8)

print(train_set.columns)
print(train_set.info())  # 결측치 = 이빨 빠진 데이터
print(train_set.describe())  # 

print(test_set.columns)
print(test_set.info())  # 결측치 = 이빨 빠진 데이터
print(test_set.describe())  # 


'''
#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum()) #null의 합계를 구함
train_set = train_set.dropna()
test_set = test_set.fillna(test_set.mean())
print(train_set.isnull().sum()) 
print(train_set.shape)  # (1328, 10)
'''


x = train_set.drop(['datetime','casual','registered','count'], axis=1)
print(x)
print(x.columns)
print(x.shape)  # (10886, 8)


y = train_set['count']
print(y)
print(y.shape)  #(10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=300)


#2. 모델 구성
model=Sequential()
model.add(Dense(128, input_dim=8))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100)


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
print(y_summit.shape)  # (6493, 1)  # 이거를 submission.csv 파일에 쳐박아야 한다


submission_set['count'] = abs(y_summit)
print(submission_set)
submission_set.to_csv('sampleSubmission.csv', index=True)

end_time = time.time() - start_time
print("걸린시간 : ", end_time)
