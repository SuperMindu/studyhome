# [과제] 만들어서 깃허브 올리기 (val 썼을 때와 아닐 때 성능비교도 해보기)

from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=3000)
# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#         train_size=0.7, shuffle=True, random_state=66)

print(x)
print(y)
print(x.shape, y.shape)   # (506, 13) (506,)  
print(datasets.feature_names)   #sklearn 에서만 가능 # 컬럼, 열의 이름들
print(datasets.DESCR) # 데이터셋 및 컬럼에 대한 설명


#2.  모델 구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))


#3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.25)


#4. 평가, 예측
# fit에서 구해진 y = wx + b에 x_test와 y_test를 넣어보고 그 차이가 loss로 나온다(?)
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) # y의 예측값은 x의 테스트 값에 wx + b


r2 = r2_score(y_test, y_predict) # 계측용 y_test값과, y 예측값을 비교한다
print('r2스코어 : ', r2)

# val 없음
# loss :  3.763612747192383
# r2스코어 :  0.7181512623733383

# val 있음
# loss :  4.701908111572266
# r2스코어 :  0.4412643861922001








