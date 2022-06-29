from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
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

#[실습] 아래를 완성할 것
# 1. train 0.7
# 2. R2 0.8 이상


#2.  모델 구성
model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(40))
model.add(Dense(200))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(150))
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(1))



#3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)



#4. 평가, 예측
# fit에서 구해진 y = wx + b에 x_test와 y_test를 넣어보고 그 차이가 loss로 나온다(?)
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) # y의 예측값은 x의 테스트 값에 wx + b


r2 = r2_score(y_test, y_predict) # 계측용 y_test값과, y 예측값을 비교한다
print('r2스코어 : ', r2)


# loss :  3.0040669441223145
# r2스코어 :  0.7677767333844028




