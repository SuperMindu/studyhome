# [과제] 만들어서 깃허브 올리기 (val 썼을 때와 아닐 때 성능비교도 해보기)

from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66)

print(x)
print(y)
print(x.shape, y.shape)  # (20640, 8) (20640,)

print(datasets.feature_names)
print(datasets.DESCR)

#2.  모델 구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(1))



#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=100, validation_split=0.25)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# val 없음
# loss :  0.6318250298500061
# r2스코어 :  0.5395433051463467

# val 있음
# loss :  0.6305921077728271
# r2스코어 :  0.5404417053426549