from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=20)

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
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(1))



#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=1)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# loss :  0.6783761382102966
# r2스코어 :  0.5019386447163507

