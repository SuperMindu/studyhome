import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array([range(10)])
# print(range(10))
# for i in range(10): # (0~9까지를 i라는 인수에 넣어서 반복)
#     print(i)
print(x.shape) # (1, 10)

x = np.transpose(x)
print(x.shape) # (10, 1)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])
print(y.shape)

y = np.transpose(y)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=400, batch_size=1)

#4. 평가, 예측 (열, 컬럼, 특성, 피처가 동일해야 한다)
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[9]])
print('[9]의 예측값 : ', result)

# loss :  2.1726508938300915e-13
# [9]의 예측값 :  [[9.9999981e+00 1.9000001e+00 9.3132257e-07]]


