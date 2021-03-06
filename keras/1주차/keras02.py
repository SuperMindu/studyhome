import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

#[실습]

model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=350)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('6의 예측값 : ', result)

# loss :  0.4262453615665436
# 6의 예측값 :  [[5.922587]]

# loss :  0.45326584577560425
# 6의 예측값 :  [[6.022912]]

# loss :  0.42535296082496643
# 6의 예측값 :  [[6.018991]]





