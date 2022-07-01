#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(50))
model.add(Dense(300))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([4])
print('4의 예측값 : ', result)

# loss :  3.126388037344441e-13
# 4의 예측값 :  [[4.0000005]]



# #6.20 
# [[1,2], [3,4], [5,6]] - (3,2)

# [[[1,2,3,4,5]]] - (1,1,5)

# [[1,2,3], [1,2,3], [4,5,6]] - (3,3)

# [4,3,2,1] - (4, )

# [[[[1,2,3], [4,5,6]]]] - (1,1,2,3)
