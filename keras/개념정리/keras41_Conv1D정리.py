import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, Conv1D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping  
# from tensorflow.keras.layers import Bidirectional
import tensorflow as tf

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# y = ?
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4,5,6,7,8,9,10])
# (N, 3, 1) -> (행, 열, 데이터를 자르는 단위) 그래서 RNN은 shape가 3차원임
# x_shape = (행, 열, 몇개씩 자르는지!)
print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(7, 3, 1) # (7, 3, 1)
print(x.shape)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(64, input_shape=(3, 1))) 
# model.add(LSTM(10, input_shape=(3, 1), return_sequences=False)) # -> 출력 (None, 3, 10)
model.add(Conv1D(10, 2, input_shape=(3, 1)))
model.add(Flatten())#└> kernel size
model.add(Dense(3, activation='relu'))  # Dense는 2차, 3차원도 입력을 받을 수 있음. Dense는 무조건 그대로 감. 근데 위에서 Flatten 해주는 게 좋음
model.add(Dense(1))
model.summary() # LSTM Total params: 517
#                 Conv1D Total params: 97 

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 3, 10)             120

#  bidirectional (Bidirectiona  (None, 10)               160   <-- Param = units * Bidirectional(2) * (units + input_dim + bias)    
#  l)                                                           └>  160  = 5 * 2 (10 + 1 + 1)

#  dense (Dense)               (None, 3)                 33

#  dense_1 (Dense)             (None, 1)                 4

# =================================================================
# Total params: 317
# Trainable params: 317
# Non-trainable params: 0
# _________________________________________________________________


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=430, mode='min', verbose=1, restore_best_weights=True)
model.fit(x, y, epochs=3000, batch_size=1, callbacks=[es], verbose=1)

#4. 평가, 예측
# loss = model.evaluate(x, y)
# result = model.predict([8,9,10])
# print('loss : ', loss)
# print('[8,9,10]의 결과 : ', result)
# 이렇게 돌리면 model.predict([8,9,10])이 1차원이라 안됨
# RNN은 3차원이기 때문에 3차원과 비교를 해줘야 함
loss = model.evaluate(x, y)
y_pred = np.array([8,9,10]).reshape(1, 3, 1) # [[[8], [9], [10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)

# loss :  9.476319974055514e-05
# [8,9,10]의 결과 :  [[10.891523]]

# Conv1D
# loss :  3.248195508680392e-14
# [8,9,10]의 결과 :  [[11.]]


