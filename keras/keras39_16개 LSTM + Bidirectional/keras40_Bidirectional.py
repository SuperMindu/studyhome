import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping  
from tensorflow.keras.layers import Bidirectional
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
model.add(SimpleRNN(10, input_shape=(3, 1), return_sequences=True)) # -> 출력 (None, 3, 10)
model.add(Bidirectional(SimpleRNN(5)))
model.add(Dense(3, activation='relu')) 
model.add(Dense(1))
model.summary()
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

'''
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

# loss :  0.00013958293129689991
# [8,9,10]의 결과 :  [[10.616289]]

# loss :  0.0002554102975409478
# [8,9,10]의 결과 :  [[10.768097]]

# loss :  7.663348515052348e-05
# [8,9,10]의 결과 :  [[10.784917]]

# loss :  0.0001033739754348062
# [8,9,10]의 결과 :  [[10.835012]]

# loss :  9.476319974055514e-05
# [8,9,10]의 결과 :  [[10.891523]]
'''

