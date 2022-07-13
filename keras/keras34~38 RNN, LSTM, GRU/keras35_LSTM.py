import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping  
# RNN의 문제점 : 장기 의존성(데이터를 많이 자르면 앞쪽의 데이터는 영향력이 미미해짐), 기울기(weight) 손실
# LSTM = Long Short-Term Memory Network
# LSTM은 학습하면서 이전의 학습을 잊어버리지 않게하는 기능이 있음
# input, output, forget ...
# LSTM 정리 해서 그림 그려서 쌤 메일 보내기 !!!!!!!!!! 
# https://wooono.tistory.com/242 <- 여기 있음 ㅎ

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
'''
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3, 1))) # [batch, timesteps, feature] (timesteps만큼 잘라서 feature)
#                      └> units           └> input_dim 
# model.add(SimpleRNN(units=10, input_length=3, input_dim=1)) # 위에거를 이렇게 바꿔서 쓸 수도 있음
# model.add(SimpleRNN(units=10, input_dim=1, input_length=3,)) # 이렇게 거꾸로 쓸 수도 있는데 가독성이 떨어져서 헷갈림
model.add(LSTM(, input_shape=(3, 1))) 
model.add(Dense(5, activation='relu')) 
model.add(Dense(1))
model.summary()
'''
model = Sequential()
model.add(LSTM(64, input_shape=(3, 1))) # SimpleRNN 에서 Dense로 넘어갈 때 자동적으로 2차원으로 던져줌
#               └> units           └> input_dim 
# model.add(SimpleRNN(10))
model.add(Dense(64, activation='relu')) # RNN의 아웃풋은 2차원 형태 (그래서 SimpleRNN에서는 RNN끼리 두개를 엮을 수 없음)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 10)                480         # <- 4*((input_shape_size +1) * ouput_node + output_node^2)
# _________________________________________________________________
# dense (Dense)                (None, 5)                 55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 541
# Trainable params: 541
# Non-trainable params: 0
# _________________________________________________________________

# [simple] units=10 -> 10 * (1 + 1 + 10) = 120
# [LSTM]   units=10 -> 4 * 10 * (1 + 1 + 10) = 480
#                20 -> 4 * 20 * (1 + 1 + 20) = 1760
# [GRU]          10 -> 3 * 10 * (1 + 1 + 10) = 360
# 결론 : LSTM = simpleRNN * 4
# 숫자 4의 의미 = cell state, input gate, output gate, forget gate 
# 결론 : GRU = simpleRNN * 3
# 숫자 4의 의미 = hidden state, reset gate, update gate

# 이건 RNN
# Model: "sequential"
# _________________________________________________________________ 
# Layer (type)                 Output Shape              Param #
# =================================================================
# simple_rnn (SimpleRNN)       (None, 10)                120          # <- Param = units * (units + input_dim + bias)      RNN에서는 units를 한번 더 더해줌 = 한번 더 돌려줌     그래서 param이 따블로 올라감
# _________________________________________________________________
# dense (Dense)                (None, 5)                 55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 181
# Trainable params: 181
# Non-trainable params: 0
# _________________________________________________________________


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=120, mode='min', verbose=1, restore_best_weights=True)
model.fit(x, y, epochs=3000, batch_size=1, callbacks=[es], verbose=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([8,9,10]).reshape(1, 3, 1) # [[[8], [9], [10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)

# loss :  3.644340904429555e-05
# [8,9,10]의 결과 :  [[10.804404]]

# loss :  7.521761290263385e-05
# [8,9,10]의 결과 :  [[10.907741]]

# loss :  1.7760978153091855e-05
# [8,9,10]의 결과 :  [[10.915004]]

# loss :  3.0987721402198076e-05
# [8,9,10]의 결과 :  [[10.921128]]

