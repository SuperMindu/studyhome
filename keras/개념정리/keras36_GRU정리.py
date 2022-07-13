import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping  
# LSTM과 GRU의 차이점

# GRU는 LSTM과 다르게 Gate가 2개이며, Reset Gate(r)과 Update Gate(z)로 이루어져있다.
# Reset Gate는 이전 상태를 얼마나 반영할지
# Update Gate는 이전 상태와 현재 상태를 얼마만큼의 비율로 반영할지
# 또한, LSTM에서의 Cell State와 Hidden State가 Hidden State로 통합되었고
# Update Gate가 LSTM에서의 forget gate, input gate를 제어한다.
# GRU에는 Output Gate가 없다.
# Reset Gate

# 이전 상태의 hidden state와 현재 상태의 x를 받아 sigmoid 처리
# 이전 hidden state의 값을 얼마나 활용할 것인지에 대한 정보
# Update Gate

# 이전 상태의 hidden state와 현재 상태의 x를 받아 sigmoid 처리
# LSTM의 forget gate, input gate와 비슷한 역할을 하며,
# 이전 정보와 현재 정보를 각각 얼마나 반영할 것인지에 대한 비율을 구하는 것이 핵심이다.
# 즉, update gate의 계산 한 번으로 LSTM의 forget gate + input gate의 역할을 대신할 수 있다.
# 따라서, 최종 결과는 다음 상태의 hidden state로 보내지게 된다.



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
model.add(GRU(64, input_shape=(3, 1))) # SimpleRNN 에서 Dense로 넘어갈 때 자동적으로 2차원으로 던져줌
#              └> units           └> input_dim 
model.add(Dense(64, activation='relu')) # RNN의 아웃풋은 2차원 형태 (그래서 SimpleRNN에서는 RNN끼리 두개를 엮을 수 없음)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()



# summary에서 Param 구하는 방법
# [simple] units=10 -> 10 * (1 + 1 + 10) = 120
# [LSTM]   units=10 -> 4 * 10 * (1 + 1 + 10) = 480
#                20 -> 4 * 20 * (1 + 1 + 20) = 1760
# [GRU]          10 -> 3 * 10 * (1 + 1 + 10) = 360
# 결론 : LSTM = simpleRNN * 4
# 숫자 4의 의미 = cell state, input gate, output gate, forget gate 
# 결론 : GRU = simpleRNN * 3
# 숫자 4의 의미 = hidden state, reset gate, update gate



'''
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
'''