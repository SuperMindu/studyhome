import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping  
# LSTM = Long Short-Term Memory Network
# RNN의 문제점 : 장기 의존성(데이터를 많이 자르면 앞쪽의 데이터는 영향력이 미미해짐), 기울기(weight) 손실
# LSTM은 RNN의 문제를 셀상태 (Cell State)와 여러개의 Gate를 가진 셀 이라는 유닛을 통해 해결함
# LSTM은 학습하면서 이전의 학습을 잊어버리지 않게하는 기능이 있음
# 셀 상태는 기존 신경망의 히든레이어라고 생각할 수 있음
# 셀 상태를 갱신하기 위해 기본적으로 3가지의 게이트가 필요함 (input, output, forget)
# sigmoid와 tanh를 적절히 사용

# Forget, input, output 게이트
# Forget : 이전 단계의 셀 상태를 얼마나 기억할 지 결정함. 0(모두 잊음)과 1(모두 기억) 사이의 값을 가짐
# input : 새로운 정보의 중요성에 따라 얼마나 반영할지 결정
# output : 셀 상태로부터 중요도에 따라 얼마나 출력할지 결정함
# 게이트는 가중치를 가진 은닉층으로 생각할 수 있음. 각 가중치는 sigmoid층에서 갱신되며 0과 1사이의 값을 가짐
# 이 값에 따라 입력되는 값을 조절하고, 오차에 의해 각 단계(time step)에서 갱신됨

# activation='tanh' Function
# sigmoid fuction을 보완하고자 나온 함수. 입력신호를 (−1,1) 사이의 값으로 normalization 해줌.
# 거의 모든 방면에서 sigmoid보다 성능이 좋음.
# 수식 : tanh(x) = e^x - e^-x / e^x + e^-x
#      d/dx tanh(x) = 1-tanh(x)^2


#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# y = ?

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) # (7, 3) (7,)
x = x.reshape(7, 3, 1) # (7, 3, 1)
print(x.shape) # (7, 3, 1)

#2. 모델구성
model = Sequential() #          ┌> input_length
model.add(LSTM(64, input_shape=(3, 1))) # input_shape는 행 빼고 들어가서 형태는 -1차원이 됨 (reshape해서 x.shape가 3차원 이었으니까 input_shape는 2차원)
#               └> units           └> input_dim 

# model.add(LSTM(10, return_sequences=True, input_shape=(3, 1))) # (N, 3, 1) -> (N, 3, 10) # return_sequences=True로 하면 아웃풋을 3차원으로 해서 던져줌 (차원이 하나 더 늘어나는 개념)
# 연속적으로 LSTM을 쓰려면 또 3차원의 데이터를 넣어줘야함                                                        
# 그걸 도와주는 옵션이 return_sequence 옵션이다 True설정하면 다음레이어에 3차원값을 그대로 줌                                    
# 그런데 연속적으로 해봤을때 좋은게 없다. 시계열 데이터를 RNN연산해서 나오는 output값은 연산을 많이 거쳐서 나온 값이므로
# 나온값들은 시계열데이터라고 보기엔 좀 힘듦. 그 특성이 많이 희석되어져서 나오기때문에 많은 기대를 하기 어려움

model.add(Dense(64, activation='relu')) # activation='tanh' 가 물론 좋긴 하지만 고정적으로 쓰진 않음
model.add(Dense(32, activation='relu')) # Dense는 주는 값을 그대로 받긴하지만 어차피 출력은 2차원으로 해야해서 언젠간 flatten으로 데이터를 펴줘야함
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))



# summary에서 Param 구하는 방법
# [simple] units=10 -> 10 * (1 + 1 + 10) = 120
# [LSTM]   units=10 -> 4 * 10 * (1 + 1 + 10) = 480
#                20 -> 4 * 20 * (1 + 1 + 20) = 1760
# [GRU]          10 -> 3 * 10 * (1 + 1 + 10) = 360
# 결론 : LSTM = simpleRNN * 4
# 숫자 4의 의미 = cell state, input gate, output gate, forget gate 
# 결론 : GRU = simpleRNN * 3
# 숫자 4의 의미 = hidden state, reset gate, update gate



#3. 컴파일, 훈련      ┌> mae도 있음 ㅎ
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=120, mode='min', verbose=1, restore_best_weights=True)
model.fit(x, y, epochs=3000, batch_size=1, callbacks=[es], verbose=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([8,9,10]).reshape(1, 3, 1) # [[[8], [9], [10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)

