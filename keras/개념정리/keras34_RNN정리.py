# 시계열
# 보통 시계열 데이터는 y값이 없음
# 그래서 데이터를 받자마자 x와 y의 값을 나눠줘야 함 (데이터를 계속 중첩시켜 가면서 나눠줌)
# 데이터를 나눠줬을 때, 그 나눠준 순서가 다음 나눠주는 순서에 영향을 끼치는가? 
# 그 순서가 영향을 끼치게끔 규칙성을 만들어 줌
# https://gruuuuu.github.io/machine-learning/lstm-doc/ <-- 여기도 참고 해보자

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN # RNN -> Recurrent 뉴럴 네트워크
from tensorflow.python.keras.callbacks import EarlyStopping  

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) # array는 행렬을 의미 모든 연산은 numpy로 함
# y = ?

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4,5,6,7,8,9,10])
# (N, 3, 1) -> (행, 열, 데이터를 자르는 단위) 그래서 RNN은 shape가 3차원임
# x.reshape = (행, 열, 몇개씩 자르는지!) (batch, timesteps, feature)  근데 어차피 여기서의 배치값은 데이터 원래의 길이에서 timesteps 있으니까 알아서 구해짐
print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(7, 3, 1) # (7, 3, 1)
print(x.shape)

#2. 모델구성
model = Sequential() #               ┌> input_length
model.add(SimpleRNN(10, input_shape=(3, 1))) # [batch, timesteps, feature] (timesteps만큼 잘라서 feature만큼 슬라이스) (input_shape에 행은 넣지 않음. row값)
#                    └> units           └> input_dim 
# model.add(SimpleRNN(units=10, input_length=3, input_dim=1)) # 위에거를 이렇게 바꿔서 쓸 수도 있음
# model.add(SimpleRNN(units=10, input_dim=1, input_length=3,)) # 이렇게 거꾸로 쓸 수도 있는데 가독성이 떨어져서 헷갈리니까 이런짓은 하지말자
# SimpleRNN 에서는 Dense로 넘어갈 때 자동적으로 2차원으로 던져줌
model.add(Dense(5, activation='relu')) 
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


# Model: "sequential"
# _________________________________________________________________ 
# Layer (type)                 Output Shape              Param #
# =================================================================                   ┌> (보통 units 에는 정수의 양수값을 넣음)
# simple_rnn (SimpleRNN)       (None, 10)                120  # <- Param = units * (units + input_dim + bias)      RNN에서는 units를 한번 더 더해줌 (한번 더 돌려줌)    그래서 param이 따블로 올라감
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
