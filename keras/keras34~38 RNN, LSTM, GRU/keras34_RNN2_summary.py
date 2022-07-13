import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping  

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
model.add(SimpleRNN(100, input_shape=(3, 1))) # [batch, timesteps, feature] (timesteps만큼 잘라서 feature)
#                    └> units           └> input_dim 
model.add(Dense(5, activation='relu')) 
model.add(Dense(1))
model.summary()

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

'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=430, mode='min', verbose=1, restore_best_weights=True)
model.fit(x, y, epochs=3000, batch_size=1, callbacks=[es], verbose=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([8,9,10]).reshape(1, 3, 1) # [[[8], [9], [10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)
'''
