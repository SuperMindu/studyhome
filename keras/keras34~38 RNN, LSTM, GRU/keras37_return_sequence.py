import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping  

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print(x.shape, y.shape) # (13, 3) (13,)


x = x.reshape(13, 3, 1) # 
print(x.shape) # (13, 3, 1)


# model = Sequential()
# model.add(GRU(512, input_shape=(3, 1))) # SimpleRNN 에서 Dense로 넘어갈 때 자동적으로 2차원으로 던져줌
# #               └> units           └> input_dim 

# model.add(Dense(256, activation='relu')) 
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1))
model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(3, 1))) # (N, 3, 1) -> (N, 3, 10) # return_sequences=True로 하면 아웃풋을 3차원으로 해서 던져줌 (차원이 하나 더 늘어나는 개념)
model.add(LSTM(5))
model.add(Dense(1))
model.summary()



# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 3, 10)             480
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 5)                 320
# _________________________________________________________________
# dense (Dense)                (None, 1)                 6
# =================================================================
# Total params: 806
# Trainable params: 806
# Non-trainable params: 0
# _________________________________________________________________


'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
model.fit(x, y, epochs=700, batch_size=1, callbacks=[es], verbose=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([50,60,70]).reshape(1, 3, 1) # [[[8], [9], [10]]]
print(y_pred.shape)
result = model.predict(y_pred)
print('loss : ', loss)
print('[50,60,70]의 결과 : ', result)

# loss :  0.025190262123942375
# [50,60,70]의 결과 :  [[76.30517]]

# loss :  0.006704407278448343
# [50,60,70]의 결과 :  [[76.67692]]
'''