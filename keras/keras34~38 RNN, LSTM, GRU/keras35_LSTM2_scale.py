import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping  

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print(x.shape, y.shape) # (13, 3) (13,)


x = x.reshape(13, 3, 1) # 
print(x.shape) # (13, 3, 1)


model = Sequential()
model.add(LSTM(512, input_shape=(3, 1))) # SimpleRNN 에서 Dense로 넘어갈 때 자동적으로 2차원으로 던져줌
#                └> units           └> input_dim 

model.add(Dense(256, activation='relu')) 
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=150, mode='min', verbose=1, restore_best_weights=True)
model.fit(x, y, epochs=1000, batch_size=1, callbacks=[es], verbose=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([50,60,70]).reshape(1, 3, 1) # [[[8], [9], [10]]]
print(y_pred.shape)
result = model.predict(y_pred)
print('loss : ', loss)
print('[50,60,70]의 결과 : ', result)

# loss :  0.0013679122785106301
# [50,60,70]의 결과 :  [[75.410995]]

# loss :  0.0177047960460186
# [50,60,70]의 결과 :  [[75.86878]]
