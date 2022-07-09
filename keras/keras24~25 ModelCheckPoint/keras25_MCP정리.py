'''
아직 미완성
나중에 시간날 때 나만의 저장방식을 정해서 튜닝 해보자
'''
import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.  모델 구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
import datetime
date = datetime.datetime.now()
print(date) # 2022-07-07 17:24:51.433145

date = date.strftime('%m%d_%H%M') 
print(date) # 0707_1724

filepath = './_ModelCheckPoint/k24'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#         {에포의 4자리}-{발로스의 소수점 4째자리} 라는 뜻

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, # save_weights_only = True 했을경우 load_weights해서 사용함 
                      filepath = "".join([filepath + 'k24_', date, '_', filename]) # 합쳐서 하나로 만들어줌
                      )
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es, mcp], verbose=1) 

# model.save('./_save/keras25_MCP정리.h5') # <-- 여기서 저장하는 값은 es 에서 주는 w값을 받아와서 저장함


#4. 평가 , 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)




