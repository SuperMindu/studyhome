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


# 3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, # save_weights_only = True 했을경우 load_weights해서 사용함
                      filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5' # 가중치 저장할 파일 경로 지정
                      )
# es에서 restore_bets_weights = True,False하든말든 mcp는 나름대로 save_best_only를 계속 수행하고 값을 저장함

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es, mcp], verbose=1) 

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  10.710644721984863
# r2 스코어 :  0.8718560785416583



