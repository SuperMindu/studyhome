from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import fetch_california_housing
import numpy as np
import time
from pathlib import Path


#1. 데이터
datasets = fetch_california_housing()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape) # (16512, 8) (4128, 8)
print(y_train.shape, y_test.shape) # (16512,) (4128,)
x_train = x_train.reshape(16512, 1, 8)
x_test = x_test.reshape(4128, 1, 8)

#2. 모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(1, 8)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto', verbose=1, restore_best_weights=True)        
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, validation_split=0.2, callbacks=[earlyStopping], verbose=1)

#4. 평가, 예측
print("=============================1. 기본 출력=================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score

print(y_test, y_predict)
print(y_test.shape, y_predict.shape)


r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

# loss :  0.25582000613212585
# r2스코어 :  0.8164935216709593

