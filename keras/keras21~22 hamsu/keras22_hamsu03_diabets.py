from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

datasets = load_diabetes()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train)) # 
# print(np.max(x_train)) # 
# print(np.min(x_test)) # 
# print(np.max(x_test))

#2. 모델 구성 
# model = Sequential()
# model.add(Dense(32, input_dim=10))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(1))
input1 = Input(shape=(10,))
dense1 = Dense(32)(input1)
dense2 = Dense(32)(dense1)
dense3 = Dense(32)(dense2)
dense4 = Dense(32)(dense3)
dense5 = Dense(32)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=10, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
# acc = accuracy_score(y_test, y_predict) 
# print('acc 스코어 : ', acc)

#                노말                        MinMax                        Standard                          MaxAbs                          Robust                              
# loss : 3322.849365234375            3139.490966796875                3039.01318359375                 3161.3798828125                3024.195068359375                                                                                                                                                
# r2 :   0.46666906731348756          0.49609881552670065              0.5122259038408296               0.49258555603261345            0.5146042927169433                                                                                                                                                       
#                                                                      3202.865478515625    
#                                                                      0.4859269891310529                                                                                                                                                                                                                         