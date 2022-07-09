from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성 
# model = Sequential()
# model.add(Dense(32, input_dim=8))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(1))
input1 = Input(shape=(8,))
dense1 = Dense(32)(input1)
drop1 = Dropout(0.15)(dense1)
dense2 = Dense(32)(drop1)
dense3 = Dense(32)(dense2)
drop2 = Dropout(0.15)(dense3)
dense4 = Dense(32)(drop2)
dense5 = Dense(32)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=100, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)
# acc = accuracy_score(y_test, y_predict) 
# print('acc 스코어 : ', acc)

#               노말                        MinMax                        Standard                         MaxAbs                         Robust                              
# loss : 0.6602479219436646          0.544556200504303              2.737544536590576               0.5575705766677856             1.5825440883636475                                                                                                                                                     
# r2 :   0.5188293167658404          0.6031424376903464            -0.9950482512218297              0.5936578681247111            -0.15331469740581483                                                                                                                                                        
# 함수형                             0.5424100160598755     
#                                    0.6047064258129948                                                                                                       
# Dropout                            0.5408299565315247             1.0221948623657227              0.5713874697685242             1.9277220964431763                                                                                                                                                                  
#                                    0.6058579410026159             0.25505232116002075             0.5835885129835018             -0.40487135380353867                                                                                                                                                           