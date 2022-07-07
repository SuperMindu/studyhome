from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from pandas import get_dummies

datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

y = get_dummies(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train)) # 
# print(np.max(x_train)) # 
# print(np.min(x_test)) # 
# print(np.max(x_test))

#2. 모델 구성 
# model = Sequential()
# model.add(Dense(64, input_dim=54))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(7, activation='softmax'))
input1 = Input(shape=(54,))
dense1 = Dense(128, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
dense5 = Dense(32, activation='relu')(dense4)
output1 = Dense(7, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=8192, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
print('-------------------------------------')

y_predict = model.predict(x_test)
y_test = tf.argmax(y_test, axis=1) 
# print(y_test)
y_predict = tf.argmax(y_predict, axis=1) 
# print(y_predict)

acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)

#               노말                        MinMax                        Standard                          MaxAbs                            Robust                              
# loss : 0.435060977935791            0.33165842294692993           0.2383122593164444               0.36211922764778137                0.22893892228603363                                                                                                                                                      
# acc :  0.8156554066458601           0.8631069854966036            0.9058426656875345               0.8497165871121718                 0.9088489076555902                                                                                                                                                    
#                                                                                                                                       0.23388461768627167 
#                                                                                                                                       0.9080743987516063
