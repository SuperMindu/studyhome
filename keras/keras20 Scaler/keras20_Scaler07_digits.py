from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping
from pandas import get_dummies
import tensorflow as tf

datasets = load_digits()
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
model = Sequential()
model.add(Dense(32, input_dim=64))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.15, callbacks=[es])

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

#                노말                       MinMax                        Standard                          MaxAbs                            Robust                              
# loss : 0.11248809099197388         0.19322744011878967            0.17672935128211975              0.1701107919216156                0.1644802689552307                                                                                                                                                      
# acc :  0.9648148148148148          0.9555555555555556             0.9574074074074074               0.9481481481481482                0.9555555555555556                                                                                                                                                     
#                                                                                                                 