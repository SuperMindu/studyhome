from sklearn.datasets import load_wine
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


datasets = load_wine()
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
model.add(Dense(32, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=10, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
print('-------------------------------------')

y_predict = model.predict(x_test)
y_test = tf.argmax(y_test, axis=1) 
print(y_test)
y_predict = tf.argmax(y_predict, axis=1) 
print(y_predict)

acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)

#              노말                          MinMax                          Standard                         MaxAbs                            Robust                              
# loss : 0.211390882730484            0.1701735407114029               0.0987773984670639               0.18023325502872467              0.1570708453655243                                                                                                                                                   
# acc :  0.9444444444444444           0.9629629629629629               0.9629629629629629               0.9629629629629629               0.9629629629629629                                                                                                                                                   
#                                                                                                                  