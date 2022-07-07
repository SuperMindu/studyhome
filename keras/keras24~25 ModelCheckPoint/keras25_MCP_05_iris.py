from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from pandas import get_dummies

from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

datasets = load_iris()
x = datasets.data
y = datasets['target']
print(x)
print(y)
print(x.shape) # (150, 4)
print(y.shape) # (150,)
print('y의 label값 : ', np.unique(y))

y = get_dummies(y)
print(y) # [150 rows x 3 columns]          
print(y.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
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
# model.add(Dense(32, input_dim=4))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax'))
input1 = Input(shape=(4,))
dense1 = Dense(32, activation='relu')(input1)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
dense5 = Dense(32, activation='relu')(dense4)
output1 = Dense(3, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)

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

#               노말                           MinMax                        Standard                          MaxAbs                          Robust                              
# loss : 0.04376697912812233            0.14083729684352875             0.2813741862773895              0.06043783947825432              0.2209908813238144                                                                                                                                                      
# acc :  0.9777777777777777             0.9555555555555556              0.9333333333333333              0.9555555555555556               0.9333333333333333                                                                                                                                                   
#                                                                                                       0.09992893785238266  
#                                                                                                       0.9555555555555556        