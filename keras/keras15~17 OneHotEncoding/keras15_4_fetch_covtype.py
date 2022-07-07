import numpy as np
from sklearn.datasets import fetch_covtype
import pandas as pd
from pandas import get_dummies
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#1. 데이터 
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(datasets.DESCR)
# **Data Set Characteristics:**
#     =================   ============
#     Classes                        7
#     Samples total             581012
#     Dimensionality                54
#     Features                     int
#     =================   ============
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) 
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64)) 
# 나중에 피처를 건들 수 있으면 얼마 없는 값들은 증폭 시켜서 어쩌고 저쩌고 
y = pd.get_dummies(y)
print(y) # [581012 rows x 7 columns]
print(y.shape) # (581012, 7)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델 구성 
model = Sequential()
model.add(Dense(128, activation='linear', input_dim=54))
model.add(Dense(128, activation='linear'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='linear'))
model.add(Dense(32 , activation='relu'))
model.add(Dense(7, activation='softmax')) 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1, batch_size=8192, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
print('-------------------------------------')
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1) 
# print(y_predict)
y_test = tf.argmax(y_test, axis=1) 
# print(y_test)
acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)


# acc 스코어 :  0.8246000533549048

# loss :  0.34630516171455383
# acc 스코어 :  0.8574477423130212
# 집가서 중간 노드 128개로 늘려서 해보기

# loss :  0.2899523675441742
# acc 스코어 :  0.8828945896405428
