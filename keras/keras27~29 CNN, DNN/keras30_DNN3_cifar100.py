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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D # 이미지 작업은 이거
from tensorflow.python.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.keras.utils import to_categorical 

#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)


x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)


# #2. 모델 구성
model = Sequential()

# # model.add(Dense(64, input_shape=(28*28, ))) 
model.add(Dense(64, input_shape=(3072, )))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
print('-------------------------------------')
y_predict = model.predict(x_test)
y_test = np.argmax(y_test, axis=1) 
# print(y_test)
y_predict = np.argmax(y_predict, axis=1) 
# print(y_predict)
acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)

