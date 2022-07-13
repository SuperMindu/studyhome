# 이건 컬러, 분류가 100개 
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
from tensorflow.keras.utils import to_categorical
import numpy as np

#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True)) # 0부터 99까지 각 500개씩 100개의 label 값
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    #    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    #    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    #    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    #    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), 
    #    array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))
    
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3))) 
model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax')) 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
print('-------------------------------------')
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)



