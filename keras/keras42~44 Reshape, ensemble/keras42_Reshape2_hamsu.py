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
from tensorflow.python.keras.layers import Dense, Conv2D, Conv1D, LSTM, Reshape
from tensorflow.python.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical 

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data() # train과 test 자동으로 나눠짐
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# reshape 할 때는 모든 객체를 곱한 값이 같아야 함 (안에 있는 데이터는 건들지 않음)
# print(x_train.shape) # (60000, 28, 28, 1)

print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

# import matplotlib.pyplot as plt
# plt.imshow(x_train[5], 'gray')
# plt.show()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)
# (60000, 10) (10000, 10)

print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000, 10)
print(x_test.shape, y_test.shape) # (10000, 28, 28, 1) (10000, 10)

#2. 모델 구성
# model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=(3,3), 
#                  padding='same', input_shape=(28, 28, 1))) 
# model.add(MaxPooling2D())#                   (N, 14, 14, 64)
# model.add(Conv2D(32, (3,3)))#                (N, 12, 12, 32)
# model.add(Conv2D(7, (3,3)))#                 (N, 10, 10, 7)
# model.add(Flatten())#                        (N, 700)
# model.add(Dense(100, activation='relu'))#    (N, 100)
# model.add(Reshape(target_shape=(100, 1)))#   (N, 100, 1)
# # Dense로 진행하다가 중간에 Conv1D를 쓰고 싶으면 Conv1D는 3차원으로 입력을 받아야 해서 위에서 reshape를 해줘야 함
# model.add(Conv1D(10, kernel_size=3))#        (N, 98, 10)
# model.add(LSTM(16))#                         (N, 16)
# model.add(Dense(32, activation='relu'))#     (N, 32)
# model.add(Dense(10, activation='softmax'))#  (N, 10)
# model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 64)        640       
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 12, 12, 32)        18464     
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 10, 10, 7)         2023      
# _________________________________________________________________
# flatten (Flatten)            (None, 700)               0
# _________________________________________________________________
# dense (Dense)                (None, 100)               70100
# _________________________________________________________________
# reshape (Reshape)            (None, 100, 1)            0
# _________________________________________________________________
# conv1d (Conv1D)              (None, 98, 10)            40
# _________________________________________________________________
# lstm (LSTM)                  (None, 16)                1728
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                330
# =================================================================
# Total params: 93,869
# Trainable params: 93,869
# Non-trainable params: 0
# _________________________________________________________________

#2. 모델 구성 (위에거 함수형으로 똑같이 만들어보기)
input1 = Input(shape=(28, 28, 1)) # <- input에는 shape만 넣음
Conv2_1 = Conv2D(64, kernel_size=(3,3), padding='same')(input1)
MaxP1 = MaxPooling2D()(Conv2_1)
Conv2_2 = Conv2D(32, (3,3))(MaxP1)
Conv2_3 = Conv2D(7, (3,3))(Conv2_2)
Flat1 = Flatten()(Conv2_3)
dense1 = Dense(100, activation='relu')(Flat1)
Reshape1 = Reshape(target_shape=(100, 1))(dense1)
Conv1_1 = Conv1D(10, kernel_size=3)(Reshape1)
LSTM1 = LSTM(16)(Conv1_1)
dense2 = Dense(32, activation='relu')(LSTM1)
output1 = Dense(10, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1)
model.summary()
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 28, 28, 1)]       0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 28, 28, 64)        640
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 12, 12, 32)        18464
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 10, 10, 7)         2023
# _________________________________________________________________
# flatten (Flatten)            (None, 700)               0
# _________________________________________________________________
# dense (Dense)                (None, 100)               70100
# _________________________________________________________________
# reshape (Reshape)            (None, 100, 1)            0
# _________________________________________________________________
# conv1d (Conv1D)              (None, 98, 10)            40
# _________________________________________________________________
# lstm (LSTM)                  (None, 16)                1728
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                330
# =================================================================
# Total params: 93,869
# Trainable params: 93,869
# Non-trainable params: 0
# _________________________________________________________________



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
print('-------------------------------------')
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)

# loss :  0.07453285157680511
# accuracy :  0.9832000136375427
