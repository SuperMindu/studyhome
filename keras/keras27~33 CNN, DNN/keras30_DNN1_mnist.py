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
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D # 이미지 작업은 이거
from tensorflow.python.keras.layers import Flatten, MaxPooling2D, Input, Dropout
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical 

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28*28, )
x_test = x_test.reshape(10000, 28*28, )
print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)


# #2. 모델 구성
model = Sequential()

# # model.add(Dense(64, input_shape=(28*28, ))) 
# model.add(Dense(64, input_shape=(784, )))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))

input1 = Input(shape=(784,))
dense1 = Dense(128, activation='relu')(input1)
dense2 = Dense(128, activation='relu')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(128, activation='relu')(drop1)
drop2 = Dropout(0.1)(dense3)
dense4 = Dense(128, activation='relu')(drop2)
drop3 = Dropout(0.1)(dense4)
dense5 = Dense(128, activation='relu')(dense4)
output1 = Dense(10, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
print('-------------------------------------')
y_predict = model.predict(x_test)
y_test = np.argmax(y_test, axis=1) 
y_predict = np.argmax(y_predict, axis=1) 
acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)

# loss :  0.11865685135126114
# acc 스코어 :  0.9725
