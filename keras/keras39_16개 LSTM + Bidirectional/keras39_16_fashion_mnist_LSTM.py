from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, LSTM
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.layers import BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)

# 2. 모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)       
start_time = time.time() 
model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.3, callbacks=[es], verbose=1)
end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)
print("걸린시간 : ", end_time)

# RTX 2080
# loss :  [0.42945802211761475, 0.8474000096321106]
# acc스코어 :  0.8474
# 걸린시간 :  1711.0845494270325 (약 28.5분)

# RTX 3090
# loss :  [0.43367519974708557, 0.842199981212616]
# acc스코어 :  0.8422
# 걸린시간 :  1350.7244834899902 (약 22.5분)

