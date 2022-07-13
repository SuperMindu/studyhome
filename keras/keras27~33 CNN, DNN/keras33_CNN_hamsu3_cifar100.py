import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(60000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

#2. 모델구성
input1 = Input(shape=(3072,))
dense1 = Dense(128)(input1)
dense2 = Dense(128, activation='relu')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(128, activation='relu')(drop1)
drop2 = Dropout(0.1)(dense3)
dense4 = Dense(64, activation='relu')(drop2)
drop3 = Dropout(0.1)(dense4)
dense5 = Dense(32, activation='relu')(drop3)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)        

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True)

model.fit(x_train, y_train, epochs=100, batch_size=1024,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)
# y_predict = to_categorical(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)