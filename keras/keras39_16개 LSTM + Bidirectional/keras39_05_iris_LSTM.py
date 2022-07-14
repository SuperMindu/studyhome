from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_iris
import numpy as np
import time
from pathlib import Path
from tensorflow.keras.utils import to_categorical


#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (150, 4) (150,)


print(np.unique(y, return_counts=True)) 
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape) # (120, 4) (30, 4)
print(y_train.shape, y_test.shape) # (120,) (30,)
print(x_train)
'''
x_train = x_train.reshape(120, 1, 4)
x_test = x_test.reshape(30, 1, 4)


#2. 모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(1, 4)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto', verbose=1, restore_best_weights=True)        
hist = model.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[earlyStopping], verbose=1)

#4. 평가, 예측
print("=============================1. 기본 출력=================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, accuracy_score
y_predict = to_categorical(y_predict)
y_predict = np.argmax(y_predict, axis= 1)
# print(y_test, y_predict)
# print(y_test.shape, y_predict.shape)


acc= accuracy_score(y_test, y_predict)
print('loss : ' , loss)
print('acc스코어 : ', acc) 
'''