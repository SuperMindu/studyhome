import numpy as np
from sklearn.datasets import load_wine
import pandas as pd
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
datasets = load_wine()
print(datasets.DESCR) # :Number of Instances: 178 (50 in each of three classes) y의 label 값이 3개

x = datasets.data
y = datasets.target
print(x, y)
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

y = to_categorical(y)
print(y) 
print(y.shape) # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델 구성 
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax')) 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
print('-------------------------------------')
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1) 
print(y_predict)
y_test = np.argmax(y_test, axis=1) 
print(y_test)
acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)

# acc 스코어 :  0.9444444444444444


