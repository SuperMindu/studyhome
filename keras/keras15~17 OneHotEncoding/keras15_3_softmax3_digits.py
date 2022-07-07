import numpy as np
from sklearn.datasets import load_digits
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
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,) (8x8짜리 이미지가 1797장이 있음) (색의 진하기 농도에 따라 다름)
# y (1797,) 를 원핫인코딩을 써서 (1797, 10)으로 바꿔줘야 한다
print(np.unique(y, return_counts=True)) # [0 1 2 3 4 5 6 7 8 9] 0~9 까지의 숫자를 찾아라

y = to_categorical(y)
print(y) 
print(y.shape) # (1797, 10)

# 이미지 보는 방법
# plt.gray()  
# plt.matshow(datasets.images[0])
# plt.show


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델 구성 
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=64))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax')) 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.15, callbacks=[es])

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

# acc 스코어 :  0.975
