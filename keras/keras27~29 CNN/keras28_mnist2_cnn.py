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

# 원핫, 소프트맥스, 카테고리컬 !!!
# acc 0.98 이상
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
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), 
                 padding='same', # 패딩을 씌우면 커널 사이즈에 상관없이 원래 shape 그대로 감 (보통 0을 씌움)
                 input_shape=(28, 28, 1))) 
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2),
                 padding='same', # 이게 padding의 디폴트 값
                 activation='relu')) 
model.add(Conv2D(32, (2,2),
                 padding='valid', # 이게 padding의 디폴트 값
                 activation='relu'))
model.add(Flatten()) 
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) 


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