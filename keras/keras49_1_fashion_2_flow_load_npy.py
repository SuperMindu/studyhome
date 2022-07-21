# 넘파이에서 불러와서 모델 구성 
# 성능 비교
from tkinter import Image
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

xy_train = np.load('d:/study_data/_save/_npy/keras46_5_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras46_5_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras46_5_test_y.npy')

# 2. 모델 구성 하고 성능 비교 
# 증폭 전 후 비교
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
model = Sequential()
model.add(Conv2D(128, (2,2), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=40, mode='auto', restore_best_weights=True, verbose=1)

hist = model.fit_generator( 
                    xy_train, epochs=100, 
                    # steps_per_epoch=32, # <- 데이터 셋을 배치사이즈로 나눈 덩어리값. 보통 이 값을 넣어줌.  전체데이터/batch = 160/5 = 32. 걍 확인차의 의미임
                    # validation_steps=4                       # steps_per_epoch가 특정된 경우에만 유의미함. 정지 전 검증할 단계(샘플 배치)의 총 개수
                    )
# accuracy = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# # print('loss : ', loss) # 이렇게 하면 모든 로스가 다 나옴 
# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
# print('accuracy : ', accuracy[-1])
# print('val_accuracy : ', val_accuracy[-1])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
print('-------------------------------------')
y_predict = model.predict(x_test)
# y_test = np.argmax(y_test, axis=1) 
# y_predict = np.argmax(y_predict, axis=1) 
acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)