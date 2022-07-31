import numpy as np
from keras.preprocessing.image import ImageDataGenerator, image
from sklearn.model_selection import train_test_split



x_train = np.load("d:/PP1_x_train.npy")
x_test = np.load("d:/PP1_x_test.npy")
y_train = np.load("d:/PP1_y_train.npy")
y_test = np.load("d:/PP1_y_test.npy")


# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(255, 255, 3), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(46, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=100, mode='auto', restore_best_weights=True, verbose=1)
hist = model.fit(x_train, y_train, epochs=500, batch_size=8, validation_split=0.1, callbacks=[es] ,verbose=1) # 이렇게 하려면 배치사이즈를 최대로 잡아주면 됨

model.save('D:/PP1_model.h5')

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# print('loss : ', loss) # 이렇게 하면 모든 로스가 다 나옴 
print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_accuracy : ', val_accuracy[-1])

