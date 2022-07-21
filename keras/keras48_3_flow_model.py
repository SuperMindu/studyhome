from tkinter import Image
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np
# 분류 모델 에서는 비교군에 있는 데이터들의 개수가 맞지 않을 때 개수를 맞춰주는 식으로 증폭을 사용하면 효과가 좋음

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 40000
#                           ┌> (여기의 정수값 사이에서, 여기의 사이즈 만큼의 갯수를 빼겠다) 0~59999 사이에서 40000개를 빼겠다
randidx = np.random.randint(x_train.shape[0], size=augument_size) # np.random.randint 는 정수 값만을 return 하는 함수임. 랜덤하게 정수 값 만을 넣음
#                           └>(x_train.shape가 (60000, 28, 28, )이니까 x_train.shape[0] 하면 60000이, [1] 하면 28이, [2] 하면 28이 나옴. [3]은 아직 안나옴. reshape 해줘야됨)
print(x_train.shape[0]) # 60000
print(randidx) # [26913  4351 39125 ... 45427 27116 18645]
print(np.min(randidx), np.max(randidx)) # 0 59997
print(type(randidx)) # <class 'numpy.ndarray'>

x_augumented = x_train[randidx].copy() # <- 무작위로 40000개를 뽑기 위해서 이렇게 해줌. .copy()해주면 좀 더 안정적임. (다른 공간으로 따로 빼주는 개념. 같이 있으면 나중에 꼬일 수도 있어서?)
y_augumented = y_train[randidx].copy() # x 와 y의 개수가 같아야 함(?)
print(x_augumented.shape, y_augumented.shape) # (40000, 28, 28) (40000,) # 얘는 완전 그냥 copy본임. 실제 모델에 집어 넣으려면 좀 ...

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) # <- 이제는 뭔 말인지 알지? ㅎㅎ

x_augumented = x_augumented.reshape(x_augumented.shape[0], 
                                    x_augumented.shape[1], 
                                    x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0] # <- x만 뺌
# 이렇게 x_augumented 하나만 정의해서 안에 y_augumented도 넣어주고  x만 빼면 y는 자동으로 빼지는 듯 

print(x_augumented)
print(x_augumented.shape) # (40000, 28, 28, 1) <- 이제는 증폭돼서 변환이 된걸 확인할 수 있음

x_train = np.concatenate((x_train, x_augumented)) # <- 클래스라서 괄호가 2개. concatenate는 괄호 2개 형식으로 제공. 왜 괄호가 2개가 되는지도 찾자 (과제)
y_train = np.concatenate((y_train, y_augumented))
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)
# print(x_test.shape, y_test.shape)
# print(x_test, y_test)
# print(x_test, y_train, y_test)

# 이렇게 증폭을 해줄 때는 test는 건들면 안됨! 

# 위에 flow로 빼줄때 x값만 빼줬기 때문에 y_train과 y_test도 원핫 인코딩을 해줘야 함
from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape)


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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=40, mode='auto', restore_best_weights=True, verbose=1)
hist = model.fit(x_train, y_train, epochs=300, batch_size=128, validation_split=0.1, callbacks=[es] ,verbose=1) # 이렇게 하려면 배치사이즈를 최대로 잡아주면 됨
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
y_test = np.argmax(y_test, axis=1) 
y_predict = np.argmax(y_predict, axis=1) 
acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)

# loss :  0.1881115883588791
# acc 스코어 :  0.8773



