# 증폭해서 npy에 저장
# y의 유니크 값을 뽑아서 가장 큰 값에 맞춰서 증폭 숫자를 맞춰보자
from tkinter import Image
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator( # 증폭시킬 파라미터 설정
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

train_datagen2 = ImageDataGenerator( # rescale만 해주는 거 따로 설정
    rescale=1./255
)

augument_size = 40000 # 증폭 시킬 값 사이즈
batch_size = 64
randidx = np.random.randint(x_train.shape[0], size=augument_size) 


x_augumented = x_train[randidx].copy() 
y_augumented = y_train[randidx].copy() 
print(x_augumented.shape, y_augumented.shape) # (40000, 28, 28) (40000,)


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) 
print(x_test.shape) # (10000, 28, 28, 1)

x_augumented = x_augumented.reshape(x_augumented.shape[0], 
                                    x_augumented.shape[1], 
                                    x_augumented.shape[2], 1) # (40000, 28, 28, 1)

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

xy_train = train_datagen2.flow(x_train,y_train,
                               batch_size=augument_size,
                               shuffle=False)
print(xy_train[0][0].shape, x_test.shape, xy_train[0][1].shape, y_test.shape)

# np.save('d:/study_data/_save/_npy/keras49_1_x_train.npy', arr=xy_train[0][0]) 
# np.save('d:/study_data/_save/_npy/keras49_1_y_train.npy', arr=xy_train[0][1]) 
# np.save('d:/study_data/_save/_npy/keras49_1_x_test.npy', arr=x_test)
# np.save('d:/study_data/_save/_npy/keras49_1_y_test.npy', arr=y_test)


