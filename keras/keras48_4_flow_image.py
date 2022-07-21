from tkinter import Image
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
# 분류 모델 에서는 비교군에 있는 데이터들의 개수가 맞지 않을 때 개수를 맞춰주는 식으로 증폭을 사용하면 효과가 좋음
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

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
# [실습]
# x_augumented 10개와 x_train 10개를 비교하는 이미지 출력할 것 !!!
# 10개의 무작위 데이터를 뽑아서 augument 시켜주는 건 어렵지 않은데 
# 문제는 그 10개 무작위 데이터의 원본값을 어떻게 찾지?
# 그냥 print(randidx) 하면 원본 값의 인덱스가 나오긴 하는데 
# 또 문제는 돌릴 때 마다 랜덤하게 나올텐데 


augument_size = 10
randidx = np.random.randint(x_train.shape[0], size=augument_size)
print(randidx)

x_augumented = x_train[randidx].copy()
randidx = randidx.reshape(10, 28, 28, 1)
augumented = train_datagen.flow(x_augumented,
                                 batch_size=augument_size,
                                 shuffle=False)

# augument_size = 40000
# #                           ┌> (여기의 정수값 사이에서, 여기의 사이즈 만큼의 갯수를 빼겠다) 0~59999 사이에서 40000개를 빼겠다
# randidx = np.random.randint(x_train.shape[0], size=augument_size) # np.random.randint 는 정수 값만을 return 하는 함수임. 랜덤하게 정수 값 만을 넣음
# #                           └>(x_train.shape가 (60000, 28, 28, )이니까 x_train.shape[0] 하면 60000이, [1] 하면 28이, [2] 하면 28이 나옴. [3]은 아직 안나옴. reshape 해줘야됨)
# print(x_train.shape[0]) # 60000
# print(randidx) # [26913  4351 39125 ... 45427 27116 18645]
# print(np.min(randidx), np.max(randidx)) # 0 59997
# print(type(randidx)) # <class 'numpy.ndarray'>

# x_augumented = x_train[randidx].copy() # <- 무작위로 40000개를 뽑기 위해서 이렇게 해줌. .copy()해주면 좀 더 안정적임. (다른 공간으로 따로 빼주는 개념. 같이 있으면 나중에 꼬일 수도 있어서?)
# y_augumented = y_train[randidx].copy() # x 와 y의 개수가 같아야 함(?)
# print(x_augumented.shape, y_augumented.shape) # (40000, 28, 28) (40000,) # 얘는 완전 그냥 copy본임. 실제 모델에 집어 넣으려면 좀 ...

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) # <- 이제는 뭔 말인지 알지? ㅎㅎ

# x_augumented = x_augumented.reshape(x_augumented.shape[0], 
#                                     x_augumented.shape[1], 
#                                     x_augumented.shape[2], 1)

# x_augumented = train_datagen.flow(x_augumented, y_augumented,
#                                   batch_size=augument_size,
#                                   shuffle=False).next()[0] # <- x만 뺌
# # 이렇게 x_augumented 하나만 정의해서 안에 y_augumented도 넣어주고  x만 빼면 y는 자동으로 빼지는 듯 

# print(x_augumented)
# print(x_augumented.shape) # (40000, 28, 28, 1) <- 이제는 증폭돼서 변환이 된걸 확인할 수 있음

# x_train = np.concatenate((x_train, x_augumented)) # <- 클래스라서 괄호가 2개. concatenate는 괄호 2개 형식으로 제공
# y_train = np.concatenate((y_train, y_augumented))
# print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)


# [실습]
# x_augumented 10개와 x_train 10개를 비교하는 이미지 출력할 것
# plt.figure(figsize=(10,10))


# plt.subplot(2, 1, 1)                # nrows=2, ncols=1, index=1
# plt.plot(randidx)
# plt.title('원래 이미지')
# # plt.ylabel('Damped oscillation')

# plt.subplot(2, 1, 2)                # nrows=2, ncols=1, index=2
# plt.plot(x_augumented)
# plt.title('증폭 이미지')
# # plt.xlabel('time (s)')
# # plt.ylabel('Undamped')

# plt.tight_layout()



plt.imshow(randidx, 'gray')
# plt.imshow(augumented, 'gray')
plt.show()





