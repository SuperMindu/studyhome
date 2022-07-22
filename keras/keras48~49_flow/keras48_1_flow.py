from tkinter import Image
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# 분류 모델 에서는 비교군에 있는 데이터들의 개수가 맞지 않을 때 개수를 맞춰주는 식으로 증폭을 사용하면 효과가 좋음

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 100 # 증폭 

print(x_train[0].shape) # (28, 28)
print(x_train[0].reshape(28*28).shape) # (784,) 
print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1).shape) # (100, 28, 28, 1) <- 100개로 증폭됨 (augument_size = 100 이니까)
# np.tile(A, repeat_shape) 형태이며, A 배열이 repeat_shape 형태로 반복되어 쌓인 형태가 반환됨
# 일단 이것도 증폭의 일종임. 근데 똑같은 데이터로만 쌓아서 과적합의 우려가 있음(?)
print(np.zeros(augument_size))
print(np.zeros(augument_size).shape) # (100,) <- 0 이 들어간 갯수 

# 여기에 들어가기 전엔 다 똑같은 이미지임
x_data = train_datagen.flow( 
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1), # <- (100, 28, 28, 1) 똑같은 데이터가 들어왔음. 이게 x
    np.zeros(augument_size),  # <------------------------------------------------------------------------------------------ 이게 y 
    batch_size=augument_size, # 
    shuffle=True,                           
) .next() # <- ################################################################################### Iterator .next()이거를 쓴 이유와 지금 이 코드에 대해서 정의해서 쌤 메일로 보내기 (과제)
# print(x_data[0])           
# print(x_data[0][0].shape) 
# print(x_data[0][1].shape)
#              ↑ .next()를 써주면 이 라인을 건너 뜀

#################################### .next() 미사용 #############################################
# print(x_data) # <keras.preprocessing.image.NumpyArrayIterator object at 0x0000020400868700>
# print(x_data[0])          # x와 y가 모두 포함 
# print(x_data[0][0].shape) # (100, 28, 28, 1)
# print(x_data[0][1].shape) # (100,)

##################################### .next() 사용 ##############################################
print(x_data) # <keras.preprocessing.image.NumpyArrayIterator object at 0x0000020400868700>
print(x_data[0])       # x와 y가 배치사이즈 단위로 모두 포함 
print(x_data[0].shape) # (100, 28, 28, 1) 
print(x_data[1].shape) # (100,)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,7))
# for i in range(49):
#     plt.subplot(7, 7, i+1)
#     plt.axis('off')
#     plt.imshow(x_data[0][i], cmap='gray')      # <- .next() 사용 하려면 이렇게
#     # plt.imshow(x_data[0][0][i], cmap='gray') # <- .next() 미사용 하려면 이렇게
# plt.show()

