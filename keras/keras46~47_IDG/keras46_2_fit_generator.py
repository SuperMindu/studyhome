import numpy as np
from keras.preprocessing.image import ImageDataGenerator


# 1. 데이터
train_datagen = ImageDataGenerator( # 아래의 모든건 다 랜덤으로 적용됨, 증폭가능
    rescale=1./255, # MinMax 해주겠다는 것과 같음 (스케일링 기능임)
    horizontal_flip=True, # 수평 반전
    vertical_flip=True, # 수직 반전
    width_shift_range=0.1, # 가로 수평 이동 0.1 (10%)
    height_shift_range=0.1, # 상하 이동 
    rotation_range=5, # 돌려돌려돌림판
    zoom_range=1.2, # 확대 
    shear_range=0.7, # 기울기? 찌그러트리기? 비틀기? 짜부? 
    fill_mode='nearest'
) # 트레인 데이터 준비만 한거임. 선언만 한거임

test_datagen = ImageDataGenerator(
    rescale=1./255 # 테스트 데이터는 증폭하면 안됨
)

xy_train = train_datagen.flow_from_directory('d:/_data/image/brain/train/', # 폴더에서 가져와서 ImageDataGenerator
                                             target_size=(150, 150), # 이미지 크기 조절. 고르지 않은 크기들을 사이즈를 지정해줌. 내 맘대로 가능
                                             batch_size=5,
                                             class_mode='binary', # 0, 1 분류. 이진분류. (3가지 이상은 categorical)
                                             color_mode='grayscale', # 이걸 따로 지정해주지 않으면 디폴트값은 컬러로 나옴. 밑에 print(xy_train[31][0].shape) 참고
                                             shuffle=True,
                                             ) 
# Found 160 images belonging to 2 classes. flow_from_directory를 통과했을 때 160개의 이미지와 2개의 클래스가 됐음

xy_test = test_datagen.flow_from_directory('d:/_data/image/brain/test/', 
                                             target_size=(150, 150), 
                                             batch_size=5,
                                             class_mode='binary', 
                                             color_mode='grayscale',
                                             shuffle=True,
                                             ) 
# Found 120 images belonging to 2 classes.

# (5, 200, 200, 1) 짜리 데이터가 32덩어리 

# 2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten


model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(150, 150, 1), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(xy_train[0][0], xy_train[0][1]) 이렇게 하려면 배치사이즈를 최대로 잡아주면 됨
hist = model.fit_generator(xy_train, epochs=10, steps_per_epoch=32, # <- 데이터 셋을 배치사이즈로 나눈 덩어리값. 보통 이 값을 넣어줌. 여기엔 따로 배치 사이즈가 없음. 전체데이터/batch = 160/5 = 32
                    validation_data=xy_test, # <- 근데 여기서는 발리데이션을 test 데이터로만 해서 ... 
                    validation_steps=4 # steps_per_epoch가 특정된 경우에만 유의미함. 정지 전 검증할 단계(샘플 배치)의 총 개수
                    )
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# print('loss : ', loss) # 이렇게 하면 모든 로스가 다 나옴 
print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_accuracy : ', val_accuracy[-1])

import matplotlib.pyplot as plt
# plt.imshow(xy_train([5,200,200,1]), 'gray')
# plt.show()

# plt.scatter(xy_train) # scatter 흩뿌리다(?) 그림처럼 보여주다(?) 점찍다
plt.plot(loss, color='red') # plot 선을 보여준다
plt.show()







