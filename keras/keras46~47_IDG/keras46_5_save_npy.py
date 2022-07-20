# 이미지 개수가 많아지면 불러올 때마다 로딩 시간이 너무 길어짐
# 그래서 데이터를 수치화 시켜서 저장을 해놓아야 함

import numpy as np
from keras.preprocessing.image import ImageDataGenerator


# 1. 데이터
train_datagen = ImageDataGenerator( # 아래의 모든건 컴터가 알아서 몇개를 적용할지 다 랜덤으로 적용됨, 증폭가능. 밑에거를 쓸지 말지는 나의 선택
    rescale=1./255, # MinMax 해주겠다는 것과 같음 (스케일링 기능임)
    # horizontal_flip=True, # 수평 반전
    # vertical_flip=True, # 수직 반전
    # width_shift_range=0.1, # 가로 수평 이동 0.1 (10%)
    # height_shift_range=0.1, # 상하 이동 
    # rotation_range=5, # 돌려돌려돌림판
    # zoom_range=1.2, # 확대 
    # shear_range=0.7, # 기울기? 찌그러트리기? 비틀기? 짜부? 
    # fill_mode='nearest'
) # 트레인 데이터 준비만 한거임. 선언만 한거임

test_datagen = ImageDataGenerator(
    rescale=1./255 # 테스트 데이터는 증폭하면 안됨
)

xy_train = train_datagen.flow_from_directory('d:/study_data/_data/image/brain/train/', # 폴더에서 가져와서 ImageDataGenerator
                                             target_size=(150, 150), # 이미지 크기 조절. 고르지 않은 크기들을 사이즈를 지정해줌. 내 맘대로 가능
                                             batch_size=500,
                                             class_mode='binary', # 0, 1 분류. 이진분류. (3가지 이상은 categorical)
                                             color_mode='grayscale', # 이걸 따로 지정해주지 않으면 디폴트값은 컬러로 나옴. 밑에 print(xy_train[31][0].shape) 참고
                                             shuffle=True,
                                             ) 
# Found 160 images belonging to 2 classes. flow_from_directory를 통과했을 때 160개의 이미지와 2개의 클래스가 됐음

xy_test = test_datagen.flow_from_directory('d:/study_data/_data/image/brain/test/', 
                                             target_size=(150, 150), 
                                             batch_size=500,
                                             class_mode='binary', 
                                             color_mode='grayscale',
                                             shuffle=True,
                                             ) 
# Found 120 images belonging to 2 classes.
# print(xy_train[0][0])
# print(xy_train[0][1]) # <- 원핫인코딩까지 돼서 0 1 로 나옴

print(xy_train[0][0].shape, xy_train[0][1].shape) # (160, 150, 150, 1) (160,)
print(xy_test[0][0].shape, xy_test[0][1].shape) # (120, 150, 150, 1) (120,) 
# (5, 150, 150, 1) 짜리 데이터가 32덩어리 
np.save('d:/study_data/_save/_npy/keras46_5_train_x.npy', arr=xy_train[0][0]) # <- xy
np.save('d:/study_data/_save/_npy/keras46_5_train_y.npy', arr=xy_train[0][1]) # <- xy
np.save('d:/study_data/_save/_npy/keras46_5_test_x.npy', arr=xy_test[0][0]) # <- xy
np.save('d:/study_data/_save/_npy/keras46_5_test_y.npy', arr=xy_test[0][1]) # <- 넘파이 수치로 변환해서 저장시켜줌 이렇게 하는게 훨씬 빠름


'''
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
                    ) # 이 fit_generator은 앞으로의 버전에서는 삭제된다고 함. model.fit을 쓰자

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
plt.figure(figsize=(9,6)) # (9 x 6 으로 그리겠다는 뜻) (figure <- 이거 알아서 찾아보기)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # marker '_' 로 하면 선, 아예 삭제해도 선으로 이어짐
plt.plot(hist.history['accuracy'], marker='.', c='blue', label='accuracy')
plt.plot(hist.history['val_loss'], marker='.', c='yellow', label='val_loss')
plt.plot(hist.history['val_accuracy'], marker='.', c='green', label='val_accuracy')
plt.grid() # 보기 편하게 하기 위해서 모눈종이 형태로 
plt.title('SuperMindu') # 표 제목 
plt.ylabel('loss') # y축 이름 설정
plt.xlabel('epochs') # x축 이름 설정 
plt.legend(loc='upper right')  # 위에 label 값이 이 위치에 명시가 된다 (디폴트 값이 upper right)
plt.show()
'''







