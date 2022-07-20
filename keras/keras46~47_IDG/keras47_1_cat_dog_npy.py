# 불러와서 모델링
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

# 1-1) 데이터 로드 
x_train = np.load('d:/study_data/_save/_npy/keras47_1_cat_dog_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras47_1_cat_dog_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_1_cat_dog_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras47_1_cat_dog_test_y.npy')

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (8005, 150, 150, 3) (8005, 2) (2023, 150, 150, 3) (2023, 2)

# 2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=50, validation_split=0.15, verbose=1) # 이렇게 하려면 배치사이즈를 최대로 잡아주면 됨
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# print('loss : ', loss) # 이렇게 하면 모든 로스가 다 나옴 
print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_accuracy : ', val_accuracy[-1])

# loss :  0.029801979660987854
# val_loss :  6.936742305755615
# accuracy :  0.9867724776268005
# val_accuracy :  0.5570358037948608

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


