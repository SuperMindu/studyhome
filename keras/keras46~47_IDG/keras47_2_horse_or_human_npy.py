# 불러와서 모델링
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping


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

x_train = np.load('d:/study_data/_save/_npy/keras47_2_horse_human_x_train.npy')
y_train = np.load('d:/study_data/_save/_npy/keras47_2_horse_human_y_train.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_2_horse_human_x_test.npy')
y_test = np.load('d:/study_data/_save/_npy/keras47_2_horse_human_y_test.npy')
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (821, 200, 200, 3) (206, 200, 200, 3) (821, 2) (206, 2)

# 2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(200, 200, 3), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True, verbose=1)
hist = model.fit(x_train, y_train, epochs=500, validation_split=0.1, callbacks=[es] ,verbose=1) # 이렇게 하려면 배치사이즈를 최대로 잡아주면 됨
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# print('loss : ', loss) # 이렇게 하면 모든 로스가 다 나옴 
print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_accuracy : ', val_accuracy[-1])

# loss :  0.024049391970038414
# val_loss :  2.2813332080841064
# accuracy :  0.9878048896789551
# val_accuracy :  0.7108433842658997
