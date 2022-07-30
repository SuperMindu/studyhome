import numpy as np
from keras.preprocessing.image import ImageDataGenerator, image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

data = train_datagen.flow_from_directory('d:/PP/', # 폴더에서 가져와서 ImageDataGenerator
                                             target_size=(255, 255), # 이미지 크기 조절. 고르지 않은 크기들을 사이즈를 지정해줌. 내 맘대로 가능
                                             batch_size=500000000,
                                             class_mode='categorical', # 0, 1 분류. 이진분류. (3가지 이상은 categorical)
                                            #  color_mode='grayscale', # 이걸 따로 지정해주지 않으면 디폴트값은 컬러로 나옴. 밑에 print(xy_train[31][0].shape) 참고
                                             shuffle=True,
                                             ) 
# Found 160 images belonging to 2 classes. flow_from_directory를 통과했을 때 160개의 이미지와 2개의 클래스가 됐음

# xy_test = test_datagen.flow_from_directory('d:/_data/image/brain/test/', 
#                                              target_size=(150, 150), 
#                                              batch_size=5,
#                                              class_mode='categorical', 
#                                              color_mode='grayscale',
#                                              shuffle=True,
#                                              ) 


x = data[0][0]
y = data[0][1]
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

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
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True, verbose=1)
hist = model.fit(x_train, y_train, epochs=500, batch_size=8, validation_split=0.1, callbacks=[es] ,verbose=1) # 이렇게 하려면 배치사이즈를 최대로 잡아주면 됨
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 4. 평가, 예측


# print('loss : ', loss) # 이렇게 하면 모든 로스가 다 나옴 
print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_accuracy : ', val_accuracy[-1])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)  
print(loss)                    

pic_path = 'd:/PP/2.jpg'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(255,255))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /=255.

    if show:
            plt.imshow(img_tensor[0])    
            plt.show()
    
    return img_tensor

img = load_my_image(pic_path)

pred = model.predict(img)

face_per = round(pred[0][0]*100,1)
body_per = round(pred[0][1]*100,1)
    

if pred[0][0] > pred[0][1]:
    print(f'이 사진은 {face_per}% 의 확률로 얼굴사진 입니다.')
else : 
    print(f'이 사진은 {body_per}% 의 확률로 전신사진 입니다.')

