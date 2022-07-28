import numpy as np
from keras.preprocessing.image import ImageDataGenerator # 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt


''' 1. 데이터 '''



''' 1-1. ImageDataGenerator 선언 '''
train_datagen = ImageDataGenerator(    # 아래의 모든건 컴터가 알아서 몇개를 적용할지 다 랜덤으로 적용됨, 증폭가능. 밑에거를 쓸지 말지는 나의 선택
    rescale=1./255,                    # MinMax 해주겠다는 것과 같음 (스케일링 기능임)
    horizontal_flip=True,              # 상하 반전
    vertical_flip=True,                # 좌우 반전
    width_shift_range=0.1,             # 가로 수평 이동 0.1 (10%)
    height_shift_range=0.1,            # 상하 이동 
    rotation_range=5,                  # 돌려돌려돌림판
    zoom_range=1.2,                    # 확대 
    shear_range=0.7,                   # 기울기? 찌그러트리기? 비틀기? 짜부? 
    fill_mode='nearest'                # 위의 여러가지 증폭 메뉴에서 설정한 파라미터대로 이미지를 건드려서 빈칸이 발생했다면 그 빈 곳(빈 픽셀)을 근처의 값으로 채우겠다는 뜻
)                                      # 아직까지는 트레인 데이터 준비만 한거임. 선언만 한거임. 밑에서 모델에 넣을 수 있도록 정제는 내가 따로 해줘야 함
test_datagen = ImageDataGenerator(
    rescale=1./255                     # 테스트 데이터는 증폭하면 안됨. 평가할 때는 원본으로 해야 정확한 검증이 가능함
)
# 이미지를 제자리에서 이동 시키든가, 상하좌우 반전을 시키든가, 기타등등 해서 여러가지 형태의 이미지를 모두 학습시킴으로써 어떤형태의 이미지라도 판단할 수 있게하는 목적임



''' 1-2. flow_from_directory '''
xy_train = train_datagen.flow_from_directory('d:/study_data/_data/image/brain/train/', # 이미지 데이터를 받아올 경로를 설정 해줌
                                             target_size=(200, 200),                   # 이미지 크기 조절. 고르지 않은 크기들을 가로세로 픽셀 사이즈를 지정해줌. 내 맘대로 가능
                                             batch_size=1,                             # 이건 그냥 냅다 1억 정도 박자. 어차피 밑에서 걍 model.fit 쓰면 배치사이즈 지정 가능
                                             class_mode='binary',                      # 0, 1 분류. 이진분류. (3가지 이상은 categorical)
                                             color_mode='grayscale',                   # 이걸 따로 지정해주지 않으면 디폴트값은 컬러로 나옴. 밑에 print(xy_train[31][0].shape) 참고
                                             shuffle=True, seed=66,                    # 안의 내용을 섞어줌  randomstate와 마찬가지로 랜덤값 고정. 변환할 정도를 랜덤셔플하게 섞어줌
                                             save_to_dir='경로 지정'                    # 건든 데이터를 출력해서 보여줌(?)
                                             ) 
# Found 160 images belonging to 2 classes. (flow_from_directory를 통과했을 때 160개의 이미지와 2개의 클래스가 됨)

xy_test = test_datagen.flow_from_directory('d:/study_data/_data/image/brain/test/', 
                                             target_size=(200, 200), 
                                             batch_size=5,
                                             class_mode='binary', 
                                             shuffle=True,
                                             ) 
# Found 120 images belonging to 2 classes.
# 파이토치의 데이터로더 및 텐서플로(케라스)의 ImageDataGenerator 등등 다 이런형식으로 저장되어있고 불러옴

# print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001D1D70A8D90> 

# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets) # <- 데이터 자체가 나오진 않음. 설명만 주저리



''' 1-3. print()해서 확인 '''
# print(xy_train[ ]) # <- x_train에 y값이 포함 돼있다는 걸 알 수 있음. (배치사이즈 5개니까 160/5=32개) 마지막 배치는 31개
# 근데 지금과 같이 이미지 개수가 160개일때 배치를 6으로 하는 경우(26.66666...) 나눠 떨어지지 않는 경우는 마지막 값을 찍어보면 그 나눈 나머지가 나옴 
# 이미지 데이터의 전체 사이즈보다 배치를 더 많이 넣어도 컴터가 알아서 해줌 
# 하지만 우리는 어차피 model.fit을 쓸거기 때문에 위에 배치 사이즈는 그냥 이미지의 개수만큼(아니면 냅다 엄청 큰 값)을 때려박아놓고 원래 하던 것처럼 model.fit에서 batch_size 지정 해주면 됨. 이렇게 하면
# print(xy_train[0])을 했을 때 첫번째 배치가 나옴. 이게 곧 전체 이미지 데이터를 의미하는 것과 다를 게 없음. dtype=float32), array([0., 0., 0., 0., 0.], dtype=float32)) <- 이런 형식으로 나옴

# print(xy_train[31][0]) 0하면 x값만 나옴. 1이 y. 
# print(xy_train[31][1]) 
# print(xy_train[30]1][.shape) # (5, 150, 150, 3) 따로 컬러를 지정해주지 않으면 흑백도 컬러로 인식함. 따라서 (n, n, n, 3)
# print(xy_train[31][0].shape, xy_train[31][1].shape)

# print(xy_train[0][0]) <- xy_train의 첫번째 덩어리의 첫번째 = x 
# print(xy_train[0][1]) <- xy_train의 첫번째 덩어리의 두번째 = y # <- 원핫인코딩까지 돼서 0 1 로 나옴
# print(xy_train[0][0].shape, xy_train[0][1].shape) -> (배치 사이즈 값, 타겟 사이즈 가로값, 타겟 사이즈 세로값, 흑백or컬러 값)

print(type(xy_train)) # 자료형을 보면 <class 'keras.preprocessing.image.DirectoryIterator'>  Iterator는 반복자라는 뜻. for문도 Iterator의 일종임
print(type(xy_train[0]))       # <class 'tuple'> 튜플 형식인 걸 알 수 있음. 튜플은 안의 값들이 수정이 불가능함
print(type(xy_train[0][0]))    # x <class 'numpy.ndarray'> 넘파이 어레이 형식
print(type(xy_train[0][1]))    # y <class 'numpy.ndarray'>
                               # x 넘파이 y 넘파이가 배치 단위로 묶여있는 구조
                               
                               
                               
''' 1-4. 전처리된 이미지 데이터 저장 '''
# 이미지 개수가 많아지면 불러올 때마다 로딩 시간이 너무 길어짐 (모델 한번 돌릴 때마다 불러와야 함)
# 그래서 데이터를 수치화 시켜서 저장을 해놓고 필요할 때 그냥 그 저장해놓은 파일을 불러오기만 하면됨
# 나중에 크기가 커지면 csv 파일도 이렇게 변환해서 저장 해놓고 불러오는 게 더 빠를 수 있음
np.save('경로', arr=xy_train[0][0]) #
np.save('경로', arr=xy_train[0][1]) # 
np.save('경로', arr=xy_test[0][0])   # 
np.save('경로', arr=xy_test[0][1])   
# 넘파이 수치로 변환해서 저장시켜줌 이렇게 하는게 훨씬 빠름



''' 1-5. 전처리된 이미지 데이터 로드 '''
x_train = np.load('경로')
y_train = np.load('경로')
x_test = np.load('경로')
y_test = np.load('경로')



''' 2. 모델 구성'''
# 알잘딱깔센
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(150, 150, 1), activation='relu'))
model.add(Conv2D(10, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))



''' 3. 컴파일, 훈련 '''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(xy_train[0][0], xy_train[0][1]) <- 우리는 그냥 이렇게 해주면 되는데 일단 fit_generator를 쓰는 방법도 알아두자

hist = model.fit_generator( # 여기엔 따로 배치 사이즈가 없음
                    xy_train, epochs=10, steps_per_epoch=32, # <- 데이터 셋을 배치사이즈로 나눈 덩어리값. 보통 이 값을 넣어줌.  전체데이터/batch = 160/5 = 32. 걍 확인차의 의미임
                    validation_data=xy_test,                 # <- 근데 여기서는 발리데이션을 test 데이터로만 해서 과적합(?)이 발생할 수 있음
                    validation_steps=4                       # steps_per_epoch가 특정된 경우에만 유의미함. 정지 전 검증할 단계(샘플 배치)의 총 개수
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



''' 4. 평가, 예측 '''




''' 5. 시각화 '''
plt.figure(figsize=(9,6))                                         # (9 x 6 으로 그리겠다는 뜻) (figure <- 이거 알아서 찾아보기)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # marker '_' 로 하면 선, 아예 삭제해도 선으로 이어짐
plt.plot(hist.history['accuracy'], marker='.', c='blue', label='accuracy')
plt.plot(hist.history['val_loss'], marker='.', c='yellow', label='val_loss')
plt.plot(hist.history['val_accuracy'], marker='.', c='green', label='val_accuracy')
plt.grid()                                                        # 보기 편하게 하기 위해서 모눈종이 형태로 
plt.title('SuperMindu')                                           # 표 이름
plt.ylabel('loss')                                                # y축 이름 설정
plt.xlabel('epochs')                                              # x축 이름 설정 
plt.legend(loc='upper right')                                     # 위에 label 값이 이 위치에 명시가 된다 (디폴트 값이 upper right)
plt.show()









