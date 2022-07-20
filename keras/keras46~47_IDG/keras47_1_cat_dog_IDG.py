# 넘파이로 저장
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

xy_train = train_datagen.flow_from_directory('d:/study_data/_data/image/cat_dog/training_set/', # 폴더에서 가져와서 ImageDataGenerator
                                             target_size=(150, 150), # 이미지 크기 조절. 고르지 않은 크기들을 사이즈를 지정해줌. 내 맘대로 가능
                                             batch_size=100000000,
                                             class_mode='binary', # 0, 1 분류. 이진분류. (3가지 이상은 categorical)
                                            #  color_mode='grayscale', # 이걸 따로 지정해주지 않으면 디폴트값은 컬러로 나옴. 밑에 print(xy_train[31][0].shape) 참고
                                             shuffle=True,
                                             ) 
# Found 160 images belonging to 2 classes. flow_from_directory를 통과했을 때 160개의 이미지와 2개의 클래스가 됐음

xy_test = test_datagen.flow_from_directory('d:/study_data/_data/image/cat_dog/test_set/', 
                                             target_size=(150, 150), 
                                             batch_size=100000000,
                                             class_mode='binary', 
                                            #  color_mode='grayscale',
                                             shuffle=True,
                                             ) 

print(xy_train[0][0].shape, xy_train[0][1].shape) # (8005, 150, 150, 3) (8005, 2)
print(xy_test[0][0].shape, xy_test[0][1].shape) # (2023, 150, 150, 3) (2023, 2)

np.save('d:/study_data/_save/_npy/keras47_1_cat_dog_train_x.npy', arr=xy_train[0][0]) 
np.save('d:/study_data/_save/_npy/keras47_1_cat_dog_train_y.npy', arr=xy_train[0][1]) 
np.save('d:/study_data/_save/_npy/keras47_1_cat_dog_test_x.npy', arr=xy_test[0][0]) 
np.save('d:/study_data/_save/_npy/keras47_1_cat_dog_test_y.npy', arr=xy_test[0][1]) 