import numpy as np
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten 
import matplotlib.pyplot as plt 


# (class) ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=0.000001, rotation_range=0, width_shift_range=0, height_shift_range=0, brightness_range=None, shear_range=0, zoom_range=0, channel_shift_range=0, fill_mode='nearest', cval=0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0, dtype=None)

train_datagen = ImageDataGenerator(
    rescale=1./255,                      # 0부터 256까지의 숫자들로 구성되어있는 이미지를 학습을 위해 0과 1 사이로 피처스케일링함. MinMax 와 같은 의미
    featurewise_center=False,            # True = 데이터셋 전체에서 입력의 평균을 0으로 함
    samplewise_center=False,             # True = 각 샘플의 평균을 0으로 함
    featurewise_std_normalization=False, # True = 入力をデータセットの標準偏差で正規化します 입력 데이터셋을 표준편차로 정규화 시킴
    samplewise_std_normalization=False,  # True = 各入力をその標準偏差で正規化します 각 입력을 그 표준편차로 정규화 시킴
    zca_whitening=False,                 # True = ZCA白色化を適用します ZCA 백색화를 적용시킴
    zca_epsilon=1e-06,                   # ZCA白色化のイプシロン．デフォルトは1e-6 ZCA 백색화의 epsilon. 디폴트는 1e-6
    rotation_range=0.0,                  # 지정된 각도 내에서 이미지를 무작위로 회전시킴
    width_shift_range=0.0,               # 전체 넓이에 0.0를 곱한 범위 내에서 원본 이미지를 무작위로 수평 이동 시킴
    height_shift_range=0.0,              # 전체 넓이의 0.0를 곱한 범위 내에서 원본 이미지를 무작위로 수직 이동 시킴
    brightness_range=[0.7,1.3],          # 이미지 밝기를 70~130% 사이에서 무작위로 
    shear_range=0.0,                     # 수치만큼 시계 반대 방향으로 밀어 변형시킨다. 단위는 라디안
    zoom_range=0.2,                      # 0.8 에서 1.2배 사이로 확대, 혹은 축소시킨다. 1을 기준으로 플러스마이너스 
    channel_shift_range=0.0,             # 무작위 채널 이동 범위
    fill_mode='nearest',                 # 이미지를 가공할 때 부득이 하게 생기는 빈 공간들을 근처의 색으로 채움
    cval=0.0,                            # fill_mode의 constant시 배경색을 정해줌. fill_mode의 constant 쓸 때만 정의 해주면 됨
    horizontal_flip=False,               # 수평으로 뒤집음
    vertical_flip=False,                 # 수직으로 뒤집음
    preprocessing_function=None,         # 이미지 처리전에 주어진 값을곱해 크기를 조정함 
    data_format=None,                    # 이미지 데이터 형식을 바꿈 
    # channels_last = (샘플, 높이, 넓이, 채널)(일반적으로 기본값)
    # channels_first = (샘플, 채널, 높이, 넓이)
    validation_split=0.0,                # 부동소수점 값으로 train과 valid 로 이미지를 자동으로 나눠줌. subset을 통해 불러옴
    dtype=None                           # 생성된 배열에 사용할 자료형을 지정함
    )


data = train_datagen.flow_from_directory('경로지정', 
                                        target_size = (256, 256), 
                                        color_mode= 'rgb', 
                                        classes= None, 
                                        class_mode= 'categorical', 
                                        batch_size= 32, 
                                        shuffle= True, seed = None, 
                                        save_to_dir = None, 
                                        save_prefix='', 
                                        save_format= 'png', 
                                        follow_links = False, 
                                        subset = None, 
                                        interpolation = 'nearest'
                                        )


# 
