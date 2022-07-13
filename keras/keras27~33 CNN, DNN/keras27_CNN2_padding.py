from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D # 이미지 작업은 이거
from tensorflow.python.keras.layers import Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), 
                 padding='same', # 패딩을 씌우면 커널 사이즈에 상관없이 원래 shape 그대로 감 (보통 0을 씌움)
                 input_shape=(28, 28, 1))) 

model.add(MaxPooling2D()) # dropout과 비슷한 개념인듯? conv2d가 kernel을 이용해서 중첩시키며 특성을 추출해나간다면 maxpoolig은 픽셀을 묶어서 그중에 가장 큰값만 뺌

model.add(Conv2D(32, (2,2),
                 padding='valid', # 이게 padding의 디폴트 값
                 activation='relu')) 
model.add(Flatten()) 
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) # 
model.summary()

# 커널 2,2 
# Model: "sequential"
# _________________________________________________________________    
# Layer (type)                 Output Shape              Param #       
# =================================================================    
# conv2d (Conv2D)              (None, 28, 28, 64)        640
# _________________________________________________________________    
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________    
# conv2d_1 (Conv2D)            (None, 13, 13, 32)        8224
# _________________________________________________________________    
# flatten (Flatten)            (None, 5408)              0
# _________________________________________________________________    
# dense (Dense)                (None, 32)                173088        
# _________________________________________________________________    
# dense_1 (Dense)              (None, 32)                1056
# _________________________________________________________________    
# dense_2 (Dense)              (None, 10)                330
# =================================================================    
# Total params: 183,338
# Trainable params: 183,338
# Non-trainable params: 0
# _________________________________________________________________ 
