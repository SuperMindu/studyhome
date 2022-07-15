# 앙상블 모델 
# 머신러닝에 가면 앙상블이 많음
# 
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical 

# 1-1) 데이터 로드
import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]) # ex) 삼전 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]) # 원유, 돈육, 밀
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)

print(x1.shape, x2.shape) # (100, 2) (100, 3)

y = np.array(range(2001, 2101)) # ex) 금리
print(y.shape) # (100,)




# print(np.unique(y, return_counts=True)) 

# y = to_categorical(y)  
# print(y) 
# print(y.shape)


# 1-2) 데이터 정제 
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2 ,y, train_size=0.8, shuffle=True, random_state=66) 
print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)
#             (80, 2)         (20, 2)        (80, 3)        (20, 3)         (80,)         (20,)

# 2. 모델 구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout

# 2-1) 모델1 (x1)
input1 = Input(shape=(2, ))
dense1 = Dense(32, activation='relu', name='ms1')(input1)
dense2 = Dense(16, activation='relu', name='ms2')(dense1)
dense3 = Dense(8, activation='relu', name='ms3')(dense2)
output1 = Dense(1, activation='relu', name='out_ms1')(dense3)

# 2-2) 모델2 (x2)
input2 = Input(shape=(3, ))
dense11 = Dense(32, activation='relu', name='ms11')(input2)
dense12 = Dense(32, activation='relu', name='ms12')(dense11)
dense13 = Dense(16, activation='relu', name='ms13')(dense12)
dense14 = Dense(16, activation='relu', name='ms14')(dense13)
output2 = Dense(1, activation='relu', name='out_ms2')(dense14)

from tensorflow.python.keras.layers import concatenate, Concatenate # 소문자는 함수, 대문자는 클래스 # concatenate는 사슬처럼 잇다. 
merge1 = concatenate([output1, output2], name='mg1') # output1의 노드개수와 output2의 노드개수가 합쳐진 하나의 (Dense)레이어가 된거임
merge2 = Dense(3, activation='relu', name='mg2')(merge1) # <----- 노드를 3 4 1 순서로 하면 loss가 계속 줄어드는데 왜 8 4 1 이렇게 점점 작게 주면 개판나는거지? 원래 더 줄어야 맞는거 아닌가?...
merge3 = Dense(4, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)
model = Model(inputs=[input1, input2], outputs=last_output) 
# model.summary()
# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_2 (InputLayer)            [(None, 3)]          0
# __________________________________________________________________________________________________
# input_1 (InputLayer)            [(None, 2)]          0
# __________________________________________________________________________________________________
# ms11 (Dense)                    (None, 11)           44          input_2[0][0]
# __________________________________________________________________________________________________
# ms1 (Dense)                     (None, 1)            3           input_1[0][0]
# __________________________________________________________________________________________________
# ms12 (Dense)                    (None, 12)           144         ms11[0][0]
# __________________________________________________________________________________________________
# ms2 (Dense)                     (None, 2)            4           ms1[0][0]
# __________________________________________________________________________________________________
# ms13 (Dense)                    (None, 13)           169         ms12[0][0]
# __________________________________________________________________________________________________
# ms3 (Dense)                     (None, 3)            9           ms2[0][0]
# __________________________________________________________________________________________________
# ms14 (Dense)                    (None, 14)           196         ms13[0][0]
# __________________________________________________________________________________________________
# out_ms1 (Dense)                 (None, 10)           40          ms3[0][0]
# __________________________________________________________________________________________________
# out_ms2 (Dense)                 (None, 10)           150         ms14[0][0]
# __________________________________________________________________________________________________
# mg1 (Concatenate)               (None, 20)           0           out_ms1[0][0]
#                                                                  out_ms2[0][0]
# __________________________________________________________________________________________________
# mg2 (Dense)                     (None, 2)            42          mg1[0][0]
# __________________________________________________________________________________________________
# mg3 (Dense)                     (None, 3)            9           mg2[0][0]
# __________________________________________________________________________________________________
# last (Dense)                    (None, 1)            4           mg3[0][0]
# ==================================================================================================
# Total params: 814
# Trainable params: 814
# Non-trainable params: 0
# __________________________________________________________________________________________________

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=30, mode='auto', verbose=1, restore_best_weights=True)        
hist = model.fit([x1_train, x2_train], y_train, epochs=300, batch_size=8, validation_split=0.2, callbacks=[es], verbose=1)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ' , loss)
# y_predict = model.predict([x1_test, x2_test])
r2= r2_score(last_output, y_test)
# print('loss : ' , loss)
print('r2 스코어 : ', r2) 

# loss :  0.02222190424799919


