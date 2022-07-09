# 12개 만들고 최적의 weight 파일을 저장하자ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ
# 전처리는 컬럼 별로 함
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler # 구글링해서 정리하고 적용해보기, 4개중에 이상치에 잘 먹는 애가 있음. 찾아보기 (과제)
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성 
input1 = Input(shape=(13,))
dense1 = Dense(32)(input1)
dense2 = Dense(32)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(32)(drop1)
drop2 = Dropout(0.1)(dense3)
dense4 = Dense(32)(drop2)
drop3 = Dropout(0.1)(dense4)
dense5 = Dense(32)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
#                       save_best_only=True, # 
#                       filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5' 
#                       )
hist = model.fit(x_train, y_train, epochs=500, batch_size=10, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


#             노말                             MinMax                            Standard                             MaxAbs                           Robust                              
# loss : 17.94440269470215              15.174883842468262                 19.268272399902344                   15.438285827636719              20.521202087402344                                                                                                                                       
# r2 :   0.7853102126165242             0.8184451978918657                 0.7694712272474835                   0.8152938097680617              0.7544809681427924                                                                                                                                        
# 함수형                                 13.969481468200684                                                                                                                                                                            
#                                       0.8328668305394888
# Dropout                               15.865423202514648                 16.704450607299805                   15.497111320495605              16.57057762145996                                                                                                          
#                                       0.810183480265888                  0.800145226034957                    0.8145900164278385              0.8017468844811555                                                                                                




'''
#2.  모델 구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dropout(0.3)) # 바로 위 레이어의 노드에서 30%를 랜덤으로 빼줌 (보통 이런 Dropout을 써주면 성능이 더 좋다고 함)
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()


# 3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True, verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es, mcp], verbose=1) 

#4. 평가 예측  <-- 여기는 Dropout이 적용되지 않고 전체 모델이 다 적용됨.
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  10.710644721984863
# r2 스코어 :  0.8718560785416583
'''


