# 전처리는 컬럼 별로 함
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler # 구글링해서 정리하고 적용해보기, 4개중에 이상치에 잘 먹는 애가 있음. 찾아보기 (과제)
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

datasets = load_boston()
x = datasets.data
y = datasets['target']

# print(np.min(x)) # x의 최소값 0.0
# print(np.max(x)) # x의 최대값 711.0

# x = (x - np.min(x)) / (np.max(x) - np.min(x))          # / (np.max(x) - np.min(x)는 최소값에서 최대값까지의 범위를 나눠준다는 개념 
# # 0 ~ 1 사이로 
# print(x[:10])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# train 데이터를 먼저 스케일링 시켜서 훈련을 시키고 거기서 나온 결과 수식대로 test 데이터를 다시 스케일링 시켜야 함 
# 스케일러를 사용할 때는 train과 test
# train을 fit 시키고 transform 시키고 이와 동일한 규칙으로 test와 val을 스케일링 시켜야 함
# 그래서 train을 fit 아앙아ㅏ아아아앙아
# fit 시켜줄 때 그 규칙이라는 게 나옴

# 전체 데이터를 전부 스케일링을 하면 범위 밖의 값을 예측할 때 과적합이 발생할 수 있기 때문에
# 

# 1. 스케일러 하기 전
# 2. MinMax
# 3. Standard 
# 이 3개 성능 비교 해보기

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform # <- fit, transform 한방에 (test는 하면 안됨)
x_test = scaler.transform(x_test)

# print(np.min(x_train)) # 0.0
# print(np.max(x_train)) # 1.0 (1.0000000000000002 <- 이딴 식으로 나옴)
# # 이미 컬럼별로 나눠서 스케일링이 돼 있는 상태임
# print(np.min(x_test)) # -0.06141956477526944
# print(np.max(x_test)) # 1.1478180091225068 (이 정도의 오차는 존재해야 함)




#2. 모델 구성 
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
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
#                                                                                                                                                                                                                   



