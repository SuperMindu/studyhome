import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time

#1. 데이터 
path = 'C:/study/study-home/_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
print(train_set) # [1459 rows x 10 columns]
print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path + 'test.csv', index_col=0)
print(test_set) # [715 rows x 9 columns]
print(test_set.shape) # (715, 9)

print(train_set.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
print(train_set.info())
# Data columns (total 10 columns):
#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
# dtypes: float64(9), int64(1)
print(train_set.describe())
# None
#               hour  hour_bef_temperature  ...  hour_bef_pm2.5        count
# count  1459.000000           1457.000000  ...     1342.000000  1459.000000
# mean     11.493489             16.717433  ...       30.327124   108.563400
# std       6.922790              5.239150  ...       14.713252    82.631733
# min       0.000000              3.100000  ...        8.000000     1.000000
# 25%       5.500000             12.800000  ...       20.000000    37.000000
# 50%      11.000000             16.600000  ...       26.000000    96.000000
# 75%      17.500000             20.100000  ...       37.000000   150.000000
# max      23.000000             30.000000  ...       90.000000   431.000000
# [8 rows x 10 columns]

#1-1. 결측치 처리 
print(train_set.isnull().sum())
# hour                        0
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117
# count                       0
# dtype: int64
train_set = train_set.fillna(train_set.mean())
test_set = test_set.fillna(test_set.mean())
print(train_set.isnull().sum())
# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0
# count                     0
# dtype: int64
print(train_set.shape) # (1459, 10)

x = train_set.drop(['count'], axis=1) 
print(x)
print(x.columns)
print(x.shape) # (1459, 9)

y = train_set['count']
print(y)
print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=31)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, activation = 'relu', input_dim=9))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=5000, batch_size=50, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x, y)
y_predict = model.predict(x_test)

def RMSE(a, b):
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print('RMSE : ', rmse)
print('r2 스코어 : ', r2)

# none activation
# loss :  [2795.192138671875, 2795.192138671875]
# RMSE :  55.78656572141627
# r2 스코어 :  0.574063206003459

# activation = 'relu'
# loss :  [2415.80517578125, 2415.80517578125]
# RMSE :  54.48658595683426
# r2 스코어 :  0.5936829024501256


#5. 제출용 제작 


# y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape) # (715, 1)

# submission_set = pd.read_csv(path + 'submission.csv', # + 명령어는 문자를 앞문자와 더해줌
#                              index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

# print(submission_set)

# submission_set['count'] = y_summit
# print(submission_set)

# submission_set.to_csv('./_data/ddarung/submission.csv', index = True)

# y_predict = model.predict(test_set)

