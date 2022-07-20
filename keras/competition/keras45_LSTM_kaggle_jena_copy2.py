# https://codingapple.com/unit/deep-learning-stock-price-ai/
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

path = './_data/kaggle_jena/'
df_weather=pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# df_weather.describe()
# print(df_weather.columns)

# 날짜 datetime 포맷으로 변환
# pd.to_datetime(df_weather['Date Time'], format='%Y%m%d')
# # 0      2020-01-07
# # 1      2020-01-06
# # 2      2020-01-03
# # 3      2020-01-02
# # 4      2019-12-30

# df_weather['일자'] = pd.to_datetime(df_weather['Date Time'], format='%Y%m%d')
# df_weather['연도'] =df_weather['Date Time'].dt.year
# df_weather['월'] =df_weather['Date Time'].dt.month
# df_weather['일'] =df_weather['Date Time'].dt.day



# Normalization 정규화
# MinMaxScaler를 해주면 전체 데이터는 0, 1사이의 값을 가짐
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale_cols = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
       'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
       'wd (deg)']
df_scaled = scaler.fit_transform(df_weather[scale_cols])

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

print(df_scaled)



# 학습을 시킬 데이터 셋 생성
# TEST_SIZE = 200은 학습은 과거부터 200일 이전의 데이터를 학습하게 되고, TEST를 위해서 이후 200일의 데이터로 모델이 주가를 예측하도록 한 다음, 
# 실제 데이터와 오차가 얼마나 있는지 확인함
TEST_SIZE = 200

train = df_scaled[:-TEST_SIZE]
test = df_scaled[-TEST_SIZE:]

def make_dataset(data, label, window_size=20): # window_size=20 과거 20일을 기준으로 그 다음날의 데이터를 예측함
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)
# 위의 함수는 정해진 window_size에 기반하여 20일 기간의 데이터 셋을 묶어 주는 역할을 함
# 즉, 순차적으로 20일 동안의 데이터 셋을 묶고, 이에 맞는 label (예측 데이터)와 함께 return 해줌



# feature 와 label(예측 데이터) 정의
feature_cols = ['p (mbar)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
       'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
       'wd (deg)']
label_cols = ['T (degC)']

train_feature = train[feature_cols]
train_label = train[label_cols]


# train dataset
train_feature, train_label = make_dataset(train_feature, train_label, 20)



# train, validation set 생성
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.2)

# print(x_train.shape, x_valid.shape)   #(336264, 20, 13) (84067, 20, 13)

# test dataset (실제 예측 해볼 데이터)
# test_feature, test_label = make_dataset(test_feature, test_label, 20)
# print(test_feature.shape, test_label.shape)




#2.모델 구성
model = Sequential()
model.add(LSTM(16, 
               input_shape=(train_feature.shape[1], train_feature.shape[2]), 
               activation='relu', 
               return_sequences=False)
          )
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,                       
#                       )

hist = model.fit(x_train, y_train, epochs=10, 
                 batch_size=1024, validation_split=0.2, 
                 callbacks=[es], 
                 verbose=1) 



# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# loss :  3.379783811396919e-05







# weight 로딩
# model.load_weights(filename)

# 예측
# pred = model.predict(test_feature)


# 실제데이터와 예측한 데이터 시각화하는 방법

# plt.figure(figsize=(12, 9))
# plt.plot(test_label, label='actual')
# plt.plot(pred, label='prediction')
# plt.legend()
# plt.show()
