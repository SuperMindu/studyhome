import tensorflow as tf
import numpy as np
import pandas as pd
from sqlalchemy import true 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, Conv1D
from keras.layers.recurrent import SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from sklearn.metrics import r2_score, accuracy_score
import datetime as dt

''' 1-1) 데이터 로드 '''
df_amore=pd.read_csv('./_data/test_amore_0718/아모레220718.csv', thousands=',', encoding='cp949') # 아모레 데이터 로드
df_samsung=pd.read_csv('./_data/test_amore_0718/삼성전자220718.csv', thousands=',', encoding='cp949') # 삼성전자 데이터 로드


''' 1-2) 데이터 정제 '''
# 결측치 처리 
df_amore = df_amore.fillna(0)
df_samsung = df_samsung.fillna(0)

# 데이터 사용할 기간 설정
df_amore = df_amore.loc[df_amore['일자']>="2018/05/04"]
df_samsung = df_samsung.loc[df_samsung['일자']>="2018/05/04"]
print(df_amore.shape, df_samsung.shape) # (1035, 17) (1035, 17)

# 데이터 오름차순 정렬
df_amore = df_amore.sort_values(by=['일자'], axis=0, ascending=True) 
df_samsung = df_samsung.sort_values(by=['일자'], axis=0, ascending=True)

# 필요없는 컬럼 삭제
df_amore = df_amore.drop(['일자', '전일비', 'Unnamed: 6', '등락률', '금액(백만)', '신용비', '개인', '외인(수량)', '프로그램', '외인비'], axis=1) 
df_samsung = df_samsung.drop(['일자', '전일비', 'Unnamed: 6', '등락률','금액(백만)', '신용비', '개인', '외인(수량)', '프로그램', '외인비'], axis=1) 
df_amore = np.array(df_amore)
df_samsung = np.array(df_samsung)
print(df_amore.shape, df_samsung.shape) # (1035, 7) (1035, 7)  

# 스플릿 함수
def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number+1 : y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
# x, y = split_xy(dataset, 3 , 1)
x1, y1 = split_xy(df_amore, 7, 3)
x2, y2 = split_xy(df_samsung, 7, 3)
print(x1.shape, x2.shape, y1.shape) # (1027, 7, 7) (1027, 7, 7) (1027, 3)

# 스플릿 
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size=0.8, shuffle=False)
print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y1_train.shape, y1_test.shape)
# (821, 7, 7) (206, 7, 7) (821, 7, 7) (206, 7, 7) (821, 1) (206, 1)

# data 스케일링
scaler = MinMaxScaler()

x1_train = x1_train.reshape(821*7, 7)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(206*7, 7)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(821*7, 7)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(206*7, 7)
x2_test = scaler.transform(x2_test)

# Conv1D에 넣기 위해 3차원화
x1_train = x1_train.reshape(821, 7, 7)
x1_test = x1_test.reshape(206, 7, 7)
x2_train = x2_train.reshape(821, 7, 7)
x2_test = x2_test.reshape(206, 7, 7)

''' 2. 모델 구성 '''
# 2-1. 모델1
input1 = Input(shape=(7, 7))
conv1 = Conv1D(64, 2, activation='relu')(input1)
lstm1 = LSTM(128, activation='relu')(conv1)
dense1 = Dense(64, activation='relu')(lstm1)
output1 = Dense(32, activation='relu')(dense1)

# 2-2. 모델2
input2 = Input(shape=(7, 7))
conv2 = Conv1D(64, 2, activation='relu')(input2)
lstm2 = LSTM(128, activation='swish')(conv2)
dense2 = Dense(64, activation='relu')(lstm2)
dense3 = Dense(32, activation='relu')(dense2)
output2 = Dense(16, activation='relu')(dense3)

from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(100, activation='relu')(merge1)
merge3 = Dense(100, name='mg3')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])
# model.summary()

''' 3. 컴파일, 훈련 '''
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)    
hist = model.fit([x1_train, x2_train], y1_train, epochs=10, batch_size=128, validation_split=0.15, callbacks=[es], verbose=1)

''' 4. 평가, 예측 '''
loss = model.evaluate([x1_test, x2_test], y1_test)
print('loss : ' , loss)
y_predict = model.predict([x1_test, x2_test])
print('7월 20일 예측 종가 : ', y_predict[-1:])
