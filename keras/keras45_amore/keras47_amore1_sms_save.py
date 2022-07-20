# 19일 09시 아모레 시가(20%), 20일 종가(80%) 맞추기
# 거래량 반드시. 7개 이상의 컬럼 쓰기
# 삼전이랑 앙상블 해서 만들기
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

###########################split 함수########################################
# def split_xy(dataset, time_steps, y_column):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps
#         y_end_number = x_end_number + y_column -1
        
#         if y_end_number > len(dataset):
#             break
#         tmp_x = dataset[i : x_end_number, : ]
#         tmp_y = dataset[x_end_number : y_end_number, : -1]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)
# # x, y = split_xy(dataset, 3 , 1) <- 마지막에 이런식으로 해주면 됨 
# 근데 이걸로 split하면 y가 행으로 잘려서 열로 자르게 바꿔줘야 함


###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감#######################

''' 1-1) 데이터 로드 '''
df_amore=pd.read_csv('./_data/test_amore_0718/아모레220718.csv', thousands=',', encoding='cp949') # 아모레 데이터 로드
df_samsung=pd.read_csv('./_data/test_amore_0718/삼성전자220718.csv', thousands=',', encoding='cp949') # 삼성전자 데이터 로드
# df_amore.describe()
# print(df_amore.columns)
# Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
#       dtype='object')

''' 1-2) 데이터 정제 '''
# 결측치 확인 및 처리 
# print(df_amore.isnull()) 
# print(df_samsung.isnull()) 
# df_amore = df_amore.dropna()
# df_samsung = df_samsung.dropna()
df_amore = df_amore.fillna(0)
df_samsung = df_samsung.fillna(0)

# 이상치 확인
# q3 = df_amore.quantile(0.75) 
# q1 = df_amore.quantile(0.25)
# iqr = q3 - q1

# 데이터 사용할 기간 설정
df_amore = df_amore.loc[df_amore['일자']>="2018/05/04"]
df_samsung = df_samsung.loc[df_samsung['일자']>="2018/05/04"]
# print(df_amore.shape, df_samsung.shape) # (1035, 17) (1035, 17)

# 데이터 오름차순 정렬
df_amore = df_amore.sort_values(by=['일자'], axis=0, ascending=True) 
df_samsung = df_samsung.sort_values(by=['일자'], axis=0, ascending=True)

# 필요없는 컬럼 삭제
df_amore = df_amore.drop(['일자', '전일비', 'Unnamed: 6', '등락률', '금액(백만)', '신용비', '개인', '외인(수량)', '프로그램', '외인비'], axis=1) 
df_samsung = df_samsung.drop(['일자', '전일비', 'Unnamed: 6', '등락률','금액(백만)', '신용비', '개인', '외인(수량)', '프로그램', '외인비'], axis=1) 
df_amore = np.array(df_amore)
df_samsung = np.array(df_samsung)
# print(df_amore.shape)
# print(df_amore.shape, df_samsung.shape) # (1035, 7) (1035, 7)

# 년월일 분리 
# df_amore['날짜_datetime'] = pd.to_datetime(df_amore['일자'])
# df_amore['년'] = df_amore['날짜_datetime'].dt.year
# df_amore['월'] = df_amore['날짜_datetime'].dt.month
# df_amore['일'] = df_amore['날짜_datetime'].dt.day
# # df_amore['요일'] = df_amore['날짜_datetime'].dt.day_name()
# df_amore = df_amore.drop(['일자', '날짜_datetime'], axis=1) 

# df_samsung['날짜_datetime'] = pd.to_datetime(df_samsung['일자'])
# df_samsung['년'] = df_samsung['날짜_datetime'].dt.year
# df_samsung['월'] = df_samsung['날짜_datetime'].dt.month
# df_samsung['일'] = df_samsung['날짜_datetime'].dt.day
# # df_samsung['요일'] = df_samsung['날짜_datetime'].dt.day_name()
# df_samsung = df_samsung.drop(['일자', '날짜_datetime'], axis=1) 
# # print(df_amore)
# # print(df_samsung)
# print(df_amore.shape, df_samsung.shape) # (1035, 10) (1035, 10)


# 피처 및 타겟 데이터 지정
# feature_data = ['시가', '고가', '저가', '종가', '거래량', '기관', '외국계']
# target_data = ['종가']
# 시계열 데이터 함수 및 train_test_split 
# def split_x(dataset, size):
#     aaa = []
#     for i in range(len(dataset) - size + 1):
#         subset = dataset[i : (i + size)]
#         aaa.append(subset)
#     return np.array(aaa)

# SIZE = 20
# x1 = split_x(df_amore[feature_data], SIZE)
# x2 = split_x(df_samsung[feature_data], SIZE)
# y = split_x(df_amore[target_data], SIZE)
# x1 = x1[:, :-1]                                
# x2 = x2[:, :-1]                                
# y = y[:, -1]
# print(y.shape) # (1016, 1)
def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number-1 : y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
# x, y = split_xy(dataset, 3 , 1)
x1, y1 = split_xy(df_amore, 7, 3)
x2, y2 = split_xy(df_samsung, 7, 3)

print(x1.shape, x2.shape, y1.shape)
# (1027, 7, 7) (1027, 7, 7) (1027, 3)

# y = split_xy(df_amore[target_data], 1, 3)
# x2 = split_x(df_samsung[feature_data], 3)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size=0.8, shuffle=False)
print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y1_train.shape, y1_test.shape)
# (821, 7, 7) (206, 7, 7) (821, 7, 7) (206, 7, 7) (821, 3) (206, 3)
print(y1_test)

# # # 스케일링 및 reshape
# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# x1_train = x1_train.reshape(821*7, 7)
# x1_train = scaler.fit_transform(x1_train)
# x1_test = x1_test.reshape(206*7, 7)
# x1_test = scaler.transform(x1_test)

# x2_train = x2_train.reshape(821*7, 7)
# x2_train = scaler.fit_transform(x2_train)
# x2_test = x2_test.reshape(206*7, 7)
# x2_test = scaler.transform(x2_test)

# x1_train = x1_train.reshape(821, 7, 7)
# x1_test = x1_test.reshape(206, 7, 7)
# x2_train = x2_train.reshape(821, 7, 7)
# x2_test = x2_test.reshape(206, 7, 7)



# ''' 2. 모델 구성 '''
# # 2-1. x1 모델
# input1 = Input(shape=(7, 7))
# dense1 = Conv1D(64, 2, activation='relu', name='ms1')(input1)
# dense2 = (LSTM(64, activation='relu', name='ms2'))(dense1)
# dense3 = Dense(32, activation='relu', name='ms3')(dense2)
# output1 = Dense(16, activation='relu', name='out_ms1')(dense3)

# # 2-2. x2 모델
# input2 = Input(shape=(7, 7))
# dense11 = Conv1D(64, 2, activation='relu', name='ms11')(input2)
# dense12 = (LSTM(64, activation='relu', name='ms12'))(dense11)
# dense13 = Dense(32, activation='relu', name='ms13')(dense12)
# output2 = Dense(16, activation='relu', name='out_ms2')(dense13)

# # 앙상블
# from tensorflow.python.keras.layers import concatenate
# merge1 = concatenate([output1, output2], name='mg1')
# merge2 = Dense(16, activation='relu', name='mg2')(merge1)
# merge3 = Dense(8, activation='relu', name='mg3')(merge2)
# last_output = Dense(3,  name='last')(merge3)
# model = Model(inputs=[input1, input2], outputs=[last_output])
# # model.summary()

# ''' 3. 컴파일, 훈련 '''
# model.compile(loss='mse', optimizer='adam')

# date = datetime.datetime.now()
# date = date.strftime("%Y%m%d_%H%M") 
# save_filepath = './_test/' + current_name + '/'
# load_filepath = './_test/' + current_name + '/'
# model = load_model(load_filepath + '0708_1753_0011-0.0731.hdf5')
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)     
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath= "".join([save_filepath, date, '_', filename]))   
# hist = model.fit([x1_train, x2_train], y1_train, epochs=10, batch_size=128, validation_split=0.15, callbacks=[es,mcp], verbose=1)


# ''' 4. 평가, 예측 '''
# loss = model.evaluate([x1_test, x2_test], y1_test)
# print('loss : ' , loss)
# y_predict = model.predict([x1_test, x2_test])
# print('7월 20일 예측 종가 : ', y_predict[-1:])
# # loss :  594746816.0
# # 7월 20일 예측 종가 :  [[160947.02]]
# # loss :  488668352.0
# # 7월 20일 예측 종가 :  [[129772.9]]
