# 19일 09시 아모레 시가(20%), 20일 종가(80%) 맞추기
# 거래량 반드시. 7개 이상의 컬럼 쓰기
# 삼전이랑 앙상블 해서 만들기
import numpy as np
import pandas as pd
from sqlalchemy import true 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from keras.layers.recurrent import SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
import datetime as dt

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
df_amore.describe()

   
''' 1-2) 데이터 정제 '''
# 결측치 확인
# print(df_amore.info()) 

# 이상치 확인
# q3 = df_amore.quantile(0.75) 
# q1 = df_amore.quantile(0.25)
# iqr = q3 - q1

# 필요없는 컬럼 삭제
df_amore = df_amore.drop(['등락률', 'Unnamed: 6', '전일비', '금액(백만)', '신용비', '개인', '외인(수량)', '프로그램', '외인비'], axis=1) 
df_amore = df_amore.drop(['Unnamed: 6'], axis=1) 
df_samsung = df_samsung.drop(['등락률'], axis=1) 
df_samsung = df_samsung.drop(['Unnamed: 6'], axis=1) 
# print(df_amore)
# print(df_samsung)

# 년월일 분리 및 요일 추가
df_amore['날짜_datetime'] = pd.to_datetime(df_amore['일자'])
df_amore['년'] = df_amore['날짜_datetime'].dt.year
df_amore['월'] = df_amore['날짜_datetime'].dt.month
df_amore['일'] = df_amore['날짜_datetime'].dt.day
df_amore['요일'] = df_amore['날짜_datetime'].dt.day_name()
df_amore = df_amore.drop(['일자', '날짜_datetime'], axis=1) 

df_samsung['날짜_datetime'] = pd.to_datetime(df_samsung['일자'])
df_samsung['년'] = df_samsung['날짜_datetime'].dt.year
df_samsung['월'] = df_samsung['날짜_datetime'].dt.month
df_samsung['일'] = df_samsung['날짜_datetime'].dt.day
df_samsung['요일'] = df_samsung['날짜_datetime'].dt.day_name()
df_samsung = df_samsung.drop(['일자', '날짜_datetime'], axis=1) 
# print(df_amore)
# print(df_samsung)
print(df_amore.shape) # (3180, 18)
print(df_samsung.shape) # (3040, 18)

# 라벨인코딩
# df_amore, df_samsung=['요일']
# encoder = LabelEncoder()
for col in ['요일']:
    encoder = LabelEncoder()
    df_amore[col] = encoder.fit_transform(df_amore[col])
df_amore.loc[:,['요일']].head()
# print(labels)
print(df_amore)

for col in ['요일']:
    encoder = LabelEncoder()
    df_samsung[col] = encoder.fit_transform(df_samsung[col])
df_samsung.loc[:,['요일']].head()
# print(labels)
print(df_samsung)

# 필요 없는 행 삭제
df_amore = df_amore.drop(index=[1773,1774,1775,1776,1777,1778,1779,1780,1781,1782])
df_samsung = df_samsung.drop(index=[1037,1038,1039])
# print(df_amore.shape) # (3170, 18)
# print(df_samsung.shape) # (3037, 18)

# 하아아아아아아아앙



'''
print(train_set)
print(train_set.shape) # (10886, 12)
                  
print(test_set)
print(test_set.shape) # (6493, 9)
print(test_set.info()) # (715, 9)
print(train_set.columns)
print(train_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력


##########################################


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (10886, 12)

y = train_set['count'] 
print(y)
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    random_state=31
                                                    )

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)

print(x_train.shape)
print(x_test.shape) 
###################리세이프#######################
x_train = x_train.reshape(8164, 4, 3)
x_test = x_test.reshape(2722, 4, 3) 
test_set = test_set.reshape(6493, 4, 3) 
print(x_train.shape)
print(np.unique(y_train, return_counts=True))
#################################################


#2. 모델구성

# model = load_model("./_save/keras22_hamsu10_kaggle_bike.h5")

# model = Sequential()
# model.add(Dense(100, activation='swish', input_dim=12))
# model.add(Dense(100, activation='elu'))
# model.add(Dense(100, activation='swish'))
# model.add(Dense(100, activation='elu'))
# model.add(Dense(1))

model = Sequential()
model.add(LSTM(200, return_sequences=True, input_shape=(4,3)))
model.add(LSTM(100))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

filepath = './_ModelCheckPoint/' + current_name + '/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([filepath, date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=10, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

model.summary()
#4. 평가, 예측
loss = model.evaluate(x_train, y_train) 
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)

# y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape) # (6493, 1)

# submission_set = pd.read_csv(path + 'sampleSubmission.csv', # + 명령어는 문자를 앞문자와 더해줌
#                              index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

# print(submission_set)

# submission_set['count'] = y_summit
# print(submission_set)


# submission_set.to_csv(path + 'submission_robust_scaler.csv', index = True)
'''