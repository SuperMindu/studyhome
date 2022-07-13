from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, LabelEncoder # <-- 라벨인코더는 트레인과 테스트를 합쳐서 해주고 다 한 뒤에 분리 해줌
from keras.layers import BatchNormalization
from sklearn.ensemble import RandomForestRegressor # 회귀
from sklearn.ensemble import RandomForestClassifier # 분류
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감#######################


#1. 데이터
path = './_data/dacon_shopping/'
train_set = pd.read_csv(path + 'train.csv' # + 명령어는 문자를 앞문자와 더해줌
                        ) # index_col=n n번째 컬럼을 인덱스로 인식

Weekly_Sales = train_set[['Weekly_Sales']]
print(train_set)
print(train_set.shape) # (6255, 12)

test_set = pd.read_csv(path + 'test.csv' # 예측에서 쓸거임                
                       )
print(test_set)
print(test_set.shape) # (180, 11)

print(train_set.columns)
print(train_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

train_set.isnull().sum().sort_values(ascending=False) # <----------------------------------------------------------------- 질문할거 (null값을 합쳐주는 거 까진 알겠는데)
test_set.isnull().sum().sort_values(ascending=False)
# sort_values : 값을 기준으로 정렬하는 메소드 
# 내림차순 : ascending=False
# 디폴트 True는 오름차순


######## 년, 월 ,일 분리 ############

# train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.Date)]
# train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.Date)]
# train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.Date)]

# test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.Date)]
# test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.Date)]
# test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.Date)]

# train_set.drop(['id', 'Date','Weekly_Sales'],axis=1,inplace=True) # (이미 id는 없애줬고, Date는 년월일 분리를 해줬고, Weekly_Sales도 없애줬기 때문에) 트레인 세트에서 원래 컬럼들을 드랍
# test_set.drop(['id', 'Date'],axis=1,inplace=True) # 이하동문



# Date 컬럼을 자료형으로 변환
train_set['Date'] = pd.to_datetime(train_set['Date'])
train_set.dtypes
train_set['Year'] = train_set['Date'].dt.year
train_set['Month'] = train_set['Date'].dt.month
train_set['Day'] = train_set['Date'].dt.day
print(train_set)

test_set['Date'] = pd.to_datetime(test_set['Date'])
test_set.dtypes
test_set['Year'] = test_set['Date'].dt.year
test_set['Month'] = test_set['Date'].dt.month
test_set['Day'] = test_set['Date'].dt.day
print(test_set)

train_set.drop(['id', 'Date','Weekly_Sales'],axis=1,inplace=True) # (이미 id는 없애줬고, Date는 년월일 분리를 해줬고, Weekly_Sales도 없애줬기 때문에) 트레인 세트에서 원래 컬럼들을 드랍
test_set.drop(['id', 'Date'],axis=1,inplace=True)





# print(train_set)
# print(test_set)

##########################################

# ####################원핫인코더###################

# df = pd.concat([train_set, test_set]) # pd.concat 은 합쳐주는 거 (train_set과 test_set를 합쳐서 df를 만들어줌)
# print(df)

# alldata = pd.get_dummies(df, columns=['day','Store','month', 'year']) # 문자열로 돼 있는 컬럼들을 원핫인코딩으로 숫자로 바꿔주고 alldata로 지정해줌
# print(alldata)

# train_set2 = alldata[:len(train_set)] # <---------------------------------------------------------- 다시 한번 질문할거
# test_set2 = alldata[len(train_set):]

# print(train_set2)
# print(test_set2)

df = pd.concat([train_set, test_set]) # pd.concat 은 합쳐주는 거 (train_set과 test_set를 합쳐서 df를 만들어줌)
print(df)

train_set2 = df[:len(train_set)] # <---------------------------------------------------------- 다시 한번 질문할거
test_set2 = df[len(train_set):]

print(train_set2)
print(test_set2)
# train_set = pd.get_dummies(train_set, columns=['Store','month', 'year', 'IsHoliday'])
# test_set = pd.get_dummies(test_set, columns=['Store','month', 'year', 'IsHoliday'])




###############프로모션 결측치 처리###############

# train_set2 = train_set2.fillna(train_set2.mean())
# test_set2 = test_set2.fillna(test_set2.mean())
train_set2 = train_set2.fillna(0) # 빈칸(결측치)을 걍 다 0으로 채움
test_set2 = test_set2.fillna(0)

print(train_set2)
print(test_set2)

##########################################

train_set2 = pd.concat([train_set2, Weekly_Sales],axis=1) # train_set2는 train_set2와 Weekly_Sales를 합쳐줌
print(train_set2)

x = train_set2.drop(['Weekly_Sales'], axis=1) # x는 train_set2에서 Weekly_Sales를 뺀거 
y = train_set2['Weekly_Sales'] # y는 train_set2에서 Weekly_Sales만 사용


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.85,
                                                    random_state=66
                                                    )


# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_set2 = scaler.transform(test_set2)

# # print(test_set2)
# print(x_train.info())
# print(y_train.info())
# print(train_set2.info())

x_train = x_train.values # <--------------------------------------------------------------------------- 질문할거
y_train = y_train.values

x_train.astype('int') # <------------------------------------------------------------------------------ 질문할거 (x_train 데이터의 타입을 정수로 바꿔준다는 의미?)
y_train.astype('int')

print(x_train.shape)
print(y_train.shape)


# 2. 모델구성
# 작은 max_features와 큰 n_estimators 는 과적합을 줄인다는 장점이 있음
# 랜덤 포레스트 모델을 쓸 때는 n_estimator 인자와 max_features인자를 조절하며
# 결정계수가 가장 높아지는 인자를 사용하는 것이 좋음
# 분류는 max_features=sqrt(n_features), 회귀는 max_features=n_features

model = RandomForestRegressor(n_estimators = 80,  
                              random_state = 15)
model.fit(x_train, y_train)
relation_square = model.score(x_test, y_test)
print('결정계수 : ', relation_square)

# max_features 안쓴거
# 10 : 0.9879816979500338
# 20 : 0.9898843110031285
# 30 : 0.9901591302042241
# 40 : 0.9904738613600972
# 50 : 0.9906250542416528
# 60 : 0.9908900171120611
# 70 : 0.9908099932010348
# 80 : 0.9911022403187818
# 90 : 0.9909545107072387
# 100 : 0.9910471489372621


















# print(model.score(x_train, y_train))
# print(model.score(x_test, y_test))


''' # <--------------------------------------------------------------------------- 질문할거 (랜덤 포레스트는 컴파일, 훈련을 따로 안 해줘도 되나?)
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)
save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/' + current_name + '/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )
hist = model.fit(x_train, y_train, epochs=3000, batch_size=128,
                 validation_split=0.3,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
# model = load_model(load_filepath + '0711_1732_2300-8791202816.발리데이션0.3.hdf5')
'''


'''
#4. 평가, 예측

print("=============================1. 기본 출력=================================")
loss = model.score(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)

print(test_set2)

y_summit = model.predict(test_set2)

print(y_summit)
print(y_summit.shape) # (180, 1)

submission_set = pd.read_csv(path + 'sample_submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

print(submission_set)

submission_set['Weekly_Sales'] = y_summit
print(submission_set)


submission_set.to_csv(path + 'submission_randomforest.csv', index = True)
'''