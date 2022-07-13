from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd   # 엑셀 데이터 불러올 때 사용
from pandas import DataFrame 
import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.utils import to_categorical 
'''
'C:/study/study-home/_data/dacon_shopping/'  <- 집에서 할때 데이터 파일 경로
'''



#1. 데이터
path = './_data/dacon_shopping/'  # .은 현재폴더라는 뜻
train_set = pd.read_csv(path + 'train.csv',  # train.csv의 데이터들이 train_set에 수치화 돼서 들어간다 
                        index_col=0)  # index_col=n. n번째 컬럼을 인덱스로 인식



print(train_set) # [6255 rows x 12 columns]
print(train_set.shape)  # (6255, 12)


test_set = pd.read_csv(path + 'test.csv',  #예측에서 씀
                       index_col=0)

submission = pd.read_csv(path + 'sample_submission.csv') # 일단 이거를 읽어와야 함
submission_set = pd.read_csv('./_data/dacon_shopping/sample_submission.csv', index_col=0)

print(test_set) # [180 rows x 11 columns]
print(test_set.shape)  # (180, 11)

print(train_set.columns)
print(train_set.info()) 

# Non-Null Count 부분을 보니 promotion 부분에 결측치
# date, isholiday 부분은 숫자가 아니기 때문에 분석 전에 데이터 전처리 필요
# 결측치(promotion), date, isholiday 전처리 필요함 

print(train_set.describe())  # 

#### 결측치 처리  ####
print(train_set.isnull().sum()) # null의 합계를 구함
train_set = train_set.fillna(0)
test_set = test_set.fillna(0)
print(train_set.isnull().sum()) 
print(train_set.shape)  # (6255, 12)


def get_month(date):
    month = date[3:5]
    month = int(month)
    return month

train_set['Month'] = train_set['Date'].apply(get_month)

x = train_set # .drop - 데이터에서 ''사이 값 빼기, # axis=1 (열을 날리겠다), axis=0 (행을 날리겠다)
print(x)
print(x.columns)
print(x.shape)  # (6255, 13)

y = train_set['Weekly_Sales']
print(y)
print(y.shape)  # (1368,)



x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, shuffle=True, random_state=31)




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
es = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=500, batch_size=10, verbose=1, validation_split=0.15, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict): #괄호 안의 변수를 받아들인다 :다음부터 적용 
    return np.sqrt(mean_squared_error(y_test, y_predict)) #루트를 씌워서 돌려줌 

rmse = RMSE(y_test, y_predict)  #y_test와 y_predict를 비교해서 rmse로 출력 (원래 데이터와 예측 데이터를 비교) 
print("RMSE : ", rmse)



# y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)  # (715, 1)  # 이거를 submission.csv 파일에 쳐박아야 한다

# submission_set['count'] = y_summit
# print(submission_set)
# submission_set.to_csv('ddarung.csv', index=True)

# end_time = time.time() - start_time
# print("걸린시간 : ", end_time)










'''
y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape) # (715, 1)

submission_set = pd.read_csv(path + 'submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식


submission_set['count'] = y_summit
submission_set.to_csv('./_data/ddarung/submission.csv', index = True)



#### .to_csv() 를 사용해서 
### submission.csv를 완성하시오 

# Non-Null Count 부분을 보니 promotion 부분에 결측치가 있다는 것을 알 수 있습니다.
# date, isholiday 부분은 숫자가 아니기 때문에 분석 전에 데이터 전처리가 필요하겠네요.
# 결측치(promotion), date, isholiday 전처리 필요함 

#1. 데이터 
path = './_data/dacon_shopping/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
print(train_set) # [6255 rows x 12 columns]
print(train_set.shape) # (6255, 12)

test_set = pd.read_csv(path + 'test.csv', index_col=0)
print(test_set) # [180 rows x 11 columns]
print(test_set.shape) # (180, 11)

print(train_set.columns)


print(train_set.info())
# Int64Index: 6255 entries, 1 to 6255
# Data columns (total 12 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   Store         6255 non-null   int64
#  1   Date          6255 non-null   object
#  2   Temperature   6255 non-null   float64
#  3   Fuel_Price    6255 non-null   float64
#  4   Promotion1    2102 non-null   float64
#  5   Promotion2    1592 non-null   float64
#  6   Promotion3    1885 non-null   float64
#  7   Promotion4    1819 non-null   float64
#  8   Promotion5    2115 non-null   float64
#  9   Unemployment  6255 non-null   float64
#  10  IsHoliday     6255 non-null   bool
#  11  Weekly_Sales  6255 non-null   float64
# dtypes: bool(1), float64(9), int64(1), object(1)

print(train_set.describe())

x = train_set.drop(['id'], axis=1) 
print(x)
print(x.columns)
print(x.shape) # (1459, 9)

y = train_set['Weekly_sales']
print(y)
print(y.shape)
'''
