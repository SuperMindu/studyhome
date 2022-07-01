import numpy as np
import pandas as pd   # 엑셀 데이터 불러올 때 사용
from pandas import DataFrame 
import time
from tensorflow.python.keras.models import Sequential # 함수는 소문자로 시작, Class는 대문자로 시작 
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = './_data/kaggle_house/'  # .은 현재폴더라는 뜻
train_set = pd.read_csv(path + 'train.csv')  # train.csv의 데이터들이 train_set에 수치화 돼서 들어간다 
print(train_set)
print(train_set.shape)  # (1460, 81)



test_set = pd.read_csv(path + 'test.csv')  #예측에서 씀
                    #    index_col=0)  index_col=n  (n번째 컬럼을 인덱스로 인식)               
print(test_set)
print(test_set.shape)  # (1459, 79)


submission = pd.read_csv(path + 'sample_submission.csv') # 일단 이거를 읽어와야 함
submission_set = pd.read_csv('./_data/kaggle_house/sample_submission.csv', index_col=0)


# 사이킷런의 알고리즘은 문자열 값을 입력 값으로 허용하지 않음
# 그래서 모든 문자열 값은 인코딩 돼서 숫자 형으로 변환해야 함 
# 문자열 피처는 일반적으로 카테고리형 피처와 텍스트형 피처를 의미
# 카테고리형 피처는 코드 값으로 표현하는 게 더 이해하기 쉬울 듯?
# 텍스트형 피처는 피처 벡터화등의 기법으로 벡터화하거나 불필요한 피처라고 판단되면 삭제하는 게 좋음
# 예를 들어 주민번호나 단순 문자열 아이디와 같은 경우 인코딩하지 않고 삭제하는 게 더 좋음
# 이러한 식별자 피처는 단순히 데이터 로우를 식별하는 용도로 사용되기 때문에 예측에 중요한 요소가 될 수 없으며 
# 알고리즘을 오히려 복잡하게 만들고 예측 성능을 떨어뜨리기 때문
# 머신러닝을 위한 대표적인 인코딩 방식은 '레이블 인코딩(Label Encoding)'과 원-핫 인코딩이 있당


#df1
print(train_set.columns)
print(train_set.info())  # 결측치 = 이빨 빠진 데이터
print(train_set.describe())  # [8 rows x 38 columns]
# describe - 데이터 프레임 컬럼별 카운트, 평균, 표준편차, 최소값, 4분위 수, 최대값을 보여줌. 문자로 구성된 컬럼은 무시.


#df2
print(test_set.columns)
print(test_set.info())  # 결측치 = 이빨 빠진 데이터
print(test_set.describe())  # [8 rows x 36 columns]



'''
#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum()) #null의 합계를 구함
train_set = train_set.dropna()
test_set = test_set.fillna(test_set.mean())
print(train_set.isnull().sum()) 
print(train_set.shape)  # 



x = train_set.drop([''], axis=1)
print(x)
print(x.columns)
print(x.shape)  # 


y = train_set['SaleType']
print(y)
print(y.shape)  # 

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=300)
'''