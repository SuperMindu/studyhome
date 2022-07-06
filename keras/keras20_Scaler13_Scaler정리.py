from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
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



#***************************************<Scaler 설정 옵션>**********************************************
# 전체 데이터를 전부 스케일링을 하면 데이터 범위 밖의 값을 예측할 때 과적합이 발생할 수 있기 때문에 
# train_test_split을 먼저 해주고 나서 x_train과 x_test를 따로따로 스케일링 시켜줘야 함

# scaler = MinMaxScaler()   # 어떤 스케일러 사용할건지 정의부터 해주고
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)       #어떤 비율로 변환할지 계산해줌.   여기서 구한 비율로 transform해준다.
x_train = scaler.transform(x_train)   
# 훈련할 데이터 변환  
# 원래대로라면 이러한 전처리는 컬럼 별로 일일이 해줘야 하지만, 이미 sklearn.preprocessing의 각 Scaler 클래스에서 다 제공함
# train 데이터를 먼저 스케일링 시켜서 fit을 시키고 거기서 나온 (결과, 수식, 비율)대로 test 데이터를 다시 스케일링 시켜야 함
# train을 -> fit -> transform 
# 이와 동일한 규칙으로 test와 val을 스케일링 시켜야 함
 
x_test = scaler.transform(x_test)    
# test할 데이터도 비율로 변환. 설령 스케일링 밖의 값을 받아도 이미 weight를 구했으므로 예측값이 나옴



#***************************************<Scaler 4가지 정리>**********************************************
#  StandardScaler
# 평균을 제거하고 데이터를 단위 분산으로 조정함. 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 됨.
# 따라서 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없음

#  MinMaxScaler
# 모든 feature 값이 0~1사이에 있도록 데이터를 재조정함. 다만 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있음
# 즉, MinMaxScaler 역시 아웃라이어의 존재에 매우 민감함

#  MaxAbsScaler
# 절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정함. 
# 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있음

#  RobustScaler
# Outlier(이상치)의 영향을 최소화한 기법. 
# 중앙값(median)과 IQR(interquartile range)을 사용하기 때문에 StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있음
# IQR = Q3 - Q1 : 즉, 25퍼센타일과 75퍼센타일의 값들을 다룬다.




# print(np.min(x_train)) # 0.0
# print(np.max(x_train)) # 1.0 (1.0000000000000002 <- 이딴 식으로 나옴)
# # 이미 컬럼별로 나눠서 스케일링이 돼 있는 상태임
# print(np.min(x_test)) # -0.06141956477526944
# print(np.max(x_test)) # 1.1478180091225068 (이 정도의 오차는 존재해야 함)