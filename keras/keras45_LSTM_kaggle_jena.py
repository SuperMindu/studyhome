# LSTM과 Conv1D를 사용해서(RNN, CNN계열 사용) evaluate까지만
# 컬럼은 뭉치로 잘라야 함
# split 함수를 적절히 사용하자
# https://ahnjg.tistory.com/33 <- 
# https://teddylee777.github.io/tensorflow/LSTM%EC%9C%BC%EB%A1%9C-%EC%98%88%EC%B8%A1%ED%95%B4%EB%B3%B4%EB%8A%94-%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90-%EC%A3%BC%EA%B0%80 <- 이건 삼전주가 예측인데 이거 보고 공부해보자
# 기존의 데이터를 가지고 학습시켜서 결국에는 2017년 1월 1일 꺼를 예측 하는 것임
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import time
# 행이 420552개;;;;;;;;;;;

# a = np.array(range(1, 11))                      
# print(a)        
# size = 5                                      

# def split_x(dataset, size):                       
#     aaa = []                                  
#     for i in range(len(dataset) - size + 1):       
#         subset = dataset[i : (i + size)]          
#         aaa.append(subset)                     
#     return np.array(aaa)                      

# bbb = split_x(a, size)                         
# print(bbb)
# print(bbb.shape) 

# x = bbb[:, :-1]                                    
# y = bbb[:, -1]
# print(x, y)
# print(x.shape, y.shape) 

# 1-1) 데이터 로드 
path = './_data/kaggle_jena/'

# "Date Time","p (mbar)","T (degC)","Tpot (K)","Tdew (degC)","rh (%)","VPmax (mbar)","VPact (mbar)","VPdef (mbar)","sh (g/kg)","H2OC (mmol/mol)","rho (g/m**3)","wv (m/s)","max. wv (m/s)","wd (deg)"
data_set = pd.read_csv(path + 'jena_climate_2009_2016.csv')
# print(data_set) # [420551 rows x 15 columns]
# print(data_set.shape) # (420551, 15)

print(type('Date time'))




