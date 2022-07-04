# [과제]
# 3가지 원핫 인코딩 방식을 비교할 것
# 1. pandas의 get_dummies -> 
# 2. tensorflow의 to_categorical -> 무적권 0부터 시작함 (0이 없으면 만듦) (앞을 채워줘야 하는 경우 요놈 쓰면 좋음)
# 3. sklearn의 OneHotEncoder -> 
# 미세한 차이를 찾아라 비밀의 열쇠 

'''
from tensorflow.python.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import to_categorical  <-- keras 에서 

from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder  <-- sklearn 에서

집가서 좀 더 아라보자 (사이킷런 원핫인코딩)
https://daily-studyandwork.tistory.com/36 <- 여기
https://psystat.tistory.com/136 <- 여기 
'''

#***************************************<OneHotEncoding 하는 방법 3가지 정리>**********************************************
# 방법 1. tensorflow.keras 의 to_categorical
from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y)
# print(y)           # y값 보면 그냥 그안에 담긴 값만 딱 나옴
# print(y.shape)     # (178, 3)
# y라벨 값을 0부터 순차적으로 끝까지 변환해줌 

# 방법 2. pandas 의 get_dummies
from pandas import get_dummies
# y = get_dummies(y)
# print(y)           # y값 보면 행 0~177 열 유니크값 대로 0 1 2  마지막에 [178 rows x 3 columns] 까지 표시해줌
# print(y.shape)     # (178, 3)
# y라벨 값을 유니크값 만큼만 변환해주는데 print(y)해보면 라벨값이랑 인덱스정보가 들어가 있음

# 방법 3. sklearn 의 OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
# enco = OneHotEncoder(sparse=False)        
# y = enco.fit_transform(y.reshape(-1,1))
# print(y)            # y값 보면 그냥 그안에 담긴 값만 딱 나옴  
# print(y.shape)      # (178, 3)
# y라벨 값을 유니크값 만큼만 변환해줌



#***************************************<본격적으로 모델을 코딩하기 전 필요한 라이브러리나 데이터셋 등을 불러오는 단계>**********************************************
import numpy as np
from sklearn.datasets import load_wine
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


 
#******************************************<1. 데이터 구성 및 정제 단계>********************************************
# sklearn은 데이터를 다 제공해줌

datasets = load_wine()
print(datasets.DESCR) # 데이터셋 및 컬럼에 대한 설명 
# :Number of Instances: 178 (50 in each of three classes) y의 label 값이 3개 
print(datasets.feature_names)   # 컬럼,열의 이름들 확인

x = datasets.data
y = datasets.target
print(x)
# [[1.423e+01 1.710e+00 2.430e+00 ... 1.040e+00 3.920e+00 1.065e+03]
#  [1.320e+01 1.780e+00 2.140e+00 ... 1.050e+00 3.400e+00 1.050e+03]
#  [1.316e+01 2.360e+00 2.670e+00 ... 1.030e+00 3.170e+00 1.185e+03]
#  ...
#  [1.327e+01 4.280e+00 2.260e+00 ... 5.900e-01 1.560e+00 8.350e+02]
#  [1.317e+01 2.590e+00 2.370e+00 ... 6.000e-01 1.620e+00 8.400e+02]
#  [1.413e+01 4.100e+00 2.740e+00 ... 6.100e-01 1.600e+00 5.600e+02]]
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]  <-- 요딴식으로 나옴. 그래서 바로 밑에 shape를 찍어보면? 
print(x.shape, y.shape) 
# 행과 열의 개수 확인가능 
# (178, 13) (178,) 
# 일단 input_dim을 13으로 해야 하는건 확인가능 
# 그리고 나서 ... 

print(np.unique(y, return_counts=True)) 
# np.unique를 찍어보면?
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64)) 
# [0 1 2] 3개인거확인 
# 이게 뭘 의미한다? 이 모델은 다중분류형태로 모델링 해줘야 한다~
# 따라서 (178,) 으로 나오는 y.shape를 (178, 3) 으로 바꿔줘야 한다~
# 무조건 이 유니크값을 기준으로 개수 생각하고 원핫인코딩 변환도 그렇게 해줘야 함. 암튼 그럼 



#******************************************<1-2. OneHotEncoding>********************************************
# np.unique 찍어보고 그냥 무턱대고 아웃풋 레이어에 3을 주고 실행을 시켜봤더니 'ValueError: Shapes (None, 1) and (None, 3) are incompatible' 이런 오류가 뜸 
# 처음에 y.shape가 (178, ) 이었는데 우선 이걸 (178, 3)으로 바꿔줘야 함
# 즉, 마지막 output레이어에 3을 주고 싶은 건 알겠는데 그 전에 먼저 y의 와꾸를 맞춰줘야 한다는 거임
# 컴터는 문자보다는 숫자를 더 잘 처리 할 수 있음
# 이를 위해 자연어 처리에서는 문자를 숫자로 바꾸는 여러가지 기법들이 있음 
# 원-핫 인코딩(One-Hot Encoding)은 그 많은 기법 중에서 단어를 표현하는 가장 기본적인 표현 방법이며, 
# 머신 러닝, 딥 러닝을 하기 위해서는 반드시 배워야 하는 표현 방법임
# 원핫인코딩은 문자를 숫자, 더 구체적으로는 벡터로 바꾸는 여러 방법 중의 하나임

y = to_categorical(y)  # <-- 자 그래서 y를 OneHotEncoding을 해주고 
print(y) 
print(y.shape) # (178, 3) 으로 바껴 있는거 확인 가능

# 1-3) train과 test 분류
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
# 여기서 잠깐, shuffle=True로 해줘야 하는 이유
# shuffle=False 로 하면 
# print(y_train) # 순차적으로만 나온다
# print(y_test) # 2 '만' 나온다
# 그래서 True로 해줘야 함!

print(x_train.shape, y_train.shape) # (142, 13) (142, 3)
print(x_test.shape, y_test.shape) # (36, 13) (36, 3)
# 마지막으로 x와 y의 train과 test의 shape를 찍어보면 와꾸가 잘 짜여져 있는 걸 확인할 수 있음



#***********************************************<2. 모델 구성 단계>***********************************************
# 대충 알잘딱깔센 하이퍼파라미터튜닝 하자 
# 뭐가 뭔지 잘 모르겠거나 헷갈리면 1주차 총정리 다시 보고 오자
model = Sequential()
model.add(Dense(3, activation='relu', input_dim=13))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='softmax')) 
# 최종 아웃풋 레이어의 개수는 결국 y의 label의 갯수 (찾아야 하는 y의 갯수대로. 분류값에 대한 숫자만큼)
# 다중분류에서는 아웃풋 레이어에 softmax 활성함수를 넣어줘야 함 (중간에는 못 넣음!)
# softmax의 결괏값은 모든 연산의 합이 1.0 으로 나옴 (결괏값 하나 하나는 0.xx 형태로. 그래서 이걸 다 더하면 1)
# 그중에 가장 큰 값을 찾아서 1로 만들어주고 나머지는 0으로 만들어줘야 함 그 이유는 밑에서



#*********************************************<3. 컴파일, 훈련 단계>***********************************************
# 대충 알잘딱깔센 하이퍼파라미터튜닝 
# 뭐가 뭔지 잘 모르겠거나 헷갈리면 1주차 총정리 다시 보고 오자
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=1, validation_split=0.15, callbacks=[es])



#*********************************************<4. 평가, 예측 단계>***********************************************
# 일단 밑에 코딩 
results = model.evaluate(x_test, y_test)
print('accuracy : ', results[1])
print('-------------------------------------')
print('loss : ', results[0])
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1) 
# print(y_test)
y_predict = np.argmax(y_predict, axis=1) 
# print(y_predict)

acc = accuracy_score(y_test, y_predict) # 여기서 test와 pred를 비교해서 acc 값을 뽑아야 함
print('acc 스코어 : ', acc)



#*********************************<4-1. 평가, 예측에 대한 설명 주저리 주저리>****************************************
# predict를 해서 나온 결과는 3개가 출력 됨. 왜?
# np.unique로 유니크값을 찍어서 [0 1 2] 3개 인걸 확인하고 
# OneHotEncoding을 해서 y.shape를 (178, 3) 로 맞춰주고
# 따라서 마지막 노드가 3개이기 때문에 

# 그 3개를 합친 값은 1. 왜?
# 다중분류이기 때문에 마지막 아웃풋 레이어에 들어가는 활성함수가 softmax 이기 때문에 

# y_test의 값은 y = to_categorical(y) 으로 이미 원핫인코딩이 돼 있는 상태임
# [[0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#    . . .
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]] # 대충 요런 형태로 나옴 (n, 3) 의 형태

# 하지만 y_predict는 x_test로 예측해서 나온 것이기 때문에 결과값은 나누어 떨어지지 않음
# [[1.12961698e-03 9.97373819e-01 1.49650523e-03]
#  [6.62750157e-04 9.93067384e-01 6.26994437e-03] 
#  [7.54746667e-04 9.91101980e-01 8.14322289e-03] 
#  [9.99910712e-01 8.92578755e-05 2.34570241e-12] 
#  [8.42118403e-04 9.97301817e-01 1.85605418e-03] 
#         ...           ...            ... 
#  [1.56306440e-03 9.96728778e-01 1.70815433e-03] 
#  [1.62243567e-04 1.97646663e-01 8.02191019e-01] 
#  [1.15605909e-03 8.78377438e-01 1.20466419e-01] 
#  [2.72255943e-06 1.46103837e-02 9.85386848e-01] 
#  [9.99938607e-01 6.13703378e-05 2.14130341e-12] 
#  [4.77243775e-05 7.33535588e-02 9.26598787e-01]] # 대충 요런 형태 (n, 3) 의 형태

# 근데 우리는 이 x_test와 y_test를 가지고 평가 (model.evaluate)를 해줘야 하는데
# 이 두개를 비교하면 비교가 되지 않음. 왜?
# acc 스코어는 딱 떨어지는 정수값을 비교 시켜야 하니까. 왜?
# acc 스코어는 두개를 비교해서 얼마나 잘 맞췄는지를 보여주는 '평가지표'라서.

# 그래서 여기서 numpy 배열에서 가장 높은 값을 가진 값의 인덱스(위치)를 반환해주는 함수인 
# np.argmax 가 필요함 (여기서 np에는 다른 변수값을 넣어도 상관없다고 썜이 그러심) 
# np.argmax 라는 것을 하게 되면 그 위치의 최고값을 숫자로 바꿔줌 그래서 y_test = np.argmax(y_test, axis=1)를 해보면
# [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2] <-- 이렇게 위치 값이 나옴

# (이건 axis 설명)
# y_test = np.argmax(y_test, axis=1) 여기서 axis=1 이 나오는데 axis=1은 열을 따라 열을 기준으로 최대값을 뽑아준다는 뜻임
# 자 이 밑에 x_test로 예측해서 나온 y_predict 값이 있는데
# [[1.12961698e-03 9.97373819e-01 1.49650523e-03]
#  [6.62750157e-04 9.93067384e-01 6.26994437e-03] 
#  [7.54746667e-04 9.91101980e-01 8.14322289e-03] 
#  [9.99910712e-01 8.92578755e-05 2.34570241e-12] 
#  [8.42118403e-04 9.97301817e-01 1.85605418e-03] 
# 만약에 axis=0을 해주면 행을 기준으로 행을 따라 최대값을 가진 위치를 뽑아주기 때문에  
# 쉽게 말해서 위에 y_predict 값들 나열된 와꾸 기준으로 세로로 쭉 보고 [3 1 0] 요딴 식으로 뽑아줌 
# 근데 우리는 그렇게 해서는 안됨 
# axis=1로 해서 y_predict = np.argmax(y_predict, axis=1) 이렇게 해서 열을 따라 열을 기준으로 
# y_predict 값도 
# [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] <-- 이렇게 위치 값을 뽑아줘야 
# 비로소 y_test와 y_predict 값이 비교가 가능해지는 거임
# 결국 argmax를 2번을 해줘야 한다는 뜻 ㅎ

