'''
1주차 (6/15 ~ 7/1)
데이콘 따릉이 코딩을 기반으로 진도, 실습, 과제 등에서 나왔던 모든 내용들을 전부 싸그리 통째로 넣고 하나하나 주석 달아서 설명
'''



#***************************************<본격적으로 모델을 코딩하기 전 필요한 라이브러리나 데이터셋 등을 불러오는 단계>***********************************************
import numpy as np 
# numpy는 행렬이나 대규모 다차원 배열을 쉽게 처리할 수 있도록 지원하는 파이썬의 라이브러리임
# 벡터나 행렬연산에 있어서 엄청난 편의성을 제공함
# 밑에 나올 pandas나 matplotlib 등의 기반이 되는 라이브러리임
# 기본적으로 array(배열)라는 단위로 데이터를 관리함
# 고등수학의 행렬과 유사한 부분이 많음

from tensorflow.python.keras.models import Sequential 
# tensorflow.python.keras.models 에서 Sequential 이라는 모델을 불러온다는 뜻
# Sequential은 순차적이라는 뜻 즉 Sequential 모델(순차 모델)을 쓸 수 있음
# Sequential 모델은 각 레이어에 정확히 하나의 입력 텐서와 하나의 출력 텐서가 있는 일반 레이어 스택에 적합함 (대충 우리가 지금 하고 있는 모델 구성 방식임)
# Sequential 모델은 모델에 다중 입력 또는 다중 출력이 있거나, 레이어에 다중 입력 또는 다중 출력이 있거나, 레이어 공유를 해야하거나 하는 경우는 적합하지 않음

from tensorflow.python.keras.layers import Dense 
# 신경망을 이해할 때 사용하는 모듈임
# 다양한 함수와 hidden layer(은닉층)을 거쳐서 나온 숫자들을 한 곳으로 모아주고, 태스크의 적절한 함수에 정보를 전달하기 위한 레이어라고 보면 편함
# Dense Layer는 여러 Layer로부터 계산된 정보들을 한 곳으로 모은 자료임
# 더 쉽게 말하면 input을 넣었을 때 output으로 바꿔주는 중간 다리임
# 한마디로 그냥 신경망 만드는 거임 

from sklearn.model_selection import train_test_split
# 사이킷런에서 제공하는 model_selection 모듈은 train data 와 test data 세트를 분리하거나 교차 검증, 분할, 평가, 하이퍼 파라미터 튜닝을 위한 다양한 함수나 클래스를 제공함
# 복잡한 딥러닝 모델을 쉽게 작성할 수 있도록 해주는 라이브러리임 
# import train_test_split 는 말 그대로 학습과 테스트 데이터셋을 분리 해주는 기능을 불러온다는 뜻
# tmi (데이터를 train 과 test로 나눠주는 이유)
# fit에서 모델 학습을 훈련시킬 때 모든 데이터를 다 학습 시켜버리면 예측 단계 -> model.predict[] 에 실제로 원하는 미래의 데이터를 넣어봤을 떄 크게 오류가 날 수 있음 
# 왜냐하면 컴퓨터는 주어진 모든 값으로 훈련'만' 하고 실전을 해본 적이 없기 때문임
# 그래서 train과 test로 나눠서 train으로 학습을 시키고 test로 실전같은 모의고사를 한번 미리 해보면 fit단계에서의 loss값과 evaluate의 loss값의 차이가 큰걸 확인할 수 있음
# 근데 아직 확인까지만 가능하고 그 이상은 뭐 할 수 없음
# 여기서 나온 loss값과 fit 단계의 loss값들의 차이가 크다 하더라도 그 차이가 fit 단계에 적용되지는 않음

import matplotlib.pyplot as plt
# matplotlib은 데이터를 시각화 해주는 기능을 제공하는 라이브러리임

from sklearn.metrics import r2_score
# R2 score, R2 제곱
# '선형회귀모델'에 대한 적합도 측정값임
# 0~1점 만점. 1에 가까울수록 정확한 값
# 음수도 나올 수 있음. 자세한 공식은 아직 내 레벨로는 알기에 부족함
# loss만 가지고 정확도를 보기에는 부족함이 있어서 R2 score로 점수를 메김

from sklearn.metrics import accuracy_score
# '분류모델'에서 씀
# accuracy (정확도)는 실제 label과 예측 label이 일치하는 비율을 나타내는 지표임
# accuracy_score 함수는 단순히 예측값과 결괏값을 비교해서 얼마나 일치하는지 비교함
# 가장 대표적인 머신러닝 성능평가 지표임

from sklearn.metrics import mean_squared_error
# sklearn의 metrics에서 제공하는 mean_squared_error 함수 기능임
# MSE (Mean squared error) (평균 제곱 에러)
# 전체 에러를 표현하기 위해서 사용하는 식
# 오차의 제곱에 대해 평균을 취한 것
# 수치가 작을 수록 원본과의 오차가 적음

import time

import pandas as pd   
# pandas는 데이터 프레임 자료구조를 사용함
# 즉, 엑셀의 스프레드시트와 유사해 데이터 처리가 쉬움
# 엑셀 데이터(csv파일) 등을 불러올 때 사용함 (Dakon이나 kaggle의 대회 데이터를 받아서 불러올 때 필요함)

from tensorflow.python.keras.callbacks import EarlyStopping  
# callbacks 모듈에는 웬만한 인공지능 기능은 이미 다 구현이 되어있음. import해서 쓰면 됨
# EarlyStopping 이건 클래스임. 그래서 대문자로 시작 (소문자로 시작하는 건 함수임)
# training 조기종료를 도와주는 기능임. 여러 옵션들이 있음





#******************************************<1. 데이터 구성 및 정제 단계>********************************************
path = 'C:/study/study-home/_data/ddarung/'  
# 경로 설정
# .은 현재폴더라는 뜻
train_set = pd.read_csv(path + 'train.csv', index_col=0)  
# train.csv의 데이터들이 train_set에 수치화 돼서 들어간다 
# index_col=n. n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape)  # (1459, 10)

test_set = pd.read_csv(path + 'test.csv', index_col=0) # 예측에서 씀
submission = pd.read_csv(path + 'submission.csv') # 일단 이거를 읽어와야 함
submission_set = pd.read_csv('C:/study/study-home/_data/ddarung/submission.csv', index_col=0)
print(test_set)
print(test_set.shape)  # (715, 9)

print(train_set.columns)
print(train_set.info())  # 결측치 = 이빨 빠진 데이터

#### 결측치 처리 1. 제거 ####
# 결측치를 처리하는 방법은 여러가지가 있는데 일단은 제거하는 방법만 배움
print(train_set.isnull().sum()) #null의 합계를 구함
train_set = train_set.dropna()
test_set = test_set.fillna(test_set.mean())
print(train_set.isnull().sum()) 
print(train_set.shape)  # (1328, 10)

x = train_set.drop(['count'], axis=1) 
# .drop([]) 데이터에서 '' 사이 값 빼기
# axis=1 (열을 날리겠다), axis=0 (행을 날리겠다)
print(x)
print(x.columns)
print(x.shape)  # (1459, 9)

y = train_set['count']
print(y) # 만약 결과값이 0,1 밖에 없는걸 본다면 보는 순간 2진분류인거 판단 + loss 값 binary_crossEntropy랑 sigmoid() 함수인거 까지 자동으로 생각하자
print(y.shape)  # (1459,) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=31)





#***********************************************<2. 모델 구성 단계>***********************************************
# layer와 parameter를 추가하여 deep learning으로 만들어짐
# 레이어층을 두껍게해서 다중신경망을 형성하여 그 뒤 컴파일하고 예측을 해보면 단일신경망일 때에 비하여 훈련량 (epochs)을 훨씬 줄여도 loss값을 구할 수 있음
model = Sequential()
model.add(Dense(1, activation='알맞은 함수를 넣자', input_dim=1))
model.add(Dense(1, activation='알맞은 함수를 넣자')) 
model.add(Dense(1, activation='알맞은 함수를 넣자'))
model.add(Dense(1, activation='알맞은 함수를 넣자'))
model.add(Dense(1, activation='알맞은 함수를 넣자'))
# #activation='linear' -> 원래값을 그대로 넘겨줌. linear는 그냥 선. 직선으로 그냥 이어줌
# 레이어에 노드와 노드 사이를 통과하면서 그냥 계속 연산만 하다보면 값이 너무 커져서 터짐
# sigmoid함수는 레이어를 거치며 커져버린 y=wx+b의 값들을 0.0 ~ 1.0사이로 잡아주는 역할을 함 (0 ~ 1 사이로 한정시킴)
# 값이 너무 큰 거 같으면 히든레이어 중간에 sigmoid를 한번 써서 줄여줄 수도 있음
# 이진분류에서는 마지막 아웃풋에서 0 과 1로 나와야 하기때문에 sigmoid는 마지막 아웃풋 레이어에 무조건 써줘야 함
# 하지만 sigmoid는 1 ~ 0 사이로 한정시키기 때문에 0과 1로 분류해주기 위해 반올림을 해줘야한다 (0.5가 기준. 이상이면 1, 미만이면 0)
# 분류모델에서 이진분류에서는 마지막 아웃풋 레이어에 sigmoid
# activation='relu' 가 킹왕짱 (중간 히든레이어 에서만 쓸 수 있음) (음수는 0으로 만들어버림) (나머지는 원래 값을 그대로 넘겨주는 linear와 동일)
# 회귀모델 activation = linear (default값) 이진분류는 sigmoid
# 다중분류 에서는 ... 





#*********************************************<3. 컴파일, 훈련 단계>***********************************************
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])
# 선형회귀모델 에서는 loss를 mse, mae 등을 씀 
# 이진분류모델 에서는 loss를 당분간 내 레벨에서는 'binary_crossentropy' 이거 하나만 씀
# optimizer='adam'은 loss를 감축시키는 역할. 보통 85퍼 이상이므로 꽤 괜찮음 
# metrics=['']는 평가지표. metrics=['accuracy']를 넣어주면 훈련할 때 verbose에서 같이 보여줌, list형식이라서 mse 같은 것들도 추가적으로 더 넣어줄 수 있음. 하지만 이진분류에서는 mse는 신뢰하지 않음
# matrics=['accuracy']는 그냥 r2 스코어처럼 지표를 보여주는거지 fit에 영향을 끼치지는 않음
# loss가 제일 중요함!  accuracy는 그냥 평가지표임. 몇개 중에 몇개 맞췄는지 보여주는 지표 ㅇㅇ
# 설령 잘 맞췄다 해도 loss값이 크면 운으로 때려맞춘거지. 좋다고 보장할 순 없음
# loss와 val_loss를 따지면 val_loss가 더 믿을만 함

es = EarlyStopping(monitor='val_loss', patience=1, mode='min', verbose=1, restore_best_weights=True)
# 아무거나 = EarlyStopping 이라고 선언을 해줘야 사용할 수 있음. 난 es로 쓸거임
# monitor='' 에서 ''에 쓰는 것 기준으로 모니터링을 하며 중지시킴
# monitor= 에서 쓸 수 있는건 fit 에서 제공하는 것만 가능
# 멈추는 시점은 최소값 발견 직후 patience값에서 멈춤 
# mode= 여기는 max도 있음(monitor='accuracy' 같이 높아질 수록 정확한 값들을 측정하기 위함). auto로 하면 알아서 잡아줌 개꿀
# restore_best_weights 라는 옵션이 있는데 이 값에 True, False를 넣어줘서 저장할 weights 값을 선택할 수 있음
# False일 경우 마지막training이 끝난 후의 weight값을 저장하고 True라면 training이 끝난 후 값이 가장 좋았을때의 weight로 복원함
# 또한 baseline 옵션을 사용하여 모델이 달성해야하는 최소한의 기준값을 선정할 수도 있음

start_time = time.time() # <- 이거 넣어주면 여기서부터 시간 재기 시작함

hist = model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_split=0.1, 
                 callbacks=[es], # <- 여기도 리스트형식 이라서 더 넣을 수 있음 (떡밥)
                 verbose=1) #verbose = 매개변수 (중간 훈련과정 스킵, 시간 단축 가능)
# 지금까지 배운 기존의 Train & Test 시스템은 훈련 후 바로 실전을 치뤄서 실전에서 값이 많이 튀었음
# 이걸 더 보완하기위해서 Train 후 validation(검증) 거치는 걸 1 epoch로 하여 계속반복하며 값을 수정 후
# 여기서 validation에서 나오는 val_loss가 fit에 직접적으로 관여하지는 않지만 방향성을 제시해줌
# 그 뒤에 실전 Test를 치루는 방식으로 더 개선시킴. 앞으로는 데이터를 받으면 train -> validation -> test 
# hist 이 값을 가지고 최저값 최소의 loss 최적의 weight 를 찾는다 
# loss와 val_loss의 차이가 작은 게 좋음 !!! (데이터 시각화 단계에서 눈으로 쉽게 확인 가능)

end_time = time.time() - start_time # <- 여기서 시간측정 끝





#*********************************************<4. 평가, 예측 단계>***********************************************
loss = model.evaluate(x_test, y_test)
# evaluate의 평가값은 1개로 귀결됨. loss를 출력해보면 값이 1개만 나옴 ... 그랬는데
# 근데 matrics=['accuracy'] 써서 정확도를 출력하니까 갑자기 값이 2개가 나옴
# 첫번재 값은 loss고 두번째 값은 matrics=['accuracy'] 값이 출력됨
# [ ] 는? 리스트. 안에 값을 여러개담을때 사용함. evaluate 값 출력했을 때 첫번째 값은 무조건 loss나오고 그 후의 값은 사전에 설정해준 값들이 순차적으로 나옴
# loss라는 이름안에 evaluate의 값들을 저장. loss가 따로있는 게 아님
# 컴파일 단계에서 도출된 weight 및 bias값에 xy_test를 넣어서 평가만. 해보고 거기서 나온 loss값들(?)을 저장함

print('loss : ', loss)
# 평가'만' 해본 후 나온 loss의 값임
# val_loss와 loss의 차이가 적을수록 validation data가 더 최적의 weights를 도출시켜줘서 실제로 평가해봐도 차이가 적게 나온다는 말이므로 차이가 적을수록 좋음
# model.evaluate도 model.fit처럼 수많은 값들을 loss안에 담아주는 줄 알았다 근데 보려고했더니
# print(loss.history) loss도 hist처럼 history를 볼 수 있을줄 알았는데 'float' object has no attribute 'history' ('float' 개체에 'history' 속성이 없습니다.) 라고 나옴
# numpy의 주료 자료형 중에 float는 실수라는 뜻

print('-------------------------------------------')
print(hist) # <tensorflow.python.keras.callbacks.History object at 0x000002B1B6638040>
print('-------------------------------------------')
print(hist.history) # loss 값과 var_loss값이 딕셔너리 형태로 저장되어 있음. epoch 값만큼의 개수가 저장되어 있움 -> 1epoch당 값을 하나씩 다 저장한다.
print('-------------------------------------------')
print(hist.history['loss']) # hist.history에서 loss키 값의 value들을 출력해줌 # 키, 밸류 상의 로스는 문자이므로 '' 표시 
print('-------------------------------------------')
print(hist.history['val_loss']) # hist.history val_loss키 값의 value들을 출력해줌
print('-------------------------------------------')
print("걸린시간 : ", end_time)

y_predict = model.predict(x_test) 
# y의 예측값은 x의 테스트 값에 wx + b
# model.predict의 값은 모델링에 따라서 크게 영향을 받음. 모델링 못하면 값이 전부 똑같이 나올수도 있음. 하이퍼 파라미터 튜닝을 잘 하자

r2 = r2_score(y_test, y_predict) # 계측용 y_test값과, y예측값을 비교한다
print('r2스코어 : ', r2)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)

def RMSE(y_test, y_predict): # 괄호 안의 변수를 받아들인다 :다음부터 적용 
    return np.sqrt(mean_squared_error(y_test, y_predict)) # 루트를 씌워서 돌려줌 
# 이건 그냥 데이콘에서 따릉이 문제를 풀 때 평가지표를 RMSE 로 한다고 해서 함수로 rmse를 만들어 준거임

rmse = RMSE(y_test, y_predict)  #y_test와 y_predict를 비교해서 rmse로 출력 (원래 데이터와 예측 데이터를 비교) 
print("RMSE : ", rmse)

y_summit = model.predict(test_set) 
print(y_summit)
print(y_summit.shape)  # (715, 1)  # 이거를 submission.csv 파일에 쳐박아야 한다

submission_set['count'] = y_summit
print(submission_set)
submission_set.to_csv('ddarung.csv', index=True) # .to_csv()를 사용해서 submission.csv 파일을 완성해줌

print("걸린시간 : ", end_time)





#*********************************************<데이터 시각화 단계>**********************************************
plt.figure(figsize=(9,6)) # (9 x 6 으로 그리겠다는 뜻) (figure <- 이거 알아서 찾아보기)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # marker '_' 로 하면 선, 아예 삭제해도 선으로 이어짐
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 보기 편하게 하기 위해서 모눈종이 형태로 
plt.title('SuperMindu') # 표 제목 
plt.ylabel('loss') # y축 이름 설정
plt.xlabel('epochs') # x축 이름 설정 
plt.legend(loc='upper right')  # 위에 label 값이 이 위치에 명시가 된다 (디폴트 값이 upper right)
plt.show()
# 그림도표를 보면서 과적합이 어떨 때 일어나는지 생각해보자
# 많이 한다고 좋은게 아니다 그래프를 보면 값들이 줄어들었다가 팡 튀고 줄어들었다가 팡 튀고 하는 경우가 있음
# 계속 여러번 돌려보면서 loss와 val_loss 격차가 많이 줄어가는걸 보면서 epoch량을 조절함
# val_loss가 최저점이다라는 말의 뜻은 y = wx + b 예측을 가장 잘했다는 뜻





#*********************************************<따로 추가>**********************************************
# 행렬 연습
# [ ] 개수대로 거슬러 올라가서 제일 작은 [ ] 안에 있는 값이 열이 됨
# 그 다음부터 순차적으로 열 행 다중행렬 다중텐서 순
import numpy as np

a1 = np.array( [ [1,2], [3,4] , [5,6]] )
a2 = np.array( [ [1,2,3], [4,5,6] ] )
a3 = np.array( [ [[ [[1],[2],[3]] , [[1],[1],[1]]]] ] )
a4 = np.array( [ [[1,2], [3,4]] , [[5,6],[5,6]] ] )
a5 = np.array( [ [[1,2,3] , [4,5,6]] ])
a6 = np.array( [ 1,2,3,4,5] )

# print(a1,a1.shape) 
# [[1 2]
#  [3 4]
#  [5 6]] (3, 2)

# print(a2,a2.shape)
# [[1 2 3]
#  [4 5 6]] (2, 3)

# print(a3,a3.shape)
# [[[[[1]
#     [2]
#     [3]]

#    [[1]
#     [1]
#     [1]]]]] (1, 1, 2, 3, 1)

# print(a4,a4.shape)
# [[[1 2]
#   [3 4]]

#  [[5 6]
#   [5 6]]] (2, 2, 2)

# print(a5,a5.shape)
# [[[1 2 3]
#   [4 5 6]]] (1, 2, 3)

# print(a6,a6.shape)
# [1 2 3 4 5] (5,)



'''
(numpy의 주요 자료형)
정수 (int), 실수(float), 복소수(complex), 논리(bool)
numpy는 수치해석을 위한 라이브러리인 만큼, 숫자형 자료형에 대해서는 
파이썬 내장 숫자자료형에 비해 더욱 더 자세히 나누어놓은 자료형이 존재함
https://kongdols-room.tistory.com/53 <- 여기 보고 공부해보자
https://numpy.org/doc/stable/user/basics.types.html <- numpy공식 홈피
'''