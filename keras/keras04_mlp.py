# mlp = 멀티 레이어 퍼섹트론
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터 (행무시, 열우선)
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]]) # 10행 2열의 데이터를 넘겨줘야 함
y = np.array([11,12,13,14,15,16,17,18,19,20])
print(x.shape) # (2, 10)
print(y.shape) # (10,) 

x = x.T
print(x)
print(x.shape) 

### 행렬의 행과 열을 바꾸기, 행렬의 축을 바꾸는 3가지 방법 ###

#1. a.T attribute (x = x.T) 이거 하나만으로 배열의 행과 열을 바꿔줌
x = x.T
print(x)
print(x.shape)
#2. # np.transpose(a) method 이거도 똑같은 기능
# x = x.transpose
# print(x)
# print(x.shape)
#3. np.swapaxes(a, 0, 1) method
# 출처: https://rfriend.tistory.com/289 [R, Python 분석과 프로그래밍의 친구 (by R Friend):티스토리]


#4 (앞으로 제일 많이 씀) x = x.reshape(10,2)
# print(x)
# print(x.shape)

'''
transpose 와 reshape의 차이점 - transpose는 변환의 개념이고, reshape는 데이터를 늘였다 줄였다 하면서 형태를 바꿔주는 개념
'''



#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))  # 벡터의 수와 같음. 컬럼, 열, 특성, 피처 
model.add(Dense(4))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=380, batch_size=1)

#4. 평가, 예측 (열, 컬럼, 특성, 피처가 동일해야 한다)
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[10, 1.4]])
print('[10, 1.4]의 예측값 : ', result)


# loss :  0.012226113118231297
# [10, 1.4]의 예측값 :  [[19.982008]]

# loss :  2.203004498824157e-07
# [10, 1.4]의 예측값 :  [[19.999296]]




