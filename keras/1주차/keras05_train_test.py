'''
(데이터를 train 과 test로 나눠주는 이유)
fit에서 모델학습을 시킬 때 모든 데이터를 다 학습 시켜버리면 x = [1~10], y = [1~10]
실제로 원하는 미래의 데이터를 넣어봤을 떄 크게 오류가 날 수 있음 -> model.predict[11]
왜냐하면 컴퓨터는 주어진 모든 값으로 훈련만 하고 실전을 해본 적이 없기 때문임
그래서 train과 test로 나눠서 x_train [1~7] x_test [8~10]
train으로 학습을 시키고 test로 실전같은 모의고사를 한번 미리 해보면
fit단계에서의 loss값과 evaluate의 loss값의 차이가 큰걸 확인할 수 있음
확인까지만 가능하고 그 이상은 뭐 할 수 없다? evaluate는 평가만 가능한거지
여기서 나온 loss값과 fit 단계의 loss값들의 차이가 크다 하더라도 그 차이가 fit 단계에 적용되지는 않는다
'''
import numpy as np 
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
x_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_train = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result)