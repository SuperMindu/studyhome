from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] 짜르기 train_test_split 으로만 짜르기 (섞여있는 10개의 트레인, 3개 테스트, 3개 val)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.625, shuffle=True, random_state=3000)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=3000)
# print(x_train) [ 8  7  3  5 11  6  2  9  4 13]
# print(x_test) [12 14 16]
# print(y_train) [ 8  7  3  5 11  6  2  9  4 13]
# print(y_test) [12 14 16]
# print(x_val) [10  1 15]
# print(x_val) [10  1 15]
# 일단 완성. 근데 숫자의 범위가 쥰내 커지면? 컴터가 알아서 나눠줌
# 이제 앞으로는 과적합을 val 로 확인함

# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))


# x_test = np.array([11,12,13]) 
# y_test = np.array([11,12,13]) # evaluate, predict 에서 씀

# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.25) # val_loss는 로스 값보다 조금 덜 떨어짐
'''
validation_split=0.25 => 트레인 데이터의 25%를 val 하겠다 
'''

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)




