import numpy as np 
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split
# 출처: https://rfriend.tistory.com/519 [R, Python 분석과 프로그래밍의 친구 (by R Friend):티스토리]

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# x_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])
# y_train = np.array([1,2,3,4,5,6,7])
# y_test = np.array([8,9,10])

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, train_size=0.7, 
    shuffle=True, 
    random_state=3000
    )
print('x_train : ', x_train) 
print('x_test : ', x_test) 
print('y_train : ', y_train) 
print('y_test : ', y_test) 

# x_train :  [1 3 6 7 2 9 4]
# x_test :  [ 5 10  8]
# y_train :  [1 3 6 7 2 9 4]
# y_test :  [ 5 10  8]


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
