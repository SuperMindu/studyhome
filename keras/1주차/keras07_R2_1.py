#회귀 모델
'''
R2 score, R2 제곱
선형회귀모델에 대한 적합도 측정값임
0~1점 만점. 1에 가까울수록 정확한 값
음수도 나올 수 있음. 자세한 공식은 아직 내 레벨로는 알기에 부족함
loss만 가지고 정확도를 보기에는 부족함이 있어서 R2 score로 점수를 메김
'''
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터 
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,8,3,8,12,13,8,14,15,9,6,17,23,20])

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Dense(400))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) # 평가해보는 단계. 이미 다 나와있는 w, b 에 test데이터를 넣어보고 평가해보는 것
print('loss : ', loss)

y_predict = model.predict(x) #y의 예측값은 x의 테스트 값에 wx + b

r2 = r2_score(y, y_predict) # 계측용 y_test값과, y예측값을 비교한다
print('r2스코어 : ', r2)


# 그래프
# import matplotlib.pyplot as plt
# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()

                                                    