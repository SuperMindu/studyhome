import numpy as np
from tensorflow.python.keras.models import Sequential # 시퀀셜 모델과
from tensorflow.python.keras.layers import Dense # 댄스 레이어를 쓸 수 있음

#1. 데이터 (정제해서 값 도출)
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

#[실습] 모델 구성
# layer와 parameter를 추가하여 deep learning으로 만들어짐
model = Sequential()
model.add(Dense(15, input_dim=1))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1))
# 모델층을 두껍게 해서 다중신경망을 형성하여 그 뒤 컴파일하고 예측을 해보면 단일신경망일 때에 비하여
# 훈련량 (epochs)을 훨씬 줄여도 loss값을 구할 수 있음

#3. 컴파일, 훈련
# 컴터가 알아듣게 훈련시킴. 컴파일. 
# y = wx+b 최적의 weight 값을 구하기 위한 최소의 mse값을 찾음
model.compile(loss='mse', optimizer='adam') 
# 평균 제곱 에러 m s e <- 이 값(loss)은 작을수록 좋음. 
# optimizer='adam'은 mse값(loss)을 감축시키는 역할. adam이 85퍼 이상이므로 꽤 괜
model.fit(x, y, epochs=200, batch_size=1) 
# epoch 훈련량을 의미.
# batch_size 몇개씩 데이터를 넣을지 지정해줌. 이게 작을수록 값이 정교해짐 시간 성능 속도?.. 근데 꼭 그렇지는 않은듯? 
# fit 위의 데이터를 가지고 훈련을 함 

#4. 평가, 예측
loss = model.evaluate(x, y) # evaluate 평가하다
print('loss : ', loss)

result = model.predict([6]) # 새로운 x값 [6]을 predict(예측)한 결과
print('6의 예측값 : ', result)

# loss :  0.4262453615665436
# 6의 예측값 :  [[5.922587]]

# loss :  0.45326584577560425
# 6의 예측값 :  [[6.022912]]

# loss :  0.42535296082496643
# 6의 예측값 :  [[6.018991]]





