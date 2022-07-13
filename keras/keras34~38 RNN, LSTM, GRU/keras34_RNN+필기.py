# 시계열
# 보통 시계열 데이터는 y값이 없음
# 그래서 데이터를 받자마자 x와 y의 값을 나눠줘야 함 (데이터를 계속 중첩시켜 가면서 나눠줌)
# 데이터를 나눠줬을 때, 그 나눠준 순서가 다음 나눠주는 순서에 영향을 끼치는가? 
# 그 순서가 영향을 끼치게끔 규칙성을 만들어 줌
# https://gruuuuu.github.io/machine-learning/lstm-doc/ <-- 


import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping  



#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# y = ?

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4,5,6,7,8,9,10])
# (N, 3, 1) -> (행, 열, 데이터를 자르는 단위) 그래서 RNN은 shape가 3차원임
# x_shape = (행, 열, 몇개씩 자르는지!)
print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(7, 3, 1) # (7, 3, 1)
print(x.shape)


#2. 모델구성
model = Sequential()
model.add(SimpleRNN(64, input_shape=(3, 1))) # SimpleRNN 에서 Dense로 넘어갈 때 자동적으로 2차원으로 던져줌
#                    └> units           └> input_dim 
# model.add(SimpleRNN(10))
model.add(Dense(64, activation='relu')) # RNN의 아웃풋은 2차원 형태 (그래서 SimpleRNN에서는 RNN끼리 두개를 엮을 수 없음)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=430, mode='min', verbose=1, restore_best_weights=True)
model.fit(x, y, epochs=3000, batch_size=1, callbacks=[es], verbose=1)

#4. 평가, 예측
# loss = model.evaluate(x, y)
# result = model.predict([8,9,10])
# print('loss : ', loss)
# print('[8,9,10]의 결과 : ', result)
# 이렇게 돌리면 model.predict([8,9,10])이 1차원이라 안됨
# RNN은 3차원이기 때문에 3차원과 비교를 해줘야 함

loss = model.evaluate(x, y)
y_pred = np.array([8,9,10]).reshape(1, 3, 1) # [[[8], [9], [10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)

# loss :  0.00013958293129689991
# [8,9,10]의 결과 :  [[10.616289]]

# loss :  0.0002554102975409478
# [8,9,10]의 결과 :  [[10.768097]]

# loss :  7.663348515052348e-05
# [8,9,10]의 결과 :  [[10.784917]]

# loss :  0.0001033739754348062
# [8,9,10]의 결과 :  [[10.835012]]

# loss :  9.476319974055514e-05
# [8,9,10]의 결과 :  [[10.891523]]

























# def split_xy1(dataset, time_steps):
#   x, y = list(), list()
#   for i in range(len(dataset)):
#     end_number = i + time_steps
#     if end_number > len(dataset) - 1:
#       break
#     tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
#     x.append(tmp_x)
#     y.append(tmp_y)
#   return np.array(x), np.array(y)

# x, y = split_xy1(datasets, 4)
# print(x, "\n", y)
# 출처: https://soccerda.tistory.com/199 [soccerda의 IT이야기:티스토리]
