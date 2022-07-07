import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import time

import tensorflow as tf
tf.random.set_seed(66) # weight의 랜덤 난수 지정 (첫 weight의 랜덤 값을 고정시켜 버림. 별로 좋진 않음. 걍 쌤이 갑자기 생각나셨다고 함)

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#1. 데이터 
datasets = load_iris()
print(datasets.DESCR)
#  :Number of Instances: 150 (50 in each of three classes)  
# y의 label 값이 3개 
print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets['data']
y = datasets.target
print(x)
print(y)
print(x.shape) # (150, 4)
print(y.shape) # (150,)
print('y의 label값 : ', np.unique(y)) # [0 1 2]

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y) # y값을 to_categorical의 y에 집어 넣음
print(y) 
print(y.shape) # (150, 3) 



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
# shuffle=False 로 하면 
# print(y_train) # 순차적으로만 나온다
# print(y_test) # 2 '만' 나온다
# 그래서 True로 해줘야 함!





#2. 모델 구성 
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=4))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax')) 
# 최종 아웃풋 레이어의 개수는 y의 label의 갯수 (찾아야 하는 y의 갯수대로. 분류값에 대한 숫자만큼)
# 다중분류에서는 아웃풋 레이어에 softmax 활성함수를 넣어줘야 함 (중간에는 못 넣음)
# softmax의 결괏값은 모든 연산의 합이 1.0 으로 나온다 (결괏값 하나 하나는 0.xx 형태로)
# 그중에 가장 큰 값을 찾으면 됨 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
# 다중분류모델에서는 categorical_crossentropy 만 씀 (99% 이것만 씀)
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

# print('------------------y_test-------------------')
# print(y_test)
# print('------------------y_pred-------------------')
# y_pred = model.predict(x_test)
# print(y_pred)
print('-------------------------------------')

# y_test = np.argmax(y_test, axis=1) # 열 기준으로 찾아야 함 (axis=1) 
# y_pred = np.argmax(y_pred, axis=1) # <-- y_pred 에서 열 기준으로 최대값을 찾음
# print(y_test)
# print(y_pred)

# y_predict = np.argmax(y_pred, axis= 1)
# print(y_predict)
# y_predict = to_categorical(y_predict)
# print(y_test)
# print(y_predict)


# acc 에서는 y_test와 y_pred는 비교가 불가능
# y_pred 에서 가장 큰 값만 1로 바꾸고 나머지는 0으로 바꿔야 함 (그래야 비교가 가능) 

y_predict = model.predict(x_test)
y_test = np.argmax(y_test, axis=1) 
print(y_test)
y_predict = np.argmax(y_predict, axis=1) 
print(y_predict)

# y_predict = y_pred.round(0)
acc = accuracy_score(y_test, y_predict) # 여기서 test와 pred를 비교해서 acc 값을 뽑아야 함
print('acc 스코어 : ', acc)



# loss :  0.047380294650793076
# accuracy :  1.0

# import tensorflow as tf
# tf.random.set_seed(66) <-- 이걸 넣어서 weight 값의 랜덤 난수를 고정해주니까 하이퍼 파라미터 튜닝을 하지 않으면 몇 번을 돌려도 loss 값이 똑같이 나옴
# loss :  0.08155178278684616
# accuracy :  0.9666666388511658
# loss :  0.08155178278684616
# accuracy :  0.9666666388511658
# loss :  0.08155178278684616
# accuracy :  0.9666666388511658
# activation만 수정 해봤더니 accuracy는 똑같이 나오는데 loss가 다르게 나옴 (물론 그 뒤로는 몇번을 돌려도 똑같이)
# loss :  0.06570298224687576
# accuracy :  0.9666666388511658
# loss :  0.06570298224687576
# accuracy :  0.9666666388511658
# activation + epochs 횟수 늘리기 (epochs 횟수는 영향이 없는듯? patience가 똑같아서 그런가?)
# loss :  0.06570298224687576
# accuracy :  0.9666666388511658
# loss :  0.06570298224687576
# accuracy :  0.9666666388511658
# 
# loss :  0.07732879370450974
# accuracy :  0.9666666388511658



#5. 데이터 시각화
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('아이리스')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.legend()
# plt.show()






# pred를 해서 나온 결과가 3개가 출력 됨
# 마지막 노드가 3개이기 때문에
# 그 3개를 합친 값은 1
# y_test 
# y_test의 값은 y = to_categorical(y) 으로 원핫인코딩이 돼 있는 상태
# [[0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]
# (n, 3) 의 형태
# pred 해서 나온 결과값은 나누어 떨어지지 않았음
# [[1.12961698e-03 9.97373819e-01 1.49650523e-03]
#  [6.62750157e-04 9.93067384e-01 6.26994437e-03] 
#  [7.54746667e-04 9.91101980e-01 8.14322289e-03] 
#  [9.99910712e-01 8.92578755e-05 2.34570241e-12] 
#  [8.42118403e-04 9.97301817e-01 1.85605418e-03] 
#  [1.56306440e-03 9.96728778e-01 1.70815433e-03] 
#  [9.99890089e-01 1.09866502e-04 6.85568182e-12] 
#  [9.99990106e-01 9.94055335e-06 3.99276288e-14] 
#  [9.99722302e-01 2.77669111e-04 1.00665143e-10] 
#  [5.18107445e-05 1.14038453e-01 8.85909677e-01] 
#  [3.40044426e-05 4.85414565e-02 9.51424479e-01] 
#  [4.50965090e-05 1.24553971e-01 8.75400901e-01] 
#  [9.99782264e-01 2.17703986e-04 2.89512459e-11] 
#  [2.17422421e-04 2.52208233e-01 7.47574329e-01] 
#  [3.12738994e-05 3.29433605e-02 9.67025399e-01] 
#  [9.99014974e-01 9.85060004e-04 1.03378994e-09] 
#  [2.59692129e-03 9.94472206e-01 2.93085189e-03] 
#  [2.40508802e-04 2.39576742e-01 7.60182798e-01] 
#  [8.10310303e-06 2.31110733e-02 9.76880908e-01] 
#  [2.60188199e-06 1.33125745e-02 9.86684859e-01] 
#  [9.99917626e-01 8.23737428e-05 4.28146537e-12] 
#  [1.18341076e-03 9.83336210e-01 1.54804019e-02] 
#  [1.01658679e-03 9.55841005e-01 4.31423038e-02] 
#  [1.62243567e-04 1.97646663e-01 8.02191019e-01] 
#  [1.15605909e-03 8.78377438e-01 1.20466419e-01] 
#  [2.72255943e-06 1.46103837e-02 9.85386848e-01] 
#  [9.99938607e-01 6.13703378e-05 2.14130341e-12] 
#  [9.94837165e-01 5.16289426e-03 2.45811567e-08] 
#  [4.92254330e-05 4.25687432e-02 9.57382023e-01] 
#  [4.77243775e-05 7.33535588e-02 9.26598787e-01]]
# 그래서 이 두개를 비교하면 비교가 되지 않음
# acc 스코어는 딱 떨어지는 정수값을 비교 시켜야 함 
# np(변수값을 넣어도 상관없음).argmax 라는 것을 하게 되면 그 위치의 최고값을 숫자로 바꿔줌 (axis=1) 
# [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2] (위치 값이 나옴)
# pred 해서 나온 결과값도 argmax 해주면 됨
# 그래서 결국 argmax를 2번을 해줘야 한다는 뜻 ㅎ

