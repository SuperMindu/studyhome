# 여태까지 회귀모델에서 했던 방식대로만 하면 loss는 잘 나오는 것 같지만 r2값이 잘 나오지 않는다 
# activation 검색 해보자
# activation(활성화함수. 한정시킨다는 의미. activation='linear'가 디폴트)은 모든 레이어에 강림 
# 히든레이어 안에 노드와 노드를 통과할 때 데이터를 한정시켜줌

import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping # EarlyStopping 이건 클래스
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)



#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data # ['data'] <- 이렇게도 쓸 수 있음
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

# print(x)
# print(y) 

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.80, shuffle=True, random_state=72)


#2.  모델 구성
model = Sequential()
model.add(Dense(128, activation='linear', input_dim=30))
model.add(Dense(128, activation='linear')) 
model.add(Dense(128, activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(1, activation='sigmoid'))
# linear는 그냥 선. 직선으로 그냥 이어줌
# sigmoid는 1 ~ 0 사이로 한정시킨다
# 마지막 아웃풋에서는 0 과 1로 나와야 하기때문에 마지막 아웃풋 레이어에 써줘야 한다 (이진분류이기 때문) (그 사이 히든레이어에서는 안 씀)
# 하지만 sigmoid는 1 ~ 0 사이로 한정시키기 때문에 0과 1로 분류해주기 위해 반올림을 해줘야한다 (0.5가 기준. 이상이면 1, 미만이면 0)
# 분류모델에서 지금과 같은 이진분류에서는 마지막 아웃풋 레이어에 sigmoid
# activation='relu' 가 킹왕짱 (중간 히든레이어 에서만 쓸 수 있음) (음수는 0으로 만들어버리고 나머지는 linear와 동일)
# 다중분류 에서는 ... 

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mse']) # metrics=['accuracy']는 평가지표. metrics=['accuracy']를 넣어주면 훈련할 때 verbose에서 같이 보여줌 ex) loss :  [0.22789520025253296, 0.9385964870452881], list형식이라서 mse 같은 것들도 넣을 수 있음. 하지만 이진분류에서는 mse는 신뢰하지 않음
# 이진분류모델 에서는 loss를 mse를 쓰지 않고, 당분간 내 레벨에서는 'binary_crossentropy' 이거 하나만 씀
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, 
                 callbacks=[es], # 여기도 리스트형식임. 더 넣을 수 있음. (떡밥?)
                 verbose=1) #verbose = 매개변수 (중간 훈련과정 스킵, 시간 단축 가능)
end_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('==============================')
print(hist) # <tensorflow.python.keras.callbacks.History object at 0x000002B1B6638040>
print('==============================')
print(hist.history) 
print("걸린시간 : ", end_time)
print('loss : ', loss)

y_predict = model.predict(x_test) #y의 예측값은 x의 테스트 값에 wx + b
# # r2 = r2_score(y_test, y_predict) # 계측용 y_test값과, y예측값을 비교한다
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)
print(y_predict)

'''
plt.figure(figsize=(9,6)) # figure 이거 알아서 찾아보기 (9 x 6 으로 그리겠다)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # marker '_' 로 하면 선, 아예 삭제해도 선으로 이어짐
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 보기 편하게 하기 위해서 모눈종이 형태
plt.title('유방암')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')  # 위에 label 값이 여기에 명시가 된다 (디폴트 값이 upper right)
plt.legend()
# plt.rcParams['font.family'] = 'Malgun Gothic'
plt.show()

# loss :  0.10185470432043076
# r2스코어 :  0.5664250982574991
'''



########################################[과제]##########################################
# activation : sigmoid, relu, linear 넣어보기 
# metrics 추가
# EarlyStopping 넣기
# 성능비교
# 느낀점 2줄 이상 


