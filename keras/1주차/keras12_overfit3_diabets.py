from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import time
# 이 밑에는 matplotlib 한글폰트 깨짐현상 해결 방법
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(x.shape, y.shape)  # (442, 10) (442,)
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.80, shuffle=True, random_state=72)


#2.  모델 구성
model = Sequential()
model.add(Dense(124, input_dim=10))
model.add(Dense(124))
model.add(Dense(124))
model.add(Dense(124))
model.add(Dense(1))


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, validation_split=0.15) #verbose = 매개변수 (중간 훈련과정 스킵, 시간 단축 가능)
end_time = time.time() - start_time


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('==============================')
print(hist) # <tensorflow.python.keras.callbacks.History object at 0x000002B1B6638040>
print('==============================')
print(hist.history) 
print("걸린시간 : ", end_time)

# {'loss': [15.577296257019043, 7.92207670211792, 7.654959678649902, 7.33324670791626, 6.890414237976074, 6.9277496337890625, 7.195715427398682, 6.570782661437988, 6.906185626983643, 6.385918140411377, 6.493862628936768], 'val_loss': [9.503546714782715, 7.9003214836120605, 13.811195373535156, 6.707312107086182, 6.845966339111328, 6.893828392028809, 5.646077632904053, 5.707865238189697, 8.865334510803223, 5.080558776855469, 5.760611534118652]}
# 딕셔너리 안에 리스트 loss : [리스트]
# loss 와 val 두 개의 딕셔너리가 있다
# 딕셔너리 (키, 벨류) 두개짜리 데이터셋
# fit 단계에서 훈련을 반환한다 어떤 형태로? 로스와 val로스로

'''
# hist.history[loss] 로스를 뽑는다
print('==============================')
print(hist.history['loss'])  # 키, 밸류 상의 로스는 문자이므로 '' 표시
print('==============================')
print(hist.history['val_loss'])
'''

import matplotlib.pyplot as plt
# import seaborn as sns

plt.figure(figsize=(9,6)) # figure 이거 알아서 찾아보기 (9 x 6 으로 그리겠다)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # marker '_' 로 하면 선, 아예 삭제해도 선으로 이어짐
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 보기 편하게 하기 위해서 모눈종이 형태
plt.title('디아벳')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')  # 위에 label 값이 여기에 명시가 된다 (디폴트 값이 upper right)
plt.legend()
# plt.rcParams['font.family'] = 'Malgun Gothic'
plt.show()


