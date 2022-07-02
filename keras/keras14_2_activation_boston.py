import numpy as np
from sklearn.tree import plot_tree
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#1. 데이터 
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# print(datasets.feature_names) (sklearn 에서만 가능)  컬럼,열의 이름들
# print(datasets.DESCR)  데이터셋 및 컬럼에 대한 설명
print(x.shape, y.shape)   # (506, 13) (506,)  

#2. 모델 구성 
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=10, verbose=1, validation_split=0.15, callbacks=[es])
end_time = time.time() - start_time

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('---------------------------------------------')
print(hist)
print('---------------------------------------------')
print(hist.history)
print('---------------------------------------------')
print(hist.history['loss'])
print('---------------------------------------------')
print(hist.history['val_loss'])
print('걸린시간 : ', end_time)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#5. 데이터 시각화
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('보스턴')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.legend()
plt.show()
