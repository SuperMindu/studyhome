import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
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

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x)
# print(y)
# print(x.shape, y.shape)   # (506, 13) (506,)  
# print(datasets.feature_names)   #sklearn 에서만 가능
# print(datasets.DESCR)


#2.  모델 구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

# model.save('./_save/keras23_1_save_model.h5')                        # (1) <- model 다음에 여기 이건 그냥 모델 세이브 

model.save_weights('./_save/keras23_5_save_weights1.h5')               # 

# model = load_model('./_save/keras23_3_save_model.h5')                # (3) <- fit 다음에 세이브 해준 모델을 3번 위에서 다시 불러오면 weight값이 새로 구해지고 기존 weight에 덮어씌워짐


# 3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True, verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es], verbose=1) 

# model.save('./_save/keras23_3_save_model.h5')                        # (2) <- fit 다음에 세이브를 해주면 model과 weight 까지 저장이 돼있음

model.save_weights('./_save/keras23_5_save_weights2.h5')               #

# model = load_model('./_save/keras23_3_save_model.h5')                # (4) <- 이게 최종형태 


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  13.494619369506836
# r2 스코어 :  0.8385481643838405



