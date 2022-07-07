import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
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
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, # 
                      filepath='./_ModelCheckPoint/keras24_ModelCheckPoint3.hdf5' # 가중치 저장할 파일 경로 지정
                      )
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es, mcp], verbose=1) 

model.save('./_save/kares24_3_save_model.h5')


#4. 평가 예측
print('--------------------- 1. 기본 출력 -------------------------')
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  11.067974090576172
# r2 스코어 :  0.8675809395878757
# 여기까지는 그냥 기본 로직
print('--------------------- 2. load_model 출력 -------------------------')
model2 = load_model('./_save/kares24_3_save_model.h5') 
loss2 = model2.evaluate(x_test, y_test)
print('loss2 : ', loss2)
y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 스코어 : ', r2)
# 저장한 걸 다시 불러와서 평가, 예측

print('--------------------- 3. ModelCheckpoint 출력 -------------------------')
model3 = load_model('./_ModelCheckPoint/keras24_ModelCheckPoint3.hdf5') 
loss3 = model3.evaluate(x_test, y_test)
print('loss3 : ', loss3)
y_predict3 = model3.predict(x_test)
r2 = r2_score(y_test, y_predict3)
print('r2 스코어 : ', r2)



# --------------------- 1. 기본 출력 -------------------------
# 4/4 [==============================] - 0s 3ms/step - loss: 9.7904
# loss :  9.790388107299805
# r2 스코어 :  0.8828661815719303
# --------------------- 2. load_model 출력 -------------------------
# 4/4 [==============================] - 0s 1ms/step - loss: 9.7904
# loss2 :  9.790388107299805
# r2 스코어 :  0.8828661815719303
# --------------------- 3. ModelCheckpoint 출력 -------------------------
# 4/4 [==============================] - 0s 3ms/step - loss: 9.7904
# loss3 :  9.790388107299805
# r2 스코어 :  0.8828661815719303


