# 컬럼을 일별로 잘라야 함 
# 월요일에 데이터를 받아서 화요일 시가(am9) (30%) 
# 그 데이터를 토대로 수요일 종가(pm3) (70%)





from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical 

# 1-1) 데이터 로드
import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]) # ex) 삼전 종가, 하이닉스 종가

x1 = np.transpose(x1_datasets)

print(x1.shape) # (100, 2) 
y1 = np.array(range(2001, 2101)) # ex) 금리
y2 = np.array(range(201, 301)) # ex) 환율
print(y1.shape) # (100,)

# print(np.unique(y, return_counts=True)) 

# y = to_categorical(y)  
# print(y) 
# print(y.shape)


# 1-2) 데이터 정제 
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, train_size=0.8, shuffle=True, random_state=66) 
print(x1_train.shape, x1_test.shape, y1_train.shape, y1_test.shape)
#             (80, 2)         (20, 2)        (80, 3)        (20, 3)         (80,)         (20,)

# 2. 모델 구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout

# 2-1) 모델1 (x1)
input1 = Input(shape=(2, ))
dense1 = Dense(64, activation='relu', name='ms1')(input1)
dense2 = Dense(64, activation='relu', name='ms2')(dense1)
dense3 = Dense(32, activation='relu', name='ms3')(dense2)
output1 = Dense(32, activation='relu', name='out_ms1')(dense3)

# concatenate
from tensorflow.python.keras.layers import concatenate, Concatenate # 소문자는 함수, 대문자는 클래스 # concatenate는 사슬처럼 잇다. 
# merge1 = concatenate([output1, output2, output3], name='mg1') # output1의 노드개수와 output2의 노드개수가 합쳐진 하나의 (Dense)레이어가 된거임
# merge1 = Concatenate()([output1, output2, output3])
# merge2 = Dense(64, activation='relu', name='mg2')(merge1)
# merge3 = Dense(32, name='mg3')(merge2)
# last_output = Dense(1, name='last1')(merge3)

# 2-4) 
output11 = Dense(10)(output1)
output22 = Dense(10)(output11)
last_output2 = Dense(1)(output22)

# 2-5) 
output31 = Dense(10)(last_output2)
output32 = Dense(10)(output31)
output33 = Dense(10)(output32)
last_output3 = Dense(1)(output33)

model = Model(inputs=[input1], outputs=[last_output2, last_output3])

# merge2 = Dense(64, activation='relu', name='mg2')(merge1)
# merge3 = Dense(32, name='mg3')(merge2)
# output4 = Dense(1, name='last1')(merge3)

# merge22 = Dense(64, activation='relu', name='mg22')(merge1)
# merge23 = Dense(32, name='mg23')(merge22)
# output5 = Dense(1, name='last21')(merge23)

# model = Model(inputs=[input1, input2, input3], outputs=[output4, output5]) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=30, mode='auto', verbose=1, restore_best_weights=True)        
hist = model.fit(x1_train, [y1_train, y2_train], epochs=300, batch_size=8, validation_split=0.2, callbacks=[es], verbose=1)

#4. 평가, 예측
loss1 = model.evaluate(x1_test, y1_test)
loss2 = model.evaluate(x1_test, y2_test)
print('loss1 : ' , loss1)
print('loss2 : ' , loss2)
y1_predict, y2_predict = model.predict(x1_test)
# print(y1_predict.shape)
r2_1= r2_score(y1_test, y1_predict)
r2_2= r2_score(y2_test, y2_predict)
# print('loss : ' , loss)
print('r2 스코어 : ', r2_1) 
print('r2 스코어 : ', r2_2) 

# loss1 :  [3203477.5, 9.8916654586792, 3203467.5]
# loss2 :  [3237236.0, 3236541.75, 694.1685180664062]
# r2 스코어 :  0.9874840928929296
# r2 스코어 :  0.12166977108300392


