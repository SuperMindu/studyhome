from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

from tensorflow.python.keras.layers import Reshape


#1. 데이터

import numpy as np
x1_datasets = np.array([range(100), range(301,401)]) # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]) # 원유, 돈육, 밀
x3_datasets = np.array([range(100,200), range(1301,1401)]) # 뭐시기, 저시기
x1 = np.transpose(x1_datasets) 
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)
print(x1.shape, x2.shape) # (100, 2) (100, 3)

y1 = np.array(range(2001,2101)) # 금리 (100, )
y2 = np.array(range(201,301)) # 금리 (100, )

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1,x2,x3,y1,y2,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

print(x1_train.shape, x1_test.shape) # (80, 2) (20, 2)
print(x2_train.shape, x2_test.shape) # (80, 3) (20, 3)
print(y1_train.shape, y1_test.shape) # (80,) (20,)
print(y2_train.shape, y2_test.shape) # (80,) (20,)


#2. 모델
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='ak1')(input1)
dense2 = Dense(200, activation='relu', name='ak2')(dense1)
dense3 = Dense(300, activation='relu', name='ak3')(dense2)
output1 = Dense(100, activation='relu', name='out_ak1')(dense3)

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(1100, activation='relu', name='ak11')(input2)
dense12 = Dense(120, activation='relu', name='ak12')(dense11)
dense13 = Dense(130, activation='relu', name='ak13')(dense12)
dense14 = Dense(140, activation='relu', name='ak14')(dense13)
output2 = Dense(100, activation='relu', name='out_ak2')(dense14)

#2-3. 모델3
input3 = Input(shape=(2,))
dense21 = Dense(1100, activation='relu', name='ak21')(input3)
dense22 = Dense(120, activation='relu', name='ak22')(dense21)
dense23 = Dense(130, activation='relu', name='ak23')(dense22)
dense24 = Dense(140, activation='relu', name='ak24')(dense23)
output3 = Dense(100, activation='relu', name='out_ak3')(dense24)

from tensorflow.python.keras.layers import concatenate, Concatenate # 앙상블모델
merge1 = concatenate([output1, output2, output3], name='mg1')
merge2 = Dense(20, activation='relu', name='mg2')(merge1)
merge3 = Dense(300, name='mg3')(merge2)
reshp1 = Reshape((300,))(merge3)
last_output1 = Dense(1, name='last1')(reshp1)

merge11 = concatenate([output1, output2, output3], name='mg11')
merge12 = Dense(20, activation='relu', name='mg12')(merge11)
merge13 = Dense(300, name='mg13')(merge12)
reshp11 = Reshape((300,))(merge13)
last_output2 = Dense(1, name='last2')(reshp11)

print(merge1.shape)

model = Model(inputs=[input1, input2, input3], outputs=[last_output1, last_output2])


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='auto', verbose=1, 
                              restore_best_weights=True)        
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
y_predict = model.predict([x1_test, x2_test, x3_test])
y_predict = np.array(y_predict) #(2, 20, 1)
print(y_predict) 
print(np.array([y1_test, y2_test]))
y_test=np.array([y1_test, y2_test])
print(y_test.shape) 
y_predict = y_predict.reshape(2, 20)
y_test = y_test.reshape(2, 20)
print(y_test.shape) 

r2 = r2_score(y_test, y_predict)
print('loss: ', loss)
print('r2스코어 : ', r2)
# print('결과값 : ', y_predict)


# loss:  [0.056787002831697464, 0.02830709144473076, 0.028479909524321556, 0.05689086765050888, 0.040387727320194244]
# r2스코어 :  0.9999999649462936
# time : 54.75899147987366