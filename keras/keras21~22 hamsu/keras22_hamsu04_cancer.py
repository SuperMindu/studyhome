from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, 
                                                    shuffle=True, 
                                                    random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train)) # 
# print(np.max(x_train)) # 
# print(np.min(x_test)) # 
# print(np.max(x_test))

#2. 모델 구성 
# model = Sequential()
# model.add(Dense(32, input_dim=30))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
input1 = Input(shape=(30,))
dense1 = Dense(32, activation='relu')(input1)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
dense5 = Dense(32, activation='relu')(dense4)
output1 = Dense(1, activation='sigmoid')(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=10, verbose=1, validation_split=0.15, callbacks=[es])

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
print('-------------------------------------')

y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('r2스코어 : ', r2)
y_predict = y_predict.round(0) # softmax를 통과해서 실수로 나오는 y_predict 값을 반올림 해서 0과 1로 맞춰줌 (acc 비교를 위해)

acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 : ', acc)

#             노말                             MinMax                           Standard                            MaxAbs                           Robust                              
# loss : 0.1739213913679123             0.19135822355747223              0.16189509630203247                 0.1253441870212555               0.10660135746002197                                                                                                                                         
# acc :  0.9122807017543859             0.9590643274853801               0.9766081871345029                  0.9707602339181286               0.9707602339181286                                                                                                                                       
#                                                                                                                                             0.5242500305175781        
#                                                                                                                                             0.9649122807017544                                                                                                                                                             