import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping  

a = np.array(range(1, 101))               
x_pred = np.array(range(96, 106))    

print(x_pred)  # [ 96  97  98  99 100 101 102 103 104 105]

# print(a)        

size = 5   # x는 4개, y는 1개                                    

def split_x(dataset, size):                       
    aaa = []                                  
    for i in range(len(dataset) - size + 1):       
        subset = dataset[i : (i + size)]          
        aaa.append(subset)                     
    return np.array(aaa)                      

bbb = split_x(a, size)                         
print(bbb)
print(bbb.shape) # (96, 5)

x = bbb[:, :-1]                                    
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape) # (96, 4) (96,)

x = x.reshape(96, 4, 1)


ccc = split_x(x_pred, 4) 
print(ccc.shape) # (7, 4)




# 모델 구성 및 평가 예측
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(4, 1))) 
#               └> units           └> input_dim 
model.add(LSTM(64, input_shape=(4, 1))) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
model.fit(x, y, epochs=1000, batch_size=32, callbacks=[es], verbose=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_predict = ccc.reshape(7, 4, 1) 
result = model.predict(y_predict)

print('loss : ', loss)
print('결과 : ', result)


