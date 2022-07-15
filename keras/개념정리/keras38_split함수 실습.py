import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping  
# 함수 먼저 빼놓기
def split_x(dataset, size):                       
    aaa = []                                  
    for i in range(len(dataset) - size + 1):       
        subset = dataset[i : (i + size)]          
        aaa.append(subset)                     
    return np.array(aaa)                      

# 1-1) 데이터 로드 
a = np.array(range(1, 101))               
x_pred = np.array(range(96, 106))    # 주의! x_pred도 형태를 x와 같은 형태로 맞춰줘야 함
print(x_pred)  # [ 96  97  98  99 100 101 102 103 104 105]

# print(a)        

# size = 5   # size라는 변수 선언하고 크기를 5로 정함 (x는 4개, y는 1개) 
# 굳이 이렇게 size 변수 선언을 안하고 그냥 필요할 때 그 자리에 원하는 숫자로 넣어서 하는 게 편할듯?                                    


# 1-2) 전처리     ┌> 원래라면 size 가 들어갔어야 할 자리임. 그냥 5로 넣어줌
bbb = split_x(a, 5) # RNN에 쓸 수 있게 split해서 연속된 데이터 형태로 바꿔줌
print(bbb)
print(bbb.shape) # (96, 5)

# ↓ 바꾼 데이터를 다시 쪼개서 x와 y로 만들어줌
x = bbb[:, :-1] # [:, :-1] <- 모든 행, 가장 마지막 열 제외                                  
y = bbb[:, -1]  # [:, -1] <- 모든 행, 가장 마지막 열만
print(x, y) # 잘 정제된 거 확인용 프린트
print(x.shape, y.shape) # (96, 4) (96,)


# print(len(x)) 일일이 몇행인지 셀 수 없으니까 x의 len을 프린트해서 x의 행의 개수 체크. 지금 이 데이터에서는 96개 ㅇㅇ 
# 그래서 x의 길이를 확인하고 LSTM에는 3차원으로 넣어줘야 하니까 3차원으로 reshape 시켜줌
# x = x.reshape(len(x),4,1) <- 이렇게 reshape 할 수도 있음. 
x = x.reshape(96, 4, 1) # (96개의 데이터를 4행 1열로 바꿔주겠다)
'''
(96, 4, 1) 은 이런형태
[[[ 1]
  [ 2]
  [ 3]
  [ 4]]

 [[ 2]
  [ 3]
  [ 4]
  [ 5]]

 [[ 3]
  [ 4]
  [ 5]
  [ 6]]     

이렇게도 reshape 시킬 수 있으니까 어떻게 나오는지 일단 확인차
(96, 2, 2) 2행 2열로 바꿔줬을 때
[[[ 1  2]
  [ 3  4]]

 [[ 2  3]
  [ 4  5]]

 [[ 3  4]
  [ 5  6]]
  
(96, 1, 4) 1행 4열로 바꿔줬을 때
[[[ 1  2  3  4]]

 [[ 2  3  4  5]]
'''


ccc = split_x(x_pred, 4) # ccc라는 변수를 선언하고 그 안에 x_pred를 split해서 4개씩 연속된 값으로 담아줌
print(ccc)
print(ccc.shape) # (7, 4)




# 모델 구성 및 평가 예측
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(4, 1))) 
#               └> units                                  └> input_dim 
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

print(y_predict)
print('loss : ', loss)
print('결과 : ', result)
