from keras.datasets import reuters
import numpy as np
import pandas as pd

# https://wikidocs.net/22933 <- 여기 보고 공부 해보자 (46개의 다중분류임)

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)#  num_words= 이게 뭔 수치지? input_dim 에 넣어주면 됨
# num_words에 1,000을 넣었다면 빈도수 순위가 1,000 이하의 단어만 가져온다는 의미이므로 데이터에서 1,000을 넘는 정수는 나오지 않습니다.
# 수치로 되어있는 데이터를 원래 문자로 바꿔줄 수 있는 소스가 있음. 알아서 찾아보자?

# print(x_train) # 엄청 많음
# print(x_train.shape, x_test.shape) # (8982,) list의 개수가 8982개 
# print(y_train) <- 8982는 행이니까 어차피 똑같음 
# print(np.unique(y_train, return_counts=True)) # 유니크 값이 46개 y값이 46개 
# print(len(np.unique(y_train))) # 유니크 값의 개수만 보고 싶으면 len 쓰면 됨 

# print(type(x_train), type(y_train)) <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# print(type(x_train[0])) <class 'list'> (list의 각각의 크기가 달라서 pad 씌워줘야 함)
# print(x_train[0].shape) AttributeError: 'list' object has no attribute 'shape'  리스트는 shape 지원 안함!
# print(len(x_train[0])) 87
# print(len(x_train[1])) 56 <- list의 각각의 크기가 다른 걸 알 수가 있음 그래서 pad를 씌워줘야 하는데 최대값으로 맞추

# print(len(max(x_train))) 83이 나오는데 왜 에러가 안나고 출력이 되지? 이건 뭐지?

print('뉴스기사의 최대길이 : ', max(len(i) for i in x_train)) # <- 이렇게 하면 반환값이 앞에 i에 들어감 그거의 길이잖아 0~8982개의 길이값이 다 저장이 되잖아 그래서 이거의 max를 해주면 되잖ㅇ아
# 뉴스기사의 최대길이 :  2376
# print('뉴스기사의 평균길이 : ', sum(map(len, x_train)) / len(x_train)) # 이건 내가 해석해야됨 이렇게만 하면 소수점 까지 나오니까
print('뉴스기사의 평균길이 : ', round(sum(map(len, x_train)) / len(x_train))) # map 찾아보기!!!
# 뉴스기사의 평균길이 :  146 


# 데이터 전처리
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre') # 앞에서부터 패딩하고 최대값은 100으로 함.. 근데 100보다 더 큰 값들은 앞에서부터 잘라줌. 왜 why? 앞에서 0을 채워줬으니까?
                        # 와꾸가 (8982,) -> (8982, 100) 이렇게 바뀌겠지 
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

# 분류라서 원핫 해야됨! 근데 sparse_categorical 써도 되는데 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, y_train.shape) # (8982, 100) (8982, 46)
print(x_test.shape, y_test.shape) # (2246, 100) (2246, 46)


# 2. 모델 구성 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding 
model = Sequential()
model.add(Embedding(input_dim=8982, output_dim=100, input_length=50)) 
model.add(LSTM(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(46, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=1)

# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1] # <- 이렇게 해서 loss나 acc를 선택해서 호출할 수 있음
print('acc : ', acc)

# 일단 돌아는 감 집가서 돌려보기

