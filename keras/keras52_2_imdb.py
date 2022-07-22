from keras.datasets import imdb
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000 # test_split 안주면 디폴트 값으로 잡힘. 디폴트가 
)

print(x_train) # 엄청 많음
print(x_train.shape, x_test.shape) # (25000,) (25000,)
print(y_train) # 0101010101 이진분류! 
print(np.unique(y_train, return_counts=True)) # 2개! 이진분류!
print(len(np.unique(y_train))) # 2개

print(type(x_train), type(y_train)) 
print(type(x_train[0])) 
# print(x_train[0].shape) AttributeError: 'list' object has no attribute 'shape'  리스트는 shape 지원 안함!
print(len(x_train[0])) # 218
print(len(x_train[1])) # 189

# print(len(max(x_train))) 83이 나오는데 왜 에러가 안나고 출력이 되지? 이건 뭐지?

print('리뷰감상의 최대길이 : ', max(len(i) for i in x_train)) # <- 이렇게 하면 반환값이 앞에 i에 들어감 그거의 길이잖아 0~8982개의 길이값이 다 저장이 되잖아 그래서 이거의 max를 해주면 되잖ㅇ아
# 뉴스기사의 최대길이 :  2494
print('리뷰감상의 평균길이 : ', round(sum(map(len, x_train)) / len(x_train))) # map 찾아보기!!!
# 뉴스기사의 평균길이 :  239

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre') # 앞에서부터 패딩하고 최대값은 100으로 함.. 근데 100보다 더 큰 값들은 앞에서부터 잘라줌. 왜 why? 앞에서 0을 채워줬으니까?
                        # 와꾸가 (8982,) -> (8982, 100) 이렇게 바뀌겠지 
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')


from sklearn.model_selection import train_test_split





# 분류라서 원핫 해야됨! 근데 sparse_categorical 써도 되는데 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, y_train.shape) # (25000, 100) (25000, 2)
print(x_test.shape, y_test.shape) # (25000, 100) (25000, 2)





# 2. 모델 구성 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding 
model = Sequential()
model.add(Embedding(input_dim=25000, output_dim=100, input_length=50)) 
model.add(LSTM(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1)

# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1] # <- 이렇게 해서 loss나 acc를 선택해서 호출할 수 있음
print('acc : ', acc)

# 일단 돌아는 감 집가서 돌려보기
