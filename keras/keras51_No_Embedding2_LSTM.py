from keras.preprocessing.text import Tokenizer
import numpy as np

# 1. 데이터 
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '민수가 못 생기긴 했어요',
        '안결 혼해요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) # (14,) 이 label값이 y
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10,
#  '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, 
# '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, 
# '민수가': 25, '못': 26, '생기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]


from keras.preprocessing.sequence import pad_sequences 
pad_x = pad_sequences(x, padding='pre', maxlen=5, truncating='pre')  # truncating= 데이터를 자른걸 앞에서부터 자를거냐 뒤에서부터 자를거냐
print(pad_x)
print(pad_x.shape) # (14, 5)
pad_x = pad_x.reshape(14, 5, 1) # <- LSTM 쓰려면 3차원으로 reshape

word_size = len(token.word_index)
print('word_size : ', word_size, '개') # 단어사전의 개수 # 30 개

print(np.unique(pad_x, return_counts=True))


# 2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding, Flatten


model = Sequential()
# model.add(Embedding(input_dim=31, output_dim=10, input_length=5)) 
model.add(LSTM(32, input_shape=(5, 1)))
model.add(Dense(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # y값이 참, 거짓을 나타내는 0 또는 1이니까 activation='sigmoid'
# model.summary()


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=20, batch_size=16, verbose=1)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1] # <- 이렇게 해서 loss나 acc를 선택해서 호출할 수 있음
print('acc : ', acc)

# [실습] 
# 함정1. 개수가 다름 (14, 5)인데 x_predict 의 단어사전의 개수는 6개 
# 함정2. 
x_predict = ['나는 형권이가 정말 재미없다 너무 정말']

token.fit_on_texts(x_predict)
x_predict = token.texts_to_sequences(x_predict)
pad_x_pred = pad_sequences(x_predict, padding='pre', maxlen=5)
result = model.predict(pad_x_pred)

print('결과 : ', result)

# 알아서 해보자

