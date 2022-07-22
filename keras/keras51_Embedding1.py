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
# 제일 큰 값에 맞춰서 나머지 부족한 애들은 그 부족분에 0을 채워줌 (LSTM이라 앞에 0을 채워주는 게 더 나을듯. 근데 Bidirectional도 있어서...)
# 근데 큰 값이 너무 크면 좀 잘라서 버려줌 

from keras.preprocessing.sequence import pad_sequences # <- 시퀀스에 패딩을 씌우는 개념
pad_x = pad_sequences(x, padding='pre', maxlen=5) # <- 'pre' 앞에서부터 채움 # maxlen= 잘라주는 최대 글자 개수
print(pad_x) # <- 0이 채워진 것을 확인할 수 있음
print(pad_x.shape) # 이 pad를 씌워준 값이 x. (14, 5) (Dense) 이걸 reshape 해서 3차원으로 만들어서 LSTM을 쓸 수 있음

word_size = len(token.word_index)
print('word_size : ', word_size, '개') # 단어사전의 개수 # 30 개

print(np.unique(pad_x, return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),  <---------- 0 까지 포함해서 유니크값이 31개 인걸 확인할 수 있음
#  array([37,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
#       dtype=int64))

# 2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding 
# Embedding은 보통 인풋 레이어에서 많이 씀
# 단어 간의 상관관계가 가까운 쪽으로 수치를 줌. 
# 많은 데이터로 단어 간의 유사도를 연산시켜서 학습시킴
# 어차피 컴터가 다 해줌 (원핫 안해도 됨)

model = Sequential()#  인풋은 (14,5) 아직 원핫을 안해줬음
#                         ┌>                  ┌> 이건 내가 정함
#                      ┌>단어사전의개수 ┌>아웃풋개수   ┌>이거는 명시 안해줘도 연산 가능 (이게 결국 timesteps임. 밑에 sumarry에서(None, 5, 10) )                                                          
model.add(Embedding(input_dim=31, output_dim=10, input_length=5)) # Embedding 레이어를 통과해서 나온 값들은 3차원의 형식으로 나옴. 따라서 LSTM과 엮어주면 됨
# model.add(Embedding(input_dim=31, output_dim=10))
# model.add(Embedding(31, 10))# <- 걍 이렇게 해줘도 알아서 돌아감
# model.add(Embedding(31, 10, 5)) # ValueError: Could not interpret initializer identifier: 5 # 이렇게 쓰려면 input_length는 명시를 해줘야 함
# model.add(Embedding(31, 10, input_length=5)) # 이렇게
# input_length짜리 덩어리가 output_dim 묶음으로 None 만큼 있다

model.add(LSTM(32))# 
model.add(Dense(1, activation='sigmoid')) # y값이 참, 거짓을 나타내는 0 또는 1이니까 activation='sigmoid'
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 5, 10)             310 <----------- 전체 단어사전의 개수 * 출력값 (input_dim * output_dim)
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                5504
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 5,847
# Trainable params: 5,847
# Non-trainable params: 0
# _________________________________________________________________


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=20, batch_size=16, verbose=1)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1] # <- 이렇게 해서 loss나 acc를 선택해서 호출할 수 있음
print('acc : ', acc)

