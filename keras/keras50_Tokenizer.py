from keras.preprocessing.text import Tokenizer
import numpy as np

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text]) # 이것 또한 리스트 형태. 토큰에 대한 index가 생성됨

print(token.word_index) # <- 어절 순으로 되어 있는 걸 확인
# {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8} <- 어절에 대한 토큰 순서. 반복 순서가 많은 애들이 앞으로 잡힘

x = token.texts_to_sequences([text]) # <- 
print(x)
# [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] 

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

# x = to_categorical(x)
# print(x)
# print(x.shape) # (1, 11, 9) <- RNN, LSTM 

x = np.array(x)
print(x)

# 원핫엔코더로 바꿔보기
ohe = OneHotEncoder(sparse=False)
x = ohe.fit_transform(x.reshape(-1,8))
print(x)
print(x.shape)



