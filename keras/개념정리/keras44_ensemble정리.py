# 앙상블 모델 
# 앙상블 모델은 트레이닝 데이터를 여러개의 데이터셋으로 나눠서 각각 따로따로 모델을 여러개를 만들어
# 그 모델들을 따로 연산한 후 합쳐서 더 좋은 결과를 기대하거나 
# 혹은 예측값이 여러개가 필요할 경우 합쳤던 모델을 다시 분류하여서 각각 더 정확한 결괏값을 기대할 수 있음
# 보통의 경우는 이러한 앙상블 모델을 쓰면 성능향상을 기대할 수 있다고 함
# 또한 각 모델별 특징이 두드러지는 즉 over fitting이 잘 되는 모델을 기본적으로 사용한다고 함 
# 흔히들 머신러닝의 랜덤포레스트같이 결정트리를 사용하는 모델들을 앙상블이라고 부름

# 하지만 아래와 같이 딥러닝 성능을 향상하기 위한 기법으로 사용할 수 있음
# 각각 다른 신경망으로 학습데이터에 대해 각각 따로 학습을 시킨 후, n개의 예측값을 출력시키고 그것을 n으로 나눈 평균값을 최종 출력으로 함 (Stacking사용)
# 오버피팅의 해결과 약간의 정확도 상승 효과가 실험을 통해 입증되었음
# 또한 평균이 아닌 투표방식, weight별 가중치를 다르게 주는 방법들도 존재함


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
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]) # ex) 원유, 돈육, 밀
x3_datasets = np.array([range(100, 200), range(1301, 1401)]) # ex) 우리반 IQ, 우리반 키

# 행렬의 행과 열을 바꿔줌. (축을 바꿔줌)
x1 = np.transpose(x1_datasets) 
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape, x2.shape, x3.shape) # (100, 2) (100, 3) (100, 2)

y1 = np.array(range(2001, 2101)) # ex) 금리
y2 = np.array(range(201, 301)) # ex) 환율
print(y1.shape) # (100,)


# 1-2) 데이터 정제 
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, x3, y1, y2, train_size=0.8, shuffle=True, random_state=66) 
print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, x3_train.shape, x3_test.shjape,  y1_train.shape, y1_test.shape, y2_train.shape, y2_test.shape)


# 2. 모델 구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout

# 함수형 모델을 사용하여 트레이닝 시킬 데이터셋의 개수만큼 모델링을 해줌
# 2-1) 모델1 (x1)
input1 = Input(shape=(2, ))
dense1 = Dense(64, activation='relu', name='ms1')(input1)
dense2 = Dense(64, activation='relu', name='ms2')(dense1)
dense3 = Dense(32, activation='relu', name='ms3')(dense2)
output1 = Dense(32, activation='relu', name='out_ms1')(dense3)

# 2-2) 모델2 (x2)
input2 = Input(shape=(3, ))
dense11 = Dense(64, activation='relu', name='ms11')(input2)
dense12 = Dense(64, activation='relu', name='ms12')(dense11)
dense13 = Dense(32, activation='relu', name='ms13')(dense12)
dense14 = Dense(32, activation='relu', name='ms14')(dense13)
output2 = Dense(32, activation='relu', name='out_ms2')(dense14)

# 2-2) 모델3 (x3)
input3 = Input(shape=(2, ))
dense21 = Dense(64, activation='relu', name='ms21')(input3)
dense22 = Dense(64, activation='relu', name='ms22')(dense21)
dense23 = Dense(32, activation='relu', name='ms23')(dense22)
dense24 = Dense(32, activation='relu', name='ms24')(dense23)
output3 = Dense(32, activation='relu', name='out_ms3')(dense24)


# 트레이닝 데이터셋의 모델링을 완료 했으면 concatenate함수 혹은 클래스로 각각의 아웃풋을 합쳐서 모델을 또 만들어 줌
# 그렇게 되면 각각의 output의 노드개수가 합쳐진 하나의 (Dense)레이어가 된거임 (지금이야 Dense레이어를 썼지만 RNN이나 CNN 같은 걸로도 만들 수 있을듯)
from tensorflow.python.keras.layers import concatenate, Concatenate # 소문자는 함수, 대문자는 클래스 # concatenate는 사슬처럼 잇다. 
# merge1 = concatenate([output1, output2, output3], name='mg1') # <- 함수로
merge1 = Concatenate()([output1, output2, output3]) #             <- 클래스로 (클래스는 name=''을 지원하지 않음 그래서 넣으면 안됨)
merge2 = Dense(64, activation='relu', name='mg2')(merge1)
merge3 = Dense(32, name='mg3')(merge2)
last_output = Dense(1, name='last1')(merge3)


# 그래서 합친 모델을 완성 했는데 만약 여러개의 예측 결과값이 필요하다면 물론 필요한 만큼 모델을 다시 나눠줄 수 있음
# 이때는 그냥 쉽게 생각해서 함수형모델 그냥 만들어 내려가듯이 만들고 아웃풋만 필요한 갯수에 맞게 뽑아주면 됨
# 2-4) 
output41 = Dense(10)(last_output)
output42 = Dense(10)(output41)
last_output2 = Dense(1)(output42)

# 2-5) 
output51 = Dense(10)(last_output2)
output52 = Dense(10)(output51)
output53 = Dense(10)(output52)
last_output3 = Dense(1)(output53)

model = Model(inputs=[input1, input2, input3], outputs=[last_output2, last_output3]) # <- 여러개의 값이 들어가니까 []로 리스트 형태로 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=30, mode='auto', verbose=1, restore_best_weights=True)        
hist = model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=300, batch_size=8, validation_split=0.2, callbacks=[es], verbose=1)

#4. 평가, 예측
# 이때도 필요한 예측 결과값의 개수 (최종 아웃풋의 개수) 만큼 evaluate 해주고 predict 해주고 평가지표 출력해주면 됨
loss1 = model.evaluate([x1_test, x2_test, x3_test], y1_test)
loss2 = model.evaluate([x1_test, x2_test, x3_test], y2_test)
print('loss1 : ' , loss1)
print('loss2 : ' , loss2)
y1_predict, y2_predict = model.predict([x1_test, x2_test, x3_test])
# print(y1_predict.shape)
r2_1= r2_score(y1_test, y1_predict)
r2_2= r2_score(y2_test, y2_predict)
# print('loss : ' , loss)
print('r2 스코어 : ', r2_1) 
print('r2 스코어 : ', r2_2) 

# loss1 :  [3203028.5, 11.057512283325195, 3203017.5] 처음 = 두번째 loss + 세번째 loss 
# loss2 :  [3235077.25, 3234379.75, 697.4199829101562]
# r2 스코어 :  0.9860089486576423
# r2 스코어 :  0.11755571357680805


