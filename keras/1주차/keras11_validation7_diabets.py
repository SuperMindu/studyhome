# [과제] 만들어서 깃허브 올리기 (val 썼을 때와 아닐 때 성능비교도 해보기)

from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)  # (442, 10) (442,)
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.80, shuffle=True, random_state=72)


#2.  모델 구성
model = Sequential()
model.add(Dense(150, input_dim=10))
model.add(Dense(150))
model.add(Dense(150))
model.add(Dense(150))
model.add(Dense(150))
model.add(Dense(150))
model.add(Dense(1))



#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.25)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# val 없음
# loss :  2301.285400390625
# r2스코어 :  0.6514903644464696

# val 있음
# loss :  2390.2373046875
# r2스코어 :  0.6380193682271182