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


# [실습]
# R2 0.62 이상


#2.  모델 구성
model = Sequential()
model.add(Dense(150, input_dim=10))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(1))



#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=10)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# loss :  2170.234130859375
# r2스코어 :  0.6713368839431841

# loss :  2151.27587890625
# r2스코어 :  0.674207950124788


