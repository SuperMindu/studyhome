# https://jamm-notnull.tistory.com/11 <--
import pandas as pd # 데이터 (주로 행렬 데이터) 를 분석하는데 쓰이는 파이썬 패키지. 특히 표 데이터를 다룰때 거의 무조건 사용한다. 
import numpy as np # 선형대수 연산 파이썬 패키지. 그냥 수학 계산을 편하고 빠르게 할 수 있는 패키지라고 생각하면 쉽다. 
# %matplotlib inline # 그래프를 주피터 노트북 화면상에 바로 띄울 수 있게 하는 코드. 근데 없어도 seaborn 그래프 잘만 되더라.
import seaborn as sns # 쉽고 직관적인 방법으로 그래프를 띄울 수 있는 패키지. 
import matplotlib.pyplot as plt # matplotlib 라는 그래프 띄우는 패키지 중 일부. 나는 여러개의 그래프를 한 결과로 띄우고 싶을때만 주로 사용한다. 
from scipy.stats import norm # 정규분포를 의미. 데이터의 분포 그래프를 띄울때, 이 norm 을 불러와주면 정규분포랑 손쉽게 비교해 볼 수 있다. 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
path = 'C:/study/_data/kaggle_titanic/'

train=pd.read_csv(path + 'train.csv', index_col='PassengerId')
test=pd.read_csv(path + 'test.csv', index_col='PassengerId')
submission=pd.read_csv(path + 'gender_submission.csv', index_col='PassengerId')
submission_set = pd.read_csv('C:/study/_data/kaggle_titanic/gender_submission.csv', index_col=0)


print(train.shape, test.shape, submission.shape) # (891, 11) (418, 10) (418, 1) 
# 우리는 train.csv 에 있는 891명의 승객 데이터를 가지고, test.csv 에 있는 418명의 생존 여부를 예측해서 제출해야 한다는 의미이다. 

# sns.countplot(train_set['Survived'])
# train_set['Survived'].value_counts()

print(train.isnull().sum())
print(test.isnull().sum())
# 각 세트 별 결측치 확인
# Survived      0
# Pclass        0
# Name          0
# Sex           0
# Age         177
# SibSp         0
# Parch         0
# Ticket        0
# Fare          0
# Cabin       687
# Embarked      2
# dtype: int64

# Pclass        0
# Name          0
# Sex           0
# Age          86
# SibSp         0
# Parch         0
# Ticket        0
# Fare          1
# Cabin       327
# Embarked      0
# dtype: int64

train=train.drop(columns='Cabin')
test=test.drop(columns='Cabin')
# 결측치 확인 후 Cabin 열 삭제 

train.loc[train['Sex']=='male', 'Sex']=0
train.loc[train['Sex']=='female','Sex']=1
test.loc[test['Sex']=='male','Sex']=0
test.loc[test['Sex']=='female','Sex']=1
# 컴터가 알아들을 수 있게 남자는 0, 여자는 1로 인코딩 

train['Pclass_3']=(train['Pclass']==3)
train['Pclass_2']=(train['Pclass']==2)
train['Pclass_1']=(train['Pclass']==1)
test['Pclass_3']=(test['Pclass']==3)
test['Pclass_2']=(test['Pclass']==2)
test['Pclass_1']=(test['Pclass']==1)
# 데이터를 보면 1등석일수록 생존 확률이 높고, 3등석에는 사망률이 높아진다는 것을 알 수 있다.
# 그럼 이 정보도 역시 인코딩 해주어야한다. 어? 근데 데이터가 1,  2, 3 숫자네? 
# 원핫인코딩

train=train.drop(columns='Pclass')
test=test.drop(columns='Pclass')

test.loc[test['Fare'].isnull(),'Fare']=0

train=train.drop(columns='Age')
test=test.drop(columns='Age')

train['FamilySize']=train['SibSp']+train['Parch']+1
test['FamilySize']=test['SibSp']+test['Parch']+1

train['Single']=train['FamilySize']==1
train['Nuclear']=(2<=train['FamilySize']) & (train['FamilySize']<=4)
train['Big']=train['FamilySize']>=5

test['Single']=test['FamilySize']==1
test['Nuclear']=(2<=test['FamilySize']) & (test['FamilySize']<=4)
test['Big']=test['FamilySize']>=5

train=train.drop(columns=['Single','Big','SibSp','Parch','FamilySize'])
test=test.drop(columns=['Single','Big','SibSp','Parch','FamilySize'])

train['EmbarkedC']=train['Embarked']=='C'
train['EmbarkedS']=train['Embarked']=='S'
train['EmbarkedQ']=train['Embarked']=='Q'
test['EmbarkedC']=test['Embarked']=='C'
test['EmbarkedS']=test['Embarked']=='S'
test['EmbarkedQ']=test['Embarked']=='Q'

train=train.drop(columns='Embarked')
test=test.drop(columns='Embarked')

train['Name']=train['Name'].str.split(', ').str[1].str.split('. ').str[0]
test['Name']=test['Name'].str.split(', ').str[1].str.split('. ').str[0]

train['Master']=(train['Name']=='Master')
test['Master']=(test['Name']=='Master')

train=train.drop(columns='Name')
test=test.drop(columns='Name')

train=train.drop(columns='Ticket')
test=test.drop(columns='Ticket')

y_train=train['Survived']
feature_names=list(test)
x_train=train[feature_names]
x_test=test[feature_names]
y_test = 

print(x_train.shape, y_train.shape, x_test.shape) # (891, 10) (891,) (418, 10)
x_train.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, shuffle=True, random_state=72)

#2. 모델 
model = Sequential()
model.add(Dense(128, activation='linear', input_dim=10))
model.add(Dense(128, activation='linear')) 
model.add(Dense(128, activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, callbacks=[es], verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
