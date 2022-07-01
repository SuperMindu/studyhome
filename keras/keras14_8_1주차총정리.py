'''
1주차 (6/15 ~ 7/1)
진도, 실습, 과제 등에서 나왔던 모든 내용들을 전부 싸그리 통째로 넣고 하나하나 각주 달아서 설명
'''



#***************************************본격적으로 모델을 코딩하기 전 필요한 라이브러리나 데이터셋 등을 불러오는 단계***********************************************
import numpy as np 
# numpy는 행렬이나 대규모 다차원 배열을 쉽게 처리할 수 있도록 지원하는 파이썬의 라이브러리임
# 벡터나 행렬연산에 있어서 어마어마한 편의성을 제공함
# 밑에 나올 pandas나 matplotlib 등의 기반이 되는 라이브러리임
# 기본적으로 array라는 단위로 데이터를 관리함
# 고등수학의 행렬과 유사한 부분이 많음

from tensorflow.python.keras.models import Sequential 
# tensorflow.python.keras.models 에서 Sequential 이라는 모델을 불러온다는 뜻
# Sequential은 순차적이라는 뜻 즉 Sequential 모델(순차 모델)을 쓸 수 있음
# Sequential 모델은 각 레이어에 정확히 하나의 입력 텐서와 하나의 출력 텐서가 있는 일반 레이어 스택에 적합함 (대충 우리가 지금 하고 있는 모델 구성 방식임)
# Sequential 모델은 모델에 다중 입력 또는 다중 출력이 있거나, 레이어에 다중 입력 또는 다중 출력이 있거나, 레이어 공유를 해야하거나 하는 경우는 적합하지 않음

from tensorflow.python.keras.layers import Dense 
# 신경망을 이해할 때 사용하는 모듈임
# 다양한 함수와 hidden layer(은닉층)을 거쳐서 나온 숫자들을 한 곳으로 모아주고, 태스크의 적절한 함수에 정보를 전달하기 위한 레이어라고 보면 편함
# Dense Layer는 여러 Layer로부터 계산된 정보들을 한 곳으로 모은 자료임
# 더 쉽게 말하면 input을 넣었을 때 output으로 바꿔주는 중간 다리임
# 한마디로 그냥 신경망 만드는 거임 

from sklearn.model_selection import train_test_split
# 사이킷런에서 제공하는 model_selection 모듈은 train data 와 test data 세트를 분리하거나 교차 검증, 분할, 평가, 하이퍼 파라미터 튜닝을 위한 다양한 함수나 클래스를 제공함
# 복잡한 딥러닝 모델을 쉽게 작성할 수 있도록 해주는 라이브러리임 
# import train_test_split 는 말 그대로 학습과 테스트 데이터셋을 분리 해주는 기능을 불러온다는 뜻
# tmi (데이터를 train 과 test로 나눠주는 이유)
# fit에서 모델 학습을 훈련시킬 때 모든 데이터를 다 학습 시켜버리면 예측 단계 -> model.predict[] 에 실제로 원하는 미래의 데이터를 넣어봤을 떄 크게 오류가 날 수 있음 (이게 바로 과적합)
# 왜냐하면 컴퓨터는 주어진 모든 값으로 훈련'만' 하고 실전을 해본 적이 없기 때문임
# 그래서 train과 test로 나눠서 train으로 학습을 시키고 test로 실전같은 모의고사를 한번 미리 해보면 fit단계에서의 loss값과 evaluate의 loss값의 차이가 큰걸 확인할 수 있음
# 근데 아직 확인까지만 가능하고 그 이상은 뭐 할 수 없음
# 여기서 나온 loss값과 fit 단계의 loss값들의 차이가 크다 하더라도 그 차이가 fit 단계에 적용되지는 않음

import matplotlib.pyplot as plt


