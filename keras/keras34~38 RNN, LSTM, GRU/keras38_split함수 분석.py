# 시계열 데이터를 짜르는 함수
# https://thestoryofcosmetics.tistory.com/35 <- for문 여기 보고 공부해보자
# for문은 정해진 횟수만큼 반복하는 구조이고
# while문은 어떤 조건이 만족되는 동안, 계속 반복하는 구조임
# for 변수 in range(종료값): # range()이건 함수임. range()함수는 입력 받은 숫자에 해당되는 범위의 값을 반복 가능한 객체로 만들어 출력해줌. 쉽게 말해서 range()함수를 이용하면 특정 구간의 정수들을 생성할 수 있음





#  RNN 쓰기 위해 데이터 timestep대로 데이터를 잘라서 만들어주는 함수. + 해석
import numpy as np

a = np.array(range(1, 11))                         # 데이터 로드
size = 5                                           # timesteps 값. 

def split_x(dataset, size):                        # 함수선언 + 데이터,timestep 값 입력.
    aaa = []                                       # aaa라는 리스트 선언
    for i in range(len(dataset) - size + 1):       # 데이터셋(a)의 길이 - timesteps값 + 1만큼의 횟수만큼 반복하겠다
        subset = dataset[i : (i + size)]           # subset이 뭐냐면 -> 로드한 dataset을 슬라이싱해서 그 부분만큼만 불러옴 리스트형태로. -> 그걸 subset에 저장
        aaa.append(subset)                         # aaa라는 리스트에 subset을 추가하겠다
    return np.array(aaa)                           # 구해낸 aaa값을 반환해준다

bbb = split_x(a, size)                             # bbb = a를 size대로 자르겠다
print(bbb)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
print(bbb.shape) # (6, 5)

x = bbb[:, :-1]                                    # 행과 열을 각각 슬라이싱해줘야 한다. ,로 구분하고 :만 쓰는건 전체를 다 쓰겠다는 뜻
y = bbb[:, -1]
print(x, y)
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]] 
# [ 5  6  7  8  9 10]
print(x.shape, y.shape) # (6, 4) (6,)