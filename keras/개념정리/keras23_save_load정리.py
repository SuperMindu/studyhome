#### model.save('저장경로') 
# 모델 구성 바로 다음에 해주면 그냥 모델만 세이브됨
# fit 단계 다음에 해주면 모델과 weight 까지 저장됨

#### model = load_model('저장경로') 
# fit 다음에 save 해준 모델을 컴파일, 훈련단계 위에서 다시 불러오면 weight값이 새로 구해지고 기존 weight에 덮어 씌워짐
# 그래서 제일 좋게 나온 weight 값을 그것만 따로 불러오려면 fit 다음에 불러와야 함 



#### model.save_weights('저장경로')
# fit 단계 다음에 해줘야 함

#### model.load_weights('저장경로')
# fit 다음에 save한 파일을 모델 밑에다 불러와주면 됨
# 얘는 훈련한 다음의 가중치가 저장 돼 있어서 loss와 r2가 동일하게 나옴 (3단계에서 컴파일만 해주면 됨)



# save_weights, load_weights는 일반 save와 다르게 model = Sequential()과 model.compile()해줘야 사용이 가능함 
# 저장된 weights를 불러올 때는 모델구성, compile을 해주면 됨 (fit 생략)
# fit단계 전에 하냐 후에 하냐에 따라 차이가 있지만 후에 쓰는게 바른 방법이고 그래야 값이 저장됨