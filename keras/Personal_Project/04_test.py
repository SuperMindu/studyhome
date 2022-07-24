# SeamlessClone을 활용한 이미지 합성 (seamlessclone.py)

import cv2
import numpy as np
import matplotlib.pylab as plt
 
#--① 합성 대상 영상 읽기
img1 = cv2.imread("c:/study/image/4.jpg")
img2= cv2.imread("c:/study/image/5.jpg")
# cropped_img1 = img1[y: y + h, x: x + w]
# cropped_img2 = img2[y: y + h, x: x + w]

#--② 마스크 생성, 합성할 이미지 전체 영역을 255로 셋팅
mask1 = np.full_like(img1, 255)
mask2 = np.full_like(img2, 255)
 
#--③ 합성 대상 좌표 계산(img2의 중앙)
height, width = img2.shape[:2]
center = (width//2, height//2)
 
#--④ seamlessClone 으로 합성 
normal = cv2.seamlessClone(img1, img2, mask1, center, cv2.NORMAL_CLONE)
mixed = cv2.seamlessClone(img1, img2, mask2, center, cv2.MIXED_CLONE)

#--⑤ 결과 출력
cv2.imshow('normal', normal)
cv2.imshow('mixed', mixed)
cv2.waitKey()
cv2.destroyAllWindows()

img = cv2.imread('test.jpg')

