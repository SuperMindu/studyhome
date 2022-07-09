# 이건 컬러, 분류가 100개 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D # 이미지 작업은 이거
from tensorflow.python.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.datasets import cifar100
import numpy as np
