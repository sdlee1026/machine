#!/usr/bin/env python
# coding: utf-8


# 여러 모델들에서 모델의 복잡도가 어떤 역할을 하는지 이야기하고 각 알고리즘들이 모델을 어떻게 만드는지 학습해보겠음
# 또한, 모델들의 장단점을 파악하고, 모델별로 어떤 데이터가 잘 들어맞을지 살펴 볼 것임
# 분류와 회귀 모델을 모두 가지고 있는 알고리즘도 많은데 이런 경우 둘다 살펴 볼 것임


import warnings

warnings.filterwarnings(action='ignore')
import collections
from IPython.display import display
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager, rc
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split

mpl.rcParams['axes.unicode_minus'] = False
font_fname = 'fonts/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()

rc('font', family=font_name)
# size, family
print('font size : ' + str(plt.rcParams['font.size']))
print('font family : ' + str(plt.rcParams['font.family']))
# import default setting


#  이진분류(classification)_forge dataset_시각화


# forge 데이터셋은 인위적으로 만든 이진 분류 데이터셋임


# dataset setting
X, y = mglearn.datasets.make_forge()
# 산점도를 그림 discrete_scatter 2차원 산점도 그래프를 위한..
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['클래스 0', '클래스1'], loc=4)
plt.xlabel('첫번째 특성')
plt.ylabel('두번째 특성')
print('X.shape', X.shape)
# dataset은 데이터26개(포인터)와 2개의 특성을 가짐


# # 회귀(regression)_wave dataset_시각화


# wave dataset : 입력특성 한 종류와 모델링을 위한 타깃변수(출력)을 가짐
# x축 : 입력 특성
# y축 : 회귀(regression)의 출력(타깃변수)


X, y = mglearn.datasets.make_wave(n_samples=40)
# n_sample : 입력 특성의 갯수
plt.plot(X, y, 'o')
plt.ylim(-2.8, 2.5)
# y축 범위 세팅
plt.xlabel('특성')
plt.ylabel('타깃')

# # 유방암 데이터셋


from sklearn.datasets import load_breast_cancer

# 유방암 데이터셋
# 각 종양은 양성종양(benign) 악성종양(malognant)로 나뉨(label화)
# 조직 데이터를 기반으로 종양이 악성인지 예측하는 것을 목표로 하겠음

cancer = load_breast_cancer()
print('type : ' + str(type(cancer)))
# scikit-learn에 포함된 데이터셋은 실제 데이터와 관련 정보를 가진 Bunch 객체에 저장되어 있음


# Bunch객체는 Dictionary와 비슷하지만, 점표기법(EX_.keys())을 사용할 수 있음
# ex) bunch['key'] 대신 bunch.key 가능
print('cancer.keys():\n', cancer.keys())
print('\n\ncancer.data:\n', cancer.data)
print('\n\ncancer.data:\n', cancer['data'])
# 동일하게 기능하는 것을 알 수 있다


print('유방암 데이터의 형태 : ', str(cancer.data.shape) + '\n569개의 데이터포인트와 30개의 특성을 가지고 있음')

print(cancer.target_names)
# 클래스 분류명
print(type(cancer.target))
print('0 : 악성, 1 : 양성 \n' + '[0,1]' + str(np.bincount(cancer.target)))
# 212 악성, 357 양성 종양
# bincount(~~)
# count of zeros is at first_index -> '0'
# count of ones is at second_index -> '1'


x = cancer.target
print(collections.Counter(x))
# collections.Counter()사용
print(np.count_nonzero(x))
# Numpy_count_nonzero 메소드 사용
print('클래스(분류)별 샘플 갯수  :\n',
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})

print('리스트 내부 포문 사용 하지 않을 시')
for n, v in zip(cancer.target_names, np.bincount(cancer.target)):
    print(n, v)

# # 보스턴 주택가격 데이터셋(회귀분석)


# 이 데이터셋을 이용, 범죄율, 찰스강 인접도, 고속도로 접근성 등의 정보를 이용해
# 1970년대 보스턴 주택 평균 가격을 예측하는 것
# 데이터 포인트 506, 특성 13


from sklearn.datasets import load_boston

boston = load_boston()
print('데이터의 형태', boston.data.shape)

# 마찬가지로 boston 객체의 DESCR 속성에서 정보 확인 가능
# 이 데이터셋에서 13개의 입력 특성뿐만 아니라 특성끼리 곱하여(상호작용) 의도적으로 확장할 것
# ex) 범죄율과 고속도로의 접근성의 개별 특성은 물론, 범죄율과 고속도로의 접근성의 곱의 벨류도 특성으로 생각한다는 것
# 이처럼 특성을 유도해내는 것을 특성 공학(feature engineering)이라함


X, y = mglearn.datasets.load_extended_boston()
# 13개의 특성, 중복을 허용한 특성의 곱에 원래 특성을 더해 총 104개의 특성을 가지게 됨
# 첫번째 특성으로 13개의 교차항이 만들어지고, 두번째 특성에서 첫번째 특성을 제외한 12개의 교차항이 ..(조합)
# 중복을 포함한 조합을 만듬 -> scikit-learn의 PolynomialFeatures 함수 사용


# 조합(이항계수)
# (13+12+13 ... +1 = 91) 91+13
# 이항계수(Binomial Coefficient)는 조합론에서 등장하는 개념으로 주어진 크기 집합에서 원하는 개수만큼 순서없이 뽑는 조합의 가짓수를 일컫는다.
# 2를 상징하는 ‘이항’이라는 말이 붙은 이유는 하나의 아이템에 대해서는 ‘뽑거나, 안 뽑거나’ 두 가지의 선택만이 있기 때문


print('이항계수 공식')
Image("./img/이항계수 공식.png")  # code안에서 나오게 할 때

print('이항계수 성질')
Image("./img/이항계수 성질.png")

# # KNN


# 가장 간단하 머신러닝 알고리즘
# 훈련 데이터 셋을 저장하는 것이 모델을 만드는 과정의 전부
# 복습) 새로운 데이터 포인트에 대한 예측을 할땐, 알고리즘이 훈련 데이터셋에서 가장 가까운 데이터 포인터, 즉 '최근접이웃'을 찾음


# forge 데이터셋에 대한 1-KNN 모델 예측 시각화
mglearn.plots.plot_knn_classification(n_neighbors=1)
# mglearn 라이브러리->plots들..->plot_knn분류
# 추가한 데이터 포인터 3개 별모양
# 그리고 추가한 데이터 포인터로 부터 가장 가까운 훈련 데이터 포인트를 연결함
# 1-KNN 알고리즘의 예측은 연결된 데이터 포인트에 연관되어 집니다(연결선)
# 1-KNN predict: blue, orange, blue


# forge 데이터셋에 대한 5-KNN 모델 예측 시각화
mglearn.plots.plot_knn_classification(n_neighbors=5)
# 연결된 데이터 포인트(이웃)의 색과 수에 따라서 예측 결과값은 달라짐
# 5-KNN predict : orange, orange, blue


# 위의 데이터 셋은 이진 분류 문제지만, 클래스가 다수인 데이터셋에도 KNN을 적용할 수 있음
