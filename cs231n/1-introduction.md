---
description: Introduction to Convolutional Neural Networks for Visual Recognition
---

# 1. Introduction to CNN for Visual Recognition

이 Lecture에서는 Computer Vision에서의 CNN 역사와 Overview에 대해 이야기 한다. 

## History

![](../.gitbook/assets/image%20%28138%29.png)

1950년대. 포유류의 시각적 처리 메커니즘이 연구되었다. 고양이나 원숭로 실험을 했는데,  이 실험에서는  어떤 자극을 줘야 일차 시각 피질의 뉴런들이 격렬하게 반응하는 지를 관찰했다. Simple cells은 일차 시각피질에서만 찾아볼 수 있는 세포들로 빛의 방향에만 반응을 했다. Complex cells, Hypercomplex cells로 가며 점차 복잡한 정보가 들어와야 반응이  일어나는 걸 볼 수 있다. 이를 통해 시각 처리가 처음에는 단순한 구조로 시작되지만 시각처리화 과정을 거쳐가며 점차 복잡해진다는 것을 알게 되었다.

![Block world](../.gitbook/assets/image%20%28188%29.png)

1960년대. 이 논문에서는 우리 눈에 보이는 사물들을 단순화 시켰다. 우리 눈에 보이는 세상을 인식하고 그 모양을 재구성하는 것이다. 

![Stages of Visual Representation](../.gitbook/assets/image%20%28342%29.png)

1970년대. 눈으로 받아들인 Image에서 실제 정보와 유사한 3D Model로 가기 위한 과정이 표현했다. Primal Sketch에서는 Edges, Bars, Ends, Virtual Lines, Curves Boundaries 를 통해 단순한 구조로 표현을 했고, 그 이후에는 시각 장면을 구성하는 surface, depth 등을 이용해 2.5D sketch를 그렸다. 그리고 최종적으로 조직화 된 3D Model을 만들어 낸다.

![Generalized Cylinder, Pictorial Structure](../.gitbook/assets/image%20%28343%29.png)

Generalized Cylinder, Pictorial Structure 에서 각각 단순한 모양과 기하학적인 구성을 이용해 복잡한 객체를 단순화 하는 방법을 다뤘다. 모든 객체는 단순한 기하학적 형태로 표현할 수 있다는 메세지를 담고 있다.

![David Lowe](../.gitbook/assets/image%20%2857%29.png)

1980년대. 단순한 구조로 실제 세계를 재구성/인식할 수 있을지에 대한 문제를 풀고자 했다. David Lowe는 Lines, Edges, Straight lines들의 조합으로 실제 물체를 재구성했다.

60, 70, 80년대에는 컴퓨터 비전으로 어떤 일을 할 수 있을 지 고민하던 시대였다. Image Recognition이 쉽지  않다면 Image Segmentation 부터 접근해보자고 생각했다.

![Normalized Cut](../.gitbook/assets/image%20%28144%29.png)

Image Segmentation은 이미지의 각 픽셀을 의미있는 방향으로 군집화 하는 방법이다. 정확히 어떤 물체인지는 모르지만 적어도 배경과 사람이 별개의 객체라는 것을 구분할 수는 있었다. Normalized Cut에서는 Image Segmentation을 해결하기 위해 Graph 이론을 입했다.

![Face Detection](../.gitbook/assets/image%20%2815%29.png)

1999, 2000년대에는 Machine Learning, 그 중에서도 특히 Statistic Machine Learning 연구가 활발히 이루어졌다. Support Vector Machine, Boosting, Graphical models, 초기 Neural Network 등.

그 중 가장 큰 기여을 한 연구는, AdaBoost를 이용해 실시간 얼굴인식을 한 연구였다. 그 이후로 얼마 되지 않아 2006년에는 Fuji film이 실시간 얼굴인식을 하는 카메라를 개발했다. 

![SIFT](../.gitbook/assets/image%20%2821%29.png)

90년대 후반부터 2010년도까지 시대를 풍미했던 알고리즘은 Feature based Object Recognition 알고리즘이였다. 그 중 가장 유명한게 SIFT 알고리즘이다. 객체에는 여러가지 feature들이 있는데 그 중에서도 다양한 변화에도 좀 더 robust 하고 불변하는 feature가 있다는 점을 발견했다. 다양한 각도, 환경, 빛의 상황에도 큰 변화가 없는 이 feature들을 통해 객체를 인식 할 수 있다고 생각했다. 한 객체에서 이와 같은 feature들을 찾아내고, 그 feature들을 다른 객체들에 매칭하여 Object Recognition을 시도하였다.  

![Spatial Pyramid Matching](../.gitbook/assets/image%20%2828%29.png)

이미지에 있는 feature를 사용하기 시작하면서 Compute Vision은 또 한번의 도약을 했다. 이제 하나의 객체가 전체 이미지를 인식하기 시작했다. 아이디어는, feature들을 잘 뽑아낼 수만 있다면 그 feature들이 해당 이미지에 대한 일종의 단서를 제공해 줄 수 있다는 것이였다. 다양한 각도와 다양한 해상도, 여러가지의 이미지들에서 추출한 feature들을 하나의 feature 기술자로 표현하고, Support Vector Algorithm을 적용했다.

![Histogram of gradients, Deformable Part Model](../.gitbook/assets/image%20%28184%29.png)

이 아이디어는 사람 인식에도 사용됐다.이는 사람인식에도 이용됐다. 이런 feature들을 통해 어떻게 사람 몸을 현실적으로 모델링 할 수 있을 지 연구했다.

![PASCAL Visual Object Challenge\(VOC\)](../.gitbook/assets/image%20%28336%29.png)

2000년대. 이젠 컴퓨터 비전으로 앞으로 풀어야 할 문제가 무엇인지 어느 정도 정의가 내려졌다. 결국은 Object Recognition 이였다.

점차 사진의 품질이 좋아지고 인터넷, 카메라의 발전이 더 좋은 실험데이터를 만들어내고 있다. Object Recognition 기술이 어디쯤 왔는 지 측정해보기 위해 Benchmark Dataset을 모으기 시작했다. PASCAL VOC에는 20개의 클래스가 있고, 클래스 당 수천 수만개의 이미지들이 있다. 이를 통해 사람들이 자신들의 Object Recognition 알고리즘을 테스트하고 측정했다. Object Recognition 성능은 꾸준히 증가했다. 많은 진보가 이루어졌다.

## ImageNet

![ImageNet](../.gitbook/assets/image%20%28270%29.png)

Princeton, Stanford에선 더 어려운 질문을 던졌다. 우린 이 세상의 모든 객체들을 인식할 수 있을 것인가?

Graphical Model, SVM, AdaBoost 같은 ML 알고리즘들의 Object Recognition의 성능이 많이 좋아졌지만 Training 과정에서 Overfit 하는 경향을 보였다. 그 원인 중 하나는 Input으로 들어오는 데이터가 너무 복잡하다는 것이다. 모델의 Input은 복잡한 고차원 데이터고, 이로 인해 모델을 fit 하기 위해선 더 많은 파라미터가 필요했다. 학습 데이터가 부족해서 Overfiting이 빠르게 발생하고 Generalize 능력이 떨어졌다.

"이 세상 모든 객체를 인식하자", "머신러닝의 Overfiting 문제를 극복하자"

이 두가지 Motivation을 바탕으로 ImageNet 프로젝트가 시작됐다. 구할 수 있는 모든 이미지를 담은 가장 큰 데이터셋을 만드는 것이였다. 이 데이터 셋으로 모델들을 학습시킬 수 있도록.

인터넷에서 수십억장의 이미지를 받아 WordNet의 수천가지 객체 클래스로 Dictionary를 정리하고 이미지 정렬, 정제, 레이블을 제공하는 플랫폼을 통해 데이터셋을 만들어 갔다. AI 분야에서 만든 가장 큰 데이터셋이였다. 

이렇게 만들어진 ImageNet을 어떻게 Benchmark 할 것인지가 문제였다. ImageNet은 2009년부터 Image Classification 문제를 푸는 알고리즘을 위한 대회를 주최하기 시작했다.

![The Image Classification Challenge](../.gitbook/assets/image%20%28356%29.png)

매년 오답률이 낮아졌다. 사람보다 오답률이 낮은 해도 있었다. 여기서 주목해야 하는 해는 2012년이다. 오답률이 급격하게 떨어졌다. 이 때 우승 알고리즘이 바로 CNN 모델이였다. 이 때부터가 CNN, Deep Learning의 시작이였다. 이 이후 대회의 우승 알고리즘은 모두 CNN 이였다.

![ImageNet Winner](../.gitbook/assets/image%20%28185%29.png)

지난 몇 해간 ImageNet의 우승자이다. 

2011년 까지의 알고리즘은 hierarchy가 있었다. Feature를 뽑고 계산하고, 최종 feature 추출기를 SVM에 넣었다. 모든 과정이 단계적으로 일어났다.

2012년. Layer CNN을 만들었다. 이는 Alexnet, SuperVision으로도 알려져있다.

2014년. 네트워크가 한층 더 deep해졌다. GoogleNet과 VGG.

2015년. Residual Network. 무려 152개의 레이어를 사용했다. 200개 이상의 레이어를 쌓으면 성능이 더 좋아지겠지만 GPU 성능이 따라가지 못한다.

11’ hierarchy. Feature 뽑고, 특징 계산하고, 최종 feature 기술자를 SVM에 넣는다. 핵심은 여전히 계층적이라는 점이다.

14’ 네트워크가 한층 더 depp 해졌다. GoogleNEt과 VGG

‘15/ Residual Network. 152개의 레이어. 200개 이상의 레이어를 쌓으면 성능이 더 좋아지겠지만 GPU 성능이 따라가지 못함.

![](../.gitbook/assets/image%20%28331%29.png)

CNN이 최근에 급격히 주목받기 시작했지만 사실 CNN은 그 이전부터 있었다.

1998년. CNN의 기초 연구인 LeCun 연구. 숫자, 문자 인식을 위해 CNN을 도입했다. 이미지를 입력으로 받아 숫자와 문자를 인식할 수 있는 CNN을 만들었다. 몇차례 Convolution하고 Subsampling 하고 마지막에 Fully Connected를 쓴다. 지금의 CNN과 크게 다를 바가 없다.

2012년 AlexNet 구조를 봐도 이와 비슷하다. 추후 아키텍쳐들은 서로 비슷하다. 90년대의 LeNet 아키텍처를 공유하기 때문이다.

![Ingredients for Deep Learning](../.gitbook/assets/image%20%28335%29.png)

이미 잘 만들어져 있는 구조인데 왜 이제와서 유명해졌을까?

일단 컴퓨터 계산속도가 빨라졌다. GPU 진보다 한 몫 한다. GPU는 병렬처리가 가능한데 계산 집약적인 CNN 모델을 고속으로 처리하는데 안성맞춤이다. 연산량 증가는 딥러닝 역사에서 아주 중요한 요소이다. 

데이터 차이도 있다. Labeled 된 Image가 많이 필요했는데, 90년대에는 이런 데이터를 구하기 쉽지 않았다. 오늘날은 Pascal, ImageNet과 같은 사용 가능한 데이터셋이 훨씬 많다.

