# Training Deep Networks with Synthetic Data Bridging the Reality Gap by Domain Randomization

### Introduction

직접 데이터셋을 구축하는 경우, 데잍어를 취득하고 Labeling 하는 데 많은 시간과 비용이 발생한다. 이런 문제를 해결하기 위해 Simulator로 실제 이미지와 비슷한 이미지를 생성하는 연구들이 많이 존재한다. 하지만  이 또한 Simulator를 제작하는 시간, 비용, 인력 등이 필요한 것은 마찬가지며 한계이다.  
그렇기 때문에 Domain Randomization 기법을 적용하여 low cost로 대량의 이미지를 합성하여 데이터셋을 만들고 향상시키는 방법에 대해 제안한다.

* Domain Randomization을 Object Detection에 적용하는 방법 제안
* Domain Randomization에 flying distractors 를 제안하여 정확도 향상
* Domain Randomization의 주요 parameter들에 대한 분석을 통해 각각의 중요성 평가

### Synthetic Dataset

* SYNTHIA \(http://synthia-dataset.net/\)
* GTA V \(https://arxiv.org/pdf/1608.02192.pdf\)
* Sim4CV \(https://sim4cv.org/\#portfolioModal2\)
* Virtual KITTI \(http://www.europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds\)

### Previous Work

* On Pre-Trained Image Features and Synthetic Images for Deep Learning
  * pretrained weight를 사용하여 앞 단의 layer는 freezing 시키는 식으로 fine-tuning을 하는 반면, 본 논문에서는 freezing 시키는 방법을 사용하지 않는 것이 더 성능이 좋다고 주장
* Driving in the matrix: Can virtual world replace human-generated annotations for real world tasks?
  * Photorealistic Synthetic 데이터셋을 이용하여 자동차를 detection 하는 문제 해결. 본 논문에서는 Photorealistic Synthetic 데이터셋 사용 대신 Domain Randomization 기반의 Synthetic 데이터셋 사용

### Domain Randomization

![](../.gitbook/assets/image%20%28293%29.png)

