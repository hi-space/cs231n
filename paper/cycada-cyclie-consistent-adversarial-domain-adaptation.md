---
description: ICML 2018
---

# CyCADA: Cyclie-Consistent Adversarial Domain Adaptation

## Related Work

### Unsupervised Domain Adaptation by Backpropagation \(ICML 2015\)

기존에 학습하던 모델에 별도의 discriminator을 추가하고, 이를 통해 domain classify loss를 reverse하여 전달하는 방식으로 두 도메인을 구분하지 못하도록 학습하는 방법이니다. 

서로 다른 두 개의 loss \( $$L_y$$ , $$L_d$$ \)가 있는데 각각의 역할은 아래와 같다.

![](../.gitbook/assets/image%20%28339%29.png)

*  $$L_y$$ : 우리가 학습하려고 하는 label에 대해서 잘 구분할 수 있는 feature extractor와 label predictor를 학습하는 것



![](../.gitbook/assets/image%20%28376%29.png)

*  $$L_d$$ : domain classifer를 위한 loss로 domain을 잘 구분할 수 있도록 학습을 시킨다. 

그리고 그 이후에 feature extractor에 domain classifier의 gradient를 전달해줄 때 이를 reverse하여서 \(-1을 곱하여서\) 전달해준다. 이렇게 하면 feature extractor 입장에서는 domain을 최대한 구분하지 못하게 학습이 되고, 위에서 살펴본 우리가 원하는 효과를 얻을 수 있다.

### Adversarial Discriminative Domain Adaptation \(CVPR 2017\)

domain을 섞기 위해 gradient를 reverse 해서 전달하는 것이 아니라, discriminator에 fake label을 주는 방식으로 학습을 진행하는 것이다. 

![](../.gitbook/assets/image%20%2828%29.png)

domain real label에 대해서 discriminator를 학습한다. \(주어진 도로 환경의 사진이 한국인지 미국인지 잘 구분하도록\)

![](../.gitbook/assets/image%20%28230%29.png)

잘 학습된 discriminator에 domain fake label을 주고 학습하면, feature extractor 입장에서는 domain을 구분하지 못하도록 학습이 된다. 

gradient reversal 방식은 domain discriminator가 converge 하게 될 경우 feature extractor로 전달될 gradient가 vanish 하는 문제가 발생하기 때문에 ADDA 방식이 더 효과적이다.

### CycleGAN

![](../.gitbook/assets/image%20%28319%29.png)

서로 다른 domain의 unpaired image들에 대해서 image-to-image translation 하는 네트워

## CYCADA

* ADDA의 경우 feature space 단에서 adversarial loss를 적용하기 때문에 pixel-level이나 low-level domain shifts를 잘 잡아내지 못한다.
* ADDA에 CycleGAN을 추가

![](../.gitbook/assets/image%20%28216%29.png)



### Evaluation

![](../.gitbook/assets/image%20%28155%29.png)





{% embed url="https://blog.lunit.io/2018/09/14/cycada-cycle-consistent-adversarial-domain-adaptation/" %}



