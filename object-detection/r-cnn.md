---
description: R-CNN (Region with CNN features)
---

# R-CNN

이미지를 분류하는 것 보다 이미지 안에 어떤 물체들이 들어 있는지 구분해 내는 것이 훨씬 어려운 문제다.

## R-CNN

### Flows

![Flow](../.gitbook/assets/image%20%28171%29.png)

크게 가능한 이미지 영역을 찾아내는 Region Proposal \(Bounding box\)부분과 이미지 분류 부분 두가지로 나뉘게 된다.  
1. Input image로 부터 Bounding Box 찾기   
 : Selective Search 알고리즘 사용.  
     - 색상이나 강도 패턴 등이 비슷한 인접한 픽셀들을 합치는 방식  
     \(1\) 근접한 픽셀들끼리 위치, 특성을 파악하여 그룹핑 한다.  
     \(2\) RGB, HVS, 영역크기, hole의 유무 등 여러 요인을 고려하여 영역들을 merge 한다.  
     \(3\) 그 영역  boundary를 이어서 ROI 영역을 만든다.  
 : 입력 이미지의 Object를 스캔하고 최대 2000개의 region proposal을 만든다.  
    - IoU\(Intersection Of Union\) 에 따라 region proposal이 지시하는 해당 클래스인지 배경인지 여부를 결정한다.  
  
2. 추출한 region proposal들을 CNN의 고정된 input 크기에 맞추기 위해 강제로 사이즈를 일원화 한다.  
  
3. fine-tuning된 ConvNet을 통해 feature를 추출해낸다.  
  
4. 이미지 분류  
 : SVM\(Support Vector Machine\) 사용 \(Hinge Loss\)  
  
5. 최종적으로 분류된 오브젝트의 region proposal  좌표를 더 정확히 맞추기 위해 linear regression 모델 사용

![](../.gitbook/assets/image%20%2878%29.png)

 Input image에서 수많은 object 후보들을 찾아내고 이를 모두 CNN 에 넣어서 feature를 뽑아낸다.  
그리고 뽑아낸 feature들을 SVM등의 Classifier에 넣어서 classification 하고 NMS와 같은 기법으로 Bounding box를 이미지 위에 그린다.  
= Region Proposal 영역을 뽑아내고, CNN에 들어가는 image size로 wrapping 한 후에 region 마다 CNN에 각각 집어 넣는다. 그리고 계산된 CNN의 마지막의 feature를 통해 classification 한다.

### Limitation

R-CNN은 training/test 속도가 매우 느리다. R-CNN Training을 느리게 만드는 가장 주된 이유는 파이프라인이 3단계라는 점이다. \(CNN layer -&gt; Classification -&gt; BBox Regression\)

* 사용할 CNN을 fine-tuning 해야한다. \(이미 알려진 Network 중 성능이 좋은 모델의 weight를 가져옴\)
* fine-tune된 CNN에 맞게 SVM을 fitting 해야한다.
* Bounding Box regressor를 학습시켜야 한다.

  
이 3가지의 단계가 순차적으로 일어나기 때문에 시간이 오래 걸린다. 또한, 모든 region proposal에 대해 전체 CNN path를 다시 계산해야한다.  
또 CNN에 넣기 위해 input size를 맞춰줘야 하는데, 이게 일반적으로 정사각형이다. Region proposal 이미지가 직사각형이면 wrap 하는 과정에서 이미지의 왜곡이 일어나 학습 결과에 부정적인 영향을 끼칠 수 있다.   
  
SVM Classification 및 Bounding Box Regression 학습할 때 Backpropagation이 잘 안돼서 CNN의 상위 layer 부분에 학습 결과가 업데이트가 되지 않는다.  
Bounding Box Regression의 Loss함수로 MSE를 사용하는데, L2 Reg term이 매우 중요해서 λ = 1000 정도로 해준다. 이렇게 하면 처음의 initial ROI 에서 거의 움직임이 없는 영역을 찾아주게 된다. \(?\)  
  
이 이후에 Fast R-CNN으로 넘어가기 전에 SPP-Net\(Spatial Pyramid Pooling\)이 있었는데 이건 추후 추가

## Appendix

### Selective Search

![](../.gitbook/assets/image%20%28298%29.png)

 image에서 object의 region을 찾기 위한 알고리즘이다.  
Color space 정보와 다양한 similarity measure\(RGB, HVS, 영역 크기, hole의 유무 등\)를 활용하여 복잡한 segmentation들을 merge 하며 grouping 하고, 그렇게 만들어진 segmentation 의 boundary를 이어서 RoI 영역을 만든다.  
R-CNN, Fast R-CNN 에서 region proposal을 할 때 사용되었다.  
  
\* Motivation  
   - Sliding window approach is not feasible for object detection with convolutional neural networks.  
   - We need a more faster method to identify object candidates.  
\* Finding object proposals  
   - Greedy hierarchical superpixel segmentation  
   - Diversification of superpixel construction and merge \(Using a variety of color spaces, different similarity measures, varying staring regions\)

### BoundingBox Regression

![](../.gitbook/assets/image%20%28188%29.png)

Bounding box 의 parameter를 찾는 regression이다. 초기의 region proposal 이 CNN이 예측한 결과와 맞지 않을 수 있끼 때문. CNN의 마지막 pooling layer에서 얻은 feature 정보를 사용해 region proposal의 regression을 계산한다.

