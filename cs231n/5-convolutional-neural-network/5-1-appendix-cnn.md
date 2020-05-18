# 5-1. Appendix CNN

## CNN

![](../../.gitbook/assets/image%20%28356%29.png)

Fully Connected Layer만으로 구성된 인공 신경망의 입력 데이터는 1차원\(배열\) 형태로 한정됩니다. 한 장의 컬러 사진은 3차원 데이터입니다. 배치 모드에 사용되는 여러장의 사진은 4차원 데이터입니다. 사진 데이터로 전연결\(FC, Fully Connected\) 신경망을 학습시켜야 할 경우에, 3차원 사진 데이터를 1차원으로 평면화시켜야 합니다. 사진 데이터를 평면화 시키는 과정에서 공간 정보가 손실될 수밖에 없습니다. 결과적으로 이미지 공간 정보 유실로 인한 정보 부족으로 인공 신경망이 특징을 추출 및 학습이 비효율적이고 정확도를 높이는데 한계가 있습니다. 이미지의 공간 정보를 유지한 상태로 학습이 가능한 모델이 바로 CNN 입니다.

CNN은 기존 Fully Connected Neural Network와 비교하여 다음과 같은 차별성을 갖습니다.

* 각 레이어의 입출력 데이터의 형상 유지
* 이미지의 공간 정보를 유지하면서 인접 이미지와의 특징을 효과적으로 인식
* 복수의 필터로 이미지의 특징 추출 및 학습
* 추출한 이미지의 특징을 모으고 강화하는 Pooling 레이어
* 필터를 공유 파라미터로 사용하기 때문에, 일반 인공 신경망과 비교하여 학습 파라미터가 매우 적음

CNN은 위 이미지와 같이 이미지의 특징을 추출하는 부분과 클래스를 분류하는 부분으로 나눌 수 있습니다. 특징 추출 영역은 Convolution Layer와 Pooling Layer를 여러 겹 쌓는 형태로 구성됩니다. Convolution Layer는 입력 데이터에 필터를 적용 후 활성화 함수를 반영하는 필수 요소입니다. Convolutional Layer 다음에 위치하는 Pooling Layer는 선택적인 레이어입니다. CNN 마지막 부분에는 이미지 분류를 위한 Fully Connected 레이어가 추가됩니다. 이미지의 특징을 추출하는 부분과 이미지를 분류하는 부분 사이에 이미지 형태의 데이터를 배열 형태로 만드는 Flatten 레이어가 위치 합니다.

CNN은 이미지 특징 추출을 위하여 입력데이터를 필터가 순회하며 합성곱을 계산하고, 그 계산 결과를 이용하여 Feature map을 만듭니다. Convolution Layer는 Filter 크기, Stride, Padding 적용 여부, Max Pooling 크기에 따라서 출력 데이터의 Shape이 변경됩니다.

### Convolutional Layer

![](../../.gitbook/assets/image%20%28350%29.png)

* **Sparse Connectivity** : Each neural only connects to part of the output of the previous layer
* **Parameter Sharing** : The neurons with different receptive fields can use the same set of parameters. 

Convolutional layer에서는 filter를 사용해서 특정 영역 \(receptive field\) 안의 데이터만 연산하여 하나의 출력값 \(피처맵의 한칸\)을 구하였다. 반면에 FC에서는 입력값 전체를 weighted sum하여 하나의 출력값을 구한다. 당연하게도 연산량이 엄청나게 늘어나며, 학습하여야 할 파라미터의 수가 엄청나게 많아진다. 

## 레이어별 출력 데이터 사이즈

### Convolution Layer 출력 데이터

* 입력 데이터 높이: H
* 입력 데이터 폭: W
* 필터 높이: FH
* 필터 폭: FW
* Stride 크기: S
* 패딩 사이즈: P

$$
OutputHeight=(H+2P−FH)/S+1
\\OutputWeight=(W+2P−FW)/S+1
$$

### Pooling Layer 출력 데이터

* 입력 데이터 높이: H
* 입력 데이터 폭: W
* Pooling 크기:  P
* Stride 크기: S

$$
OutputRowSize = (W- P) / S + 1\\
OutputColumnSize = (H-P) /S + 1
$$

## Example

![](../../.gitbook/assets/image%20%28302%29.png)

![](../../.gitbook/assets/image%20%2899%29.png)

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.net = new Sequential(
            # shape : (39, 31, 1)
            
            # filter : (4, 4, 20) / parameters : 4 x 4 x 20 = 320
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, 
                        stride=1, padding=0
            nn.ReLU(),
            
            # shape : (36, 28, 20)
            
            # pooling : (2, 2) / parameters : 0
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # shape : (18, 14, 20)
            
            # filter : (3, 3, 40) / parameters : 3 x 3 x 40 = 360
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, 
                        stride=1, padding=0),
            nn.ReLU(),

            # shape : (16, 12, 40)
            
            # pooling : (2, 2) / parameters : 0
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # shape : (8, 6, 40)
            
            # filter : (3, 3, 60) / parameters : 3 x 3 x 60 = 540
            nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3,
                        stride=1, padding=0),
            nn.ReLU(),
            
            # shape : (6, 4, 60)
            
            # pooling : (2, 2) / parameters : 0
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # shape : (3, 2, 60)
            
            # filter : (2, 2, 80) / parameters : 2 x 2 x 80 = 320
            nn.Conv2d(in_channels=60, out_channels=80, kernel_size=2,
                        stride=1, padding=0),
            nn.ReLU(),
            
            # shape : (2, 1, 80)
            
            # parmeters : 0
            nn.Flatten(),
            
            # shape : (2 x 1 x 80, 1)
            
            # parameters : 160 x num_classes
            nn.Linear(160, num_classes)
            
            # shape : (num_classes, 1)
        )
    
    def forward(self, x):
        return self.net(x)
```

{% hint style="info" %}
[http://taewan.kim/post/cnn/](http://taewan.kim/post/cnn/)
{% endhint %}

위의 예시에서 parameter 갯수는은 weight 의 학습 파라미터만 계산한 값이다. bias 값이 고려되지 않았다.

## 학습 파라미터 갯수

### Input Layer

단순히 input 값을 읽어오는 layer 이기 때문에 학습 파라미터가 없다.

### Conv Layer

![](../../.gitbook/assets/image%20%28182%29.png)

위의 그림을 예시로 들어보면, \(W x H x 32\)의 입력에 \(3 x 3 x 32\) 필터가 64개 취해진다. \(모든 필터의 depth는 항상 input 의 채널 값과 같다\) 

* FW = 3
* FH = 3
* C = 32
* N = 64
* weight 학습 파라미터 수 = FW x FH x C x N                                                    \(3 x 3 x 32 x 64\)
* bias 학습 파라미터 수 = N \(32\)
* ConvLayer의 학습 파라미터 수 = \(FW x FW x C + 1\) x N

공식으로 정리하면 다음과 같다.

* 필터 폭 : FW
* 필터 높이 : FH
* 필터 갯수 :  N \(출력 채널\)
* 입력 채널 : C

$$
NumOfParameters = (FW * FH * C + 1) * N
$$

### Pooling Layer

n x n  행렬에서 max 값만 뽑아내는 대체하는 단순 연산이므로 학습 파라미터가 없다.

### Fully-Connected Layer

모든 input 값들이 각각의 weight와 각각의 output을 가지고 있으므로, 각 weight의 갯수는 N x M 이고 bias 의 갯수는 N 개 이다. 정리하면 \(N + 1\) x M 이 된다.

* input 사이즈 : N
* output 사이즈 : M 

$$
NumOfParameters = (N+1) * M
$$

### Output Layer

출력 레이어는 보통 fully-connected layer이다. 그러므로 \(N +1\) x M 개의 학습 파라미터를 가지고 있다.

### Example

| Layer | Filter size |
| :--- | :--- |
| input | \(1, 28, 28\) |
| conv2d\_1 | \(5, 5, 32\) |
| maxpool\_1 | \(2, 2\) |
| conv2d\_2 | \(3, 3, 32\) |
| maxpool\_2 | \(2, 2\) |
| dense \(fc\) | \(256, 1\) |
| output | \(10, 1\) |

위와 같이 구성된 네트워크가 있으면 각 레이어에서의 output size와 parameters 값을 계산해보면 아래와 같다.

```scheme
#     layer          output size                    parameters
---  --------  -------------------------    ------------------------
  0  input                        1x28x28                           0
  1  conv2d_1   (28-(5-1))=24 -> 32x24x24    (5*5*1+1)*32   =     832
  2  maxpool_1                   32x12x12                           0
  3  conv2d_2   (12-(3-1))=10 -> 32x10x10    (3*3*32+1)*32  =    9248
  4  maxpool_2                     32x5x5                           0
  5  dense (fc)                       256    (32*5*5+1)*256 =  205056
  6  output                            10    (256+1)*10     =    2570
```

결국 이 네트워크의 학습 파라미터는 0 + 832 + 9248 + 0 + 205056 + 2570 = 217706 이 된다.

## Example

![AlexNet Example](../../.gitbook/assets/image%20%28242%29.png)

| **Layer Name** | **Tensor Size** | **Weights** | **Biases** | **Parameters** |
| :--- | :--- | :--- | :--- | :--- |
| Input Image | 227x227x3 | 0 | 0 | 0 |
| Conv-1 | 55x55x96 | 34,848 | 96 | 34,944 |
| MaxPool-1 | 27x27x96 | 0 | 0 | 0 |
| Conv-2 | 27x27x256 | 614,400 | 256 | 614,656 |
| MaxPool-2 | 13x13x256 | 0 | 0 | 0 |
| Conv-3 | 13x13x384 | 884,736 | 384 | 885,120 |
| Conv-4 | 13x13x384 | 1,327,104 | 384 | 1,327,488 |
| Conv-5 | 13x13x256 | 884,736 | 256 | 884,992 |
| MaxPool-3 | 6x6x256 | 0 | 0 | 0 |
| FC-1 | 4096×1 | 37,748,736 | 4,096 | 37,752,832 |
| FC-2 | 4096×1 | 16,777,216 | 4,096 | 16,781,312 |
| FC-3 | 1000×1 | 4,096,000 | 1,000 | 4,097,000 |
| Output | 1000×1 | 0 | 0 | 0 |
| **Total** |  |  |  | **62,378,344** |

