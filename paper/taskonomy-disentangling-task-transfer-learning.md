# Taskonomy: Disentangling Task Transfer Learning

CVPR 2018에서 best paper award 를 받을 정도로 인정을 받은 논문이다.

## Introduction

이 논문은 다음과 같은 질문을 던지면서 시작합니다. “과연 task들이 관계를 가지고 있을까?” 여기서 말하는 task는 object classification, semantic segmentation 등등 컴퓨터 비전 분야에서 풀고자 하는 문제들을 의미합니다. 일단 task들이 관계를 가지고 있을 것은 당연해 보입니다. 왜냐하면 실제로 많은 증거들이 있기 때문입니다. 예를 들어서 풀고자 하는 visual task의 데이터셋이 부족할 때\(심지어는 부족하지 않더라도\), object classification을 다룬 imagenet 데이터셋을 transfer learning 시키면 더 높은 성능을 얻을 수 있다는 것은 최근에는 거의 상식으로 받아들여지고 있습니다. 이것은 **풀고자 하는 task가 visual task이므로 object classification과 비슷한 관계를 가지고 있기 때문에** 가능합니다. 하지만 각각의 task가 얼마나 object classification과 비슷한지는 다 다르기 때문에 transfer learning 이후의 성능은 다 제각각입니다.

Transfer learning과 task relation은 실제 문제를 해결할 때 굉장히 중요한 이슈입니다. 만약 우리가 풀고자 하는 _target task_ A가 labeling cost가 많은 문제라고 가정합시다. 그래서 이에 대한 해결책으로 비슷한 task이면서 label을 얻기 쉬운 다른 _source task_ B를 골라서, B의 dataset을 모으\(거나 기존의 데이터셋을 활용하\)고 task B -&gt; task A의 transfer learning을 사용해서 최대한의 성능을 끌어내는 방법을 사용하기로 결정했다고 가정합니다. 그러면 어떤 task A, B를 골라야 가장 효율적인 학습이 가능할까요? Taskonomy는 **최적의 source task 및 target task를 고를 수 있는 fully-computational한 방법을 제시**합니다.

복수 개의 서로 다른 task들 간에 잠재적으로 존재하는 이러한 관계들을 graph 형태로 구조화하여 표현하고, 이를 사용하여 어느 새로운 task에 대한 딥러닝 모델의 학습을 보다 효과적이고\(성능 향상\) 효율적으로\(레이블링된 데이터의 양 감소\) 할 수 있는 **Taskonomy** 방법을 제안하였습니다.

## Related Work

\(1\) **Self-Supervised Learning**은, 레이블링 비용이 상대적으로 낮은 task로부터 학습한 정보를 활용하여, 그와 내재적으로 관련되어 있으면서 레이블링 비용이 더 높은 task에 대한 학습을 시도하는 방법들을 통칭합니다. 이는 source task를 사전에 사람이 수동으로 지정해줘야 한다는 측면에서, Taskonomy 방법과 차이가 있습니다.

\(2\) **Unsupervised Learning**은, 각 task들에 대한 명시적인 레이블이 주어지지 않은 상황에서 데이터셋 자체에 공통적으로 내재되어 있는 속성을 표현하는 feature representation을 찾아내는 것과 관련되어 있습니다. Taskonomy 방법의 경우 각 task 별 레이블을 명시적으로 필요로 하며, task 상호 간의 transferability 극대화를 위한 feature representation 활용에 초점을 맞추고 있다는 측면에서 차이가 있습니다.

\(3\) **Meta-Learning**은, 러닝 모델 학습을 meta-level\(상위 레벨\)에서의 좀 더 ‘추상화된’ 관점으로 조명하고, ‘학습 데이터셋의 종류를 막론하고, 모델을 좀 더 효과적으로 학습하기 위한 일반적인 방법’을 찾아내는 데 초점이 맞춰져 있습니다. 복수 개의 task들 간의 transferability를 좀 더 meta-level에서 조망하면서 이들의 structure를 찾기 위한 일반적인 방법을 제안한다는 점에서, Taskonomy 방법과 일종의 공통점이 있다고 할 수 있습니다.

\(4\) **Multi-Task Learning**은, 입력 데이터가 하나로 정해져 있을 때 이를 기반으로 여러 task들에 대한 예측 결과들을 동시에 출력할 수 있도록 하는 방법을 연구하는 주제입니다. 대상이 되는 task들을 동시에 커버할 수 있는 feature representation을 찾는다는 측면에서 Taskonomy 방법과 공통점이 일부 존재하나, Taskonomy 방법은 두 task들 간의 관계를 명시적으로 모델링한다는 측면에서 차이가 있습니다.

\(5\) **Domain Adaptation**은 transfer learning의 하나의 특수한 형태로, task는 동일하나 입력 데이터의 domain\(도메인; 속성\)이 크게 달라지는 경우\(source domain -&gt; target domain\) 최적의 transfer policy를 찾기 위한 연구 주제입니다. Taskonomy의 경우 domain이 아닌 task가 달라지는 경우를 가정하기 때문에, 이와는 차이가 있습니다.

\(6\) **Learning Theoretic** 방법들은 위의 주제들과 조금씩 겹치는 부분들이 존재하며, ‘모델의 generalization\(일반화\) 성능을 담보하기 위한’ 방법들에 해당합니다. 단 기존에 나와 있던 다양한 Learning Theoretic 방법들의 경우 대부분 intractable한 계산들을 포함하거나, 이러한 계산들을 배제하기 위해 모델 또는 task에 많은 제한을 둔 바 있습니다. Taskonomy 방법의 아이디어는 Learning Theoretic 방법들로부터 일부 영감을 얻었다고 할 수 있으나, 엄밀한 이론적 증명을 피하면서 좀 더 실용적인 접근을 시도한 것이라고 할 수 있습니다.

## Task

실제 논문에서 제시한 task는 총 26개입니다. 모든 task들은 실제 indoor scene inference에서 많이 쓰입니다.

![](../.gitbook/assets/image%20%28101%29.png)

* Surface Normals
* Image Reshading
* 2D Texture Edges
* Vanishing Points
* Unsupervised 2.5D Segm.
* Room Layout
* Scene Classification
* 3D Keypoints
* 3D Occlusion Edges
* Autoencoding
* Euclidean Distance
* Semantic Segmentation
* Unsupervised 2D Segm.
* 3D Curvature
* 2D Keypoints
* Object Classification
* Z-buffer Depth
* Denoising
* Autoencoding
* Colorization
* Image In-painting

각 task에 대한 더 자세한 내용은 논문 보충 자료에 있습니다.  Task들이 워낙 많다 보니 이 task들을 일일이 label을 얻는 것도 힘들고, 설사 공들여 label을 받았다 하더라도 거의 비슷한 task들도 있고 그래서 “이 중에 몇 개만 fully-annotation 받고, 나머지는 fine-tuning 하면 되지 않을까?”라는 생각을 할 수 있습니다.

## Method

![](../.gitbook/assets/image%20%28227%29.png)

Taskonomy는 어떤 task 들의 set에서 어떤 task를 labeling하고 어떤 task를 fine-tuning하면 되는지를 나타낸 도식입니다. 그래프의 각 node는 task를 나타내고 edge의 source는 source task, target은 target task를 의미합니다. 그래프를 보시면 한 target에 2개 이상의 source가 있는 경우가 있는데, 말 그대로 2개 이상의 source를 사용해 fine-tuning한 것을 의미합니다.

* maximum order of transfer function : 이 parameter를 사용해서 최대 몇 개의 source task를 동시에 사용할 수 있는지를 지정할 수 있습니다.
* budget : 얼마 만큼 많은 label을 받았는지를 의미합니다. budget이 아주 많으면 모든 task에 대해서 fully-annotation을 받으면 되고, budget이 부족하면 최대한 작은 task만 source task로 정의해서 annotation을 받고 나머지는 조금만 labeling을 받아서 source task를 fine-tuning 해야 합니다.

![](../.gitbook/assets/image%20%28219%29.png)

Taskonomy는 다음과 같은 방식으로 만들어집니다.

1. **Task-Specific Modeling**  : S 내의 각 task에 대해 특화된 모델인 task-specific network를 각각 독립적으로 학습합니다. 
2. **Transfer Modeling**  : 지정된 transfer order k 하에서, 서로 간의 조합 연산을 통해 만들어지는 source task\(s\) -&gt; target task 의 각 조합 별 transferability가 수치화된 형태로 계산됩니다.
3. **Ordinal Normalization using Analytic Hierarchy Process \(AHP\)**  : 앞서 계산된 transferability에 대한 normalization\(정규화\)를 통해 affinity matrix를 얻습니다.
4. **Computing the Global Taxonomy**  : 각 target task에 대하여 최적의 성능을 발휘하는 transfer policy를 탐색합니다.

### 1. Task-Specific Modeling

![Task-Specific Modeling](../.gitbook/assets/image%20%28242%29.png)

먼저 Source task를 학습시킵니다. 이 때 network는 encoder와 decoder로 이루어져 있습니다. Encoder의 결과로 image의 task에 대한 representation들이 학습되고, decoder를 통해서 pixel-level prediction 혹은 single prediction이 학습될 것입니다.

### 2. Transfer Modeling

![](../.gitbook/assets/image%20%289%29.png)

 Encoder는 고정한 채 target task들에 대해서 decoder를 fine-tuning합니다. 여기에 쓰이는 데이터는 source data에 비해서 훨씬 작은 양으로 학습이 가능합니다. taskonomy를 만들기 위해서는 **모든 source task와 target task의 pair**에 대해서 transfer learning을 시행해야 합니다.

High-Order transfer

하나의 target task를 학습시키는 데 두 개 이상의 source task의 representation이 필요합니다. Taskonomy를 만들기 위해서는 물론 이 pair들 모두에 대해서도 학습을 시켜 봐야 합니다. 하지만 high-order로 갈 수록 가능한 조합의 수가 많아지기 때문에 현실적으로 어렵습니다. 따라서 논문에서는 _beam search_를 사용했습니다: 가장 성능이 좋은 5개\(k≤5인 경우\) source task들에 대해서만 combination을 해서 다음 order 후보군을 fine-tuning합니다.

### 3. Ordinal Normalization using AHP

![](../.gitbook/assets/image%20%2840%29.png)

앞선 step에서 계산된 네트워크를 가지고 affinity matrix를 구합니다. 논문에서는 이 matrix를 그냥 구할 시에는 각 task별 bias 때문에 제대로 된 affinity가 구해지지 않으므로 사회학 등에서 사용하는 AHP라는 방법을 사용해 normalization을 진행한다고 합니다.

### 4. Computing the Global Taxonomy

위에서 계산된 affinity matrix를 사용해서 global taxonomy를 계산합니다. 앞서 설명한 바와 같이 global taxonomy의 목적은 어떤 task를 source task로 삼아서 labeling하고 어떤 task를 fine-tuning하면 되는지를 알아보는 것입니다. 따라서 다음과 같은 제약 사항이 들어갑니다.

1. if a transfer is included in the subgraph, all of its source nodes/tasks must be included too
2. each target task has exactly one transfer in
3. supervision budget is not exceeded

1번 조건은 그래프 모양을 만들기 위해서 필요합니다. 그리고 하나의 target task를 학습시키기 위한 두가지 이상의 방법을 제시할 필요는 없기 때문에 2번 조건이 존재합니다. 그리고 budget 상황에 따라서 다양한 taskonomy를 제시하기 위해서 3번 조건이 필요합니다. 논문에서는 _binary integer programming\(BIP\)_를 사용해서 문제를 해결했습니다. 문제를 BIP로 수식화 할 수 있으면, 보통 library를 사용해서 optimization을 수행할 수 있기 때문에 현명한 접근입니다.

{% embed url="http://research.sualab.com/review/2018/08/14/taskonomy-task-transfer-learning.html" %}

{% embed url="https://blog.lunit.io/2018/09/05/taskonomy-disentangling-task-transfer-learning/" %}



