# Transfer Learning

 처음부터 CNN을 학습시키려고 하면 너무 시간이 오래 걸리고 효율이 떨어지니까 성능이 입증된 CNN 모델을 가져가다 feature를 추출하고, 이를 바탕으로 우리가 원하는 Classification을 수행하도록 만드는 것이다. 실질적으로 전체 CNN을 처음부터 끝까지 학습시키기에는 데이터셋이 너무 크기 때문에 불가능하다. 이미 대규모 데이터를 대상으로 학습이 끝난 ConvNet을 가져다가 초기값으로 사용하거나 fixed feature extractor로 사용할 수 있다.  
  
 - **ConvNet as fixed feature extractor**   
: ConvNet의 끝에 있는 classification layer를 제거하고 convolutional layer를 통해 처리 되는 값을 얻으면 완전한 feature extractor가 된다.\(이렇게 얻어진 feature를 CNN codes라고 함\) 이 CNN codes와새로운 training set을 사용해 linear classifier\(ex, Linear SVM or Softmax\)를 학습한다.  
     -&gt; 마지막의 Classification layer만 retrain  
  
 - **Fine-tuning the ConvNet**   
: 끝의 fully-connected layer도 없애고 앞 단의 convolutional layer를 새로운 데이터로 다시 학습시켜서 역전파를 통해 weight를 업데이트 한다. training 데이터가 많을 때 사용할 수 있는 방법이다. 경우에 따라 앞쪽 레이어는 고정시키고 뒤쪽 레이어만 fine-tuning 하기도 한다. \(앞단의 레이어를 통해 얻어지는 것은 직선이나 곡선같은 general한 feature를 학습하지만, 뒷단의 고차원적 feature는 특정 도메인에 종속 될 수 있다\)  
     -&gt; pretrain된 전체 네트워크를 재조정\(fine-tuning\)  
  
Q\) 언제, 어떻게 transfer learning 을 할 것인가?  
 - 새 데이터가 작지만 원래 데이터와 비슷한 경우 : CNN codes를 이용해서 linear classifier를 학습  
 - 새 데이터가 크고 원래 데이터와 비슷한 경우 : 데이터가 많으니까 fine-tuning through the full network  
 - 새 데이터가 작고 원래 데이터와 많이 다른 경우 : 데이터가 적으니까 분류기만 학습. 데이터셋 자체가 많이 다르니까 ConvNet을 모두 통과한 결과보다는 앞 쪽 레이어의 값들을 사용해서 분류기를 학습시킴.  
 - 새 데이터가 크고 원래 데이터와 많이 다른 경우 : 데이터가 많으니까 처음부터 CNN을 구축해도 됨. 그래도 pretrained model의 weight로 초기값을 설정하고 학습시키는게 더 낫다. convolution layer를 처음부터 끝까지 fine-tuning 하는 것도 가능함.  


## Transfer Learning / Domain Adaptation

딥 러닝을 적용하는 경우에는 더 많은 데이터를 필요로 합니다.  일반적인 학습 환경에서 데이터는 input과 label의 쌍으로 이루어져있죠. 모든 문제에서 ImageNet같이 큰 표준 dataset이 존재한다면 좋겠지만, 현실에서는 데이터 부족에 시달리는 경우가 많습니다.

Label이 부족한 경우에는 ImageNet 등에서 pre-trained 된 모델을 베이스로 해서 transfer learning을 시행하는 것이 가장 일반적인 접근법입니다. 그러나 어떤 문제들에서는 transfer learning에 필요한 만큼의 데이터를 구하는 것도 어려울 때가 있습니다. Sample의 label을 얻는 작업은 항상 지루한 data notation 및 refinement 과정을 거쳐야 하고, 심지어 어떤 task들의 경우에는 label이 아예 구할 수 없거나 극소량의 input sample들만 존재하는 경우도 있기 때문이죠.

Domain adaptation 기법\(이하 DA\)이 바로 이 때 고려할 수 있는 방법들 중 하나입니다. 예를 들어 target domain에 정답이 아예 없고 입력 영상들만 있다고 가정하면, 일반적인 transfer learning은 아예 적용할 수 없습니다. 그러나 DA는 어떻게든 이전 task에서 배운 지식을 사용해 \(정답이 없는\) 새로운 상황에서도 맞출 확률을 올려주는 것을 목표로 합니다.

* Transfer learning is commonly understood to be the problem of taking what you learned in problem A and applying it to problem B
* Domain adaptation, however, is a slightly different story. In this case, what you're interested in doing is to adapt some general model that you have to a particular sub-problem. Sometimes people use this term in the sense of adapting a model trained on Domain A to Domain B, which is quite similar to the transfer learning problem. But in other contexts, it could refer to adapting a model trained on Domain A to many different sub-domains.

## Domain Adaptation

 학습을 위해서는 가장 기본적으로 필요한 것이 교본**\(training data, or labeled data\)**입니다. 그러나 보통 이런 교본은 매우 값이 비싸지요. 때에 따라서는 현실적으로 training data를 만드는 것이 불가능한 경우까지도 있습니다. 게다가 우리가 알고 있는 학습 기법들은 많은 경우 이미 학습된 혹은 training data **domain**이 test data **domain**과 비슷한 경우에 효율적으로 동작합니다.  
  
따라서 이런 여러 상황을 고려할 때, 가장 먼저 생각해볼 수 있는 것이 이미 알고 있는 지식을 이용해서 새로운 상황을 학습하는데 사용해보는 것**\(knowledge transfer\)**입니다.  
  
벌써 어느 정도 감이 오실텐데요 이런 아이디어를 실제로 적용해본 것이 **domain adaptation**입니다. 우리가 이미 알고 있는 training data domain의 분포**\(source knowledge\)**로부터 시작해서 새로운 test data domain 분포**\(target knowledge\)**와 비슷하게 조정해나가면 아무래도 아무 것도 없는 상황에서 새로 학습하는 것보다 좀 더 낫겠죠.  
  
이런 개념들은 이미 1995년 NIPS workshop에서 "Learning to learn"이라는 주제로 소개가 되었을만큼 아주 오랜 역사를 가지고 있습니다. 이해를 돕기 위해 실제로 여러 분야에서 유용하게 연구된 예시들을 보면 다음과 같습니다.

1. 가장 직관적인 예시로, 영화에 대한 리뷰가 긍정적인지 부정적인지를 판별하는 것을 학습시킨 모델에서 책에 대한 리뷰를 판별하도록 하는 sentiment classification이 있겠습니다. 꼭 영화와 책이 아니더라도 다양한 상품에 대해 개별적으로 학습을 시키려면 대량의 교본\(labeled data\)가 필요할 것이나 transfer learning을 이용한다면 좀 더 효율적으로 학습을 할 수 있습니다.
2. 또한 데이터가 자주 갱신되어 시간이 지나면 특정 시간대에 수집한 데이터로 학습한 모델이 다른 시간대에 수집한 데이터에 잘 적용이 안되는 경우에도 비슷한 논리로 적용을 해볼 수 있겠네요.

  
이렇게 training과 test 분포들이 서로 약간 다른\(presence of shift\) 환경에서 효율적으로 학습을 하려는 것이 domain adaptation \(DA\) 입니다. 이 것이 잘 된다면 아주 큰 반향이 있을 수 있는 것이 비지도 학습\(unsupervised learning\)에 상당한 영향을 줄 수 있기 때문입니다. 세상에는 답이 있는 경우보다 답이 없는 경우가 훨씬 더 많으니까요.



## Meta Learning

배우는 방법을 배우는 모델인데, 다소 생소할 수 있지만, 우리도 어떤 과목을 공부하지만 공부 잘하는 법 자체도 공부하기도 했었죠? 바로 그것입니다. 자동화된 딥러닝 모델로 보시면 될 듯 합니다. 이게 발전되면 더이상 딥러닝 모델을 만들 필요가 없습니다. 딥러닝 모델이 문제를 보고 적당한 딥러닝 모델을 만들테니깐요.

Deep NN 같이 복잡한 모델의 경우 학습 시키는데 상당한 시간이 많이 들며, 새로운 데이터의 양이 충분히 확보되지 않은 경우 좋은 성능을 나타낼 수 없다. Transfer Learning 이러한 한계점을 극복하기 위해, 고안 되었다. 기존 학습 환경\(Source Domain\)과 새로운 환경\(Target Domain\) 간의 유사점을 바탕으로, Target과 유사한 Source 내의 데이터 셋을 추출해서 활용하던지, Source에서 생성된 모델의 하이퍼파라미터 및 구조를 그대로 사용하는 연구 등이 진행되어 왔다. 하지만 기존의 Transfer Learning의 한계점은 Target과 Source 많은 공통점을 갖고 있다는 가정이다. 사실 주어진 Target과 Source 이 실제로 유사한 점이 있는지 확인하는 것 또한 많은 비용이 존재 할 뿐만 아니라, 축적했던 Source가 매우 다양한 경우 유사한 Source 은 찾아낸 것 또한 많은 시간과 비용이 드는 작업이다. 이러한 Transfer Learning한계점을 극복하기 위해, Meta Learning 연구들이 활발히 진행되고 있었다. Meta Learning의 프레임워크는 다음과 같다. 기존에 다양한 Source 간 특징을 반영하는 하는 모델1 \(meta-learner\) 하나와 Source 에서 전달 받은 정보를 활용에서 실제 분류/예측 문제를 해결하는 모델2 총 2두개의 모델이 존재한다. 예를 들어 초기 주어진 파라미터 셋팅에서 모델2가 분류/예측을 진행했을 때 발생하는 에러를 바탕으로 모델2는 모델1의 초기 파라미터 값을 정한다. 반복적 학습을 통해 meta leaner모델1은 A라는 Target이 들어왔을 때 적절한 초기 모델의 파라미터를 뱉어주기 때문에 적은 영의 데이터 상황에서 빠르고 정확한 모델 피팅이 이뤄진다. Meta-learning은 초기 모델 파라미터 선정 이외에 다양한 부분에서 연구되고 있다. Meta-learning 핵심은 인간의 다재다능함을 학습시키는 것이다. 특정 Source에 포함되어 있는 특징만을 학습하는 것이 아니라, 새로운 Source가 나타났을 때 현재 모델에서 어떻게 변화되어야 하는지 유연함을 학습시키는 것으로 볼 수 있다.

Meta-learning, also known as “learning to learn”, intends to design models that can learn new skills or adapt to new environments rapidly with a few training examples. There are three common approaches: 

1\) learn an efficient distance metric \(metric-based\);

2\) use \(recurrent\) network with external or internal memory \(model-based\);

3\) optimize the model parameters explicitly for fast learning \(optimization-based\)

