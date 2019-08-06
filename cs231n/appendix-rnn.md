# --- Appendix RNN

## RNN \(Recurrent and Recursive Nets\)

 Sequence Modeling : Recurrent and Recursive Nets

![Feed-forward Net \(FFNet\)](../.gitbook/assets/image%20%28250%29.png)

일반적으로 우리가 알고 있는 neural network인 feed-forward neural network \(FFNet\) 구조이다. input 데이터를 넣으면 hidden layer를 거쳐 ouput  까지 차례대로 진행되는 형태이다. hidden layer의 노드를 딱 한번만 지나가도록 되어 있다. 현재 주어진 input 값에 대한 ouput 값만을 찾기 때문에 이전에 어떤 데이터가 나왔었고 이후에 어떤 데이터가 나올지에 대한 고려가 전혀 되어 있지 않았다. 즉, input 값과 output 값이 각각 독립적이였다.

![RNN Nodes](../.gitbook/assets/image%20%28204%29.png)

 반면에 RNN은 hidden layer의 output 값이 다시 hidden layer의 input 값으로 들어가고 있다. 과거 자신의 state를 기억하고 이를 학습에 반영하게 때문에 sequential data를 다루기에 좋은 network 구조이다.

![RNN](../.gitbook/assets/image%20%28283%29.png)

RNN을 간략하게 나타내면 위와 같은 형태로 도식화 할 수 있다. 현재 시점의 hidden state는 현재 input 값과 이전 시점의 hidden state를 받아서 갱신된다. 현재까지 계산된 결과에 대한 메모리 정보를 가지고 있다고 보면 된다.

$$
h_t = f_W(h_{t-1}, x_t) \\
 ↓ \\\
h_t = tanh(W_{hh}h_{t-1} +  W_{xh}x_t) \\
y_t = W_{hy}h_t
$$

 Activation 함수로는 주로 tanh, relu를 사용한다. 출력값의 범위를 제한해주면서 \(\[-1, 1\], \[0, 1\]\) 전 구간 미분이 가능하기 때문에 backpropagation이 잘 되기 때문이다.

* $$W_{hh}$$ : $$h_t$$ 와 $$h_{t-1}$$의 관계를 나타내는 Matrix로, Markov 체인의 transition matrix와 비슷하다고 보면 된다.
* $$W_{xh}$$ ****: 지금 들어온 input 값이 얼마나 중요한지 판단하는 값. 이 값이 크면 input 값이 이전의 데이터 값\(이전 hidden state 값\)보다 중요하다고 판단

![RNN Unfold](../.gitbook/assets/image%20%2842%29.png)

RNN의 recurrent한 구조를 unfold 하면 위와 같이 볼 수 있다.

* $$X$$ : t 에서의 input 값
* $$S$$ : t 에서의 hidden state. 네트워크의 '메모리' 부분. t-1스텝의 hidden state와 t 스텝의 input 값에 의해 계산됨
* $$O$$ : t에서의 output 값 \(ex, 확률 벡터, softmax\)

s는 과거의 time step에서 일어난 일들에 대한 정보를 모두 담고 있고, o는 현재 step의 메모리에만 의존한다.

FFNet의 경우, layer 마다 parameter 값들이 전부 다 달랐지만, RNN은 모든 시간 step에 대해 prameter 값을 전부 공유하고 있다. 그렇기 때문에 학습해야 하는 parameter의 수가 많이 줄어든다. \(the same function and the same set of parameters are used at every time step\)

### Example : Character level model

![Character level model](../.gitbook/assets/image%20%28246%29.png)

RNN의 예제로 자주 나오는 Character level model 이다. 어떤 문자가 나오면 바로 그 다음의 글자를 예측해주는 모델이다.

학습하기에 앞서 일단 One-hot-encoding으로 h, e, l, o를 아래와 같은 vector형태로 변환한다.

```python
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 1, 0]
```



 'h' 문자를 input으로 넣으면 $$W_{xh}$$ 파라미터를 이용해  h1\(0.3, -0.1, 0.9\) 값을 만들고, $$W_{hy}$$ 파라미터를 이용해 y1값을 생성한다. Output layer 에서는 softmax를 취해서 다음에 나올 글자에 대한 score를 받게 된다. 초록색으로 표시된 부분이 정답에 해당하는 index를 의미하는데, 위의 예시로 봤을 때는 정확하지 않은 것을 볼 수 있다. 이 정확도의 평균을 loss 함수로 해서 값이 최소화 하도록 backpropagation을 수행하며 parameter 값들\( $$W_{xh}$$,$$W_{hh}$$, $$W_{hy}$$\)을 갱신해 나가게 된다.

* $$W_{xh}$$ : input layer -&gt; hidden layer로 가는 parameter
* $$W_{hh}$$ : 현재 hidden layer -&gt; 다음 hidden layer로 가는 parameter
* $$W_{hy}$$ : 현재 hidden layer -&gt;output layer로 가는 parameter

모든 시점의 state에서 이 parameter 값들은 동일하게 적용된다.

### 학습 방법

그럼 이 RNN은 어떻게 학습을 시킬까? 다른 neural network와 마찬가지로 backpropagation과 gradient descent를 사용하여 학습한다.

FFnets의 backprop은 출력단에서 구한 오차를 신경망을 거슬러 가면서 각 parameter를 업데이트 한다. 각 parameter가 output layer의 오차에 얼마나 기여하는 지를 계산하고, 그 기여도\(편미분 ∂E/∂w\) 만큼 업데이트 하게 된다.

![BPTT \(Backpropagation Through Time\)](../.gitbook/assets/image%20%2892%29.png)

RNN은 backprop을 조금 변형한 BPTT를 사용해 parameter를 학습시킨다. 기본 backprop 개념과 같은데, 다만 시간을 거슬러 올라가며 backprop을 적용한다는 점만 다르다. 각 출력 부분에서의 gradient가 현재 시간 step 에만 의존하지 않고, 이전 시간 step들에도 의존하기 때문에 이와 같은 방법을 사용한다.

Forward through entire sequence to compute loss, then backward through entire sequence to compute gradient

![](../.gitbook/assets/image%20%28141%29.png)

매 time step 마다 W에 대한 gradient를 더해준다. 즉, 위와 같이 t = 4의 gradient를 계산하기 위해서는 time step 3개 이전의 gradient를 전부 더해줘야 한다. 기존의 NN 구조에서는 layer 별로 parameter를 공유하지 않기 때문에 계산 결과들을 서로 더해줄 필요가 없었다. 에러값들을 더하듯이 매 time step의 gradient도 하나의 학습 데이터에 대해 모두 더해준다.

![Truncated BPTT](../.gitbook/assets/image%20%28180%29.png)

Truncated BPTT는 시간 전체를 거슬러 올라가는 BPTT를 간략화한 것이다. 데이터가 길어지면 hidden layer에 저장해야 하는 양이 계속 늘어나기 때문에 메모리도 부족해지고 성능도 나빠질 수 있다. 그래도 일정 범위까지만 기억을 하는 것이다. 모든 것을 기억할 수 없다는 현실적인 문제가 있어서 사용되곤 한다.

Carry hidden states forward in time forever, but only backpropagate for some smaller number of steps

![](../.gitbook/assets/image%20%2893%29.png)

deep learning book을 보면 BPTT 말고도 Teacher forcing이라는 학습 방법도 있다. 

이전 hidden state를 넘겨주는 것이 아니라, target ouput 값을 현재 hidden layer로 넘겨주는 방식이다. 말 그대로 정답을 바로 가르쳐주면서 훈련시키는 방식인데, 이전 state 값들을 고려하지 않고 단순히 input 값에 대한 output 값만 주게 된다.

계산량이 많이 줄어들어서 training이 좀 간단해지긴 하지만 training 데이터 외의 데이터가 들어온 경우에는 정확도가 많이 떨어진다. Fragile 하거나 limited 하기 때문에 많이 쓰진 않는다. 예측 성능이 저하 된다. 

간혹 어떤 모델은 BPTT와 Teacher forcing을 혼합하여 트레이능 시키는 경우도 있다고 한다. 아마 training 시간을 줄이기 위함인 듯 하다. \(애초 RNN 컨셉과는 맞지 않는 training 방법인데 왜 사용되는 지 잘 모르겠다.\)

> *  We originally motivated teacher forcing as allowing us to avoid back-propagation through time in models that lack hidden-to-hidden connections. Teacher forcing may still be applied to models that have hidden-to-hidden connections so long as they have connections from the output at one time step to values computed in the next time step. However, as soon as the hidden units become a function of earlier time steps, the BPTT algorithm is necessary. Some models may thus be trained with both teacher forcing and BPTT.
> * However, as soon as the hidden units become a function of earlier time steps, the BPTT algorithm is necessary
> * One approach commonly used for models that predict a discrete value output, such as a word, is to perform a search across the predicted probabilities for each word to generate a number of likely candidate output sequences. This approach is used on problems like machine translation to refine the translated output sequence.

![Recurrence through only the Output](../.gitbook/assets/image%20%28157%29.png)

* feedback connection from the output to the hidden layer
* Unless o is very high-dimensional and rich, it will usually lack important information from the past. 
* Less powerful, but it may be easier to train because each time step can be trained in isolation from the others, allowing greater parallelization during training

### RNN Architecture

![Recurrent Networks offer a lot of flexibility](../.gitbook/assets/image%20%28296%29.png)

 RNN은 시퀀스 길이에 관계 없이 input, output을 받을 수 있기 때문에 유연하게 네트워크를 설계할 수 있다.

* One-to-One : 일반적인 Vanilla Neural Networks 형태.
* One-to-many : Image Captioning. 하나의 input image 가 들어가게 되면, 단어들을 나열하여 captioning 한 output을 준다.
* Many-to-one : Sentiment Classification. 단어들의 나열을 통해서 상태나 감정 정보를 파악
* Many-to-many : Machine Translation
* Many-to-many : video frame들을 분석해서 video classification

![](../.gitbook/assets/image%20%2844%29.png)

RNN 네트워크 형태를 변형하여 이와 같은 구조도 설계할 수 있다.

* Bidirectional RNN
* Deep RNN
* Encoder-Decoder RNN

![Sequence to Sequence Architecture](../.gitbook/assets/image%20%2835%29.png)

* Encoder RNN : input sequence를 read
* Decoder RNN : output sequence를 generate \(vector-to-sequence\)
* Input and output sequences sets not of the same length
* C : input sequence를 요약한 context. Fixed-size vector

Encoder RNN의 마지막 hidden unit의 state는 C로 사용되고, 이 C는 Decoder RNN의 input으로 들어간다.

One clear limitation of this architecture is when the context C output by the encoder RNN has a dimension that is too small to properly summarize a long sequence.

![Long-term dependencies](../.gitbook/assets/image%20%28129%29.png)

이 RNN의 구조가 가지고 있는 가장큰 한계는 Long-term dependencies 이다. 멀리 있는 데이터들에 대한 학습 내용은 잘 기억하지 못한다는 것이다. 예를 들어 문장의 의미를 파악한다고 했을 때, 주요 단어들 사이에 많은 시간 step이 지나게 되면 잘 기억을 하지 못한다. 가까이 있지 않은 단어들끼리 밀접한 관계가 있을 수도 있는데, 이런 부분에 대한 처리가 잘 되지 않는다.

Backprop을 하기 위해서는 한 cycle에 대해서 gradient를 계속 계산하고 또 곱해서 다음 gradient를 업데이트 하는 과정을 반복적으로 수행하게 되는데, 이 gradient 값이 곱셈이기 때문에 학습하면서\(과거로 거슬러 올라가면서\) 계산이 잘 되지 않는다. \(ex, gradient 값이 0.1일 때 0.1 을 두번만 곱해도 0.01로 크게 줄어든다. 학습이 거의 되지 않는 형태로 간다는 뜻이다\)

반복적으로 곱을 하다보면 아주 큰 값으로 발산하는 exploding gradient problem과 0으로 수렴하는 vanishing gradient problem이 발생하게 된다.

![Gradient Clipping](../.gitbook/assets/image%20%28255%29.png)

그래도 Exploding gradient 값은 경우는, 최대값을 제한해주는 방식을 통해 어느 정도 제어가 가능하다.

Gradient Clipping : gradient의 최대갯수를 제한하고, 그 최대치를 넘으면 크기를 재조정하여 gradient를 크기를 조정한다. 가야할 방향은 유지하면서 업데이트 되어야 하는 learning rate를 조절한다.

![Exploding gradient problem](../.gitbook/assets/image%20%2875%29.png)

 그보다 큰 문제는 Vanishing gradient problem 이다. gradient 값이 0에 가까워질 수록 학습 속도가 매우 느려지거나, 학습 자체가 잘 되지 않을 수 있다.

위의 그림과 같이 sigmoid를 4번만 곱해도 거의 0에 수렴하게 되는데, 100개의 sequence 데이터를 학습시킨다고 생각하면 당연히 학습이 안될 수 밖에 없다.

그래서 이를 해결하기 위해 RNN 네트워크 구조를 변형시킨 LSTM, GRU 같은 네트워크 구조들이 제안되었다.

## LSTM \(Long Short-Term Memory Units\)

LSTM은 RNN에서 발생하는 vanishing gradient 문제를 해결하기 위해 고안되었다. 더 오래 전 일들을 더 잘 기억하도록.

![Vanilla RNN](../.gitbook/assets/image%20%2841%29.png)

![LSTM](../.gitbook/assets/image%20%2898%29.png)

LSTM은 기존 RNN의 hidden state 에 cell state가 추가된 형태이다. LSTM unit에 있는 cell은 여러개의 gate가 연결되어 있는 형태로 구성되어 있다. 각 gate들도 gradient descent를 이용해서 학습이 되는데, 한마디로 cell state와 hidden state가 재귀적으로 구해지는 네트워크라고 볼 수 있다.

![Cell state](../.gitbook/assets/image%20%2856%29.png)

가장 기본이 되는 건 맨 위에 이 cell state를 나타내는 C 부분. 컨베이어 벨트에 비유되어 많이 설명 된다. 기억하고자 하는 정보는 이 cell의 흐름에 올리고, 잊고자 하는 정보는 제거한다. 데이터가 컨베이어 벨트를 타고 흘러가는 느낌이다.

이 부분에서 어떤 정보를 잊어버릴지, 불러올 지, 유지할 지, 내부적으로 결정을 해서 다음 cell state \( $$C_t$$ \) 값으로 넘겨준다.  
 $$C$$ 값은 0 ~ 1 사이의 값으로 나오게 되는데, 0이면 cell state를 잊어버리고, 1이면 기억하겠다는 의미이다. 값이 크면 클 수록 새로운 정보를 더 기억하려고 한다.  
기존에는 $$t$$ 와 $$h_{t-1}$$ 값에 의해서 바로 게이트 값이 정해졌었는데, 그 사이에 정보를 잊고, 기억하는 과정이 추가 된 것이다.

이 Cell state는 forget gate, input gate, output gate 3가지의 gate로 구성되어 있다.

![Forget gate](../.gitbook/assets/image%20%28111%29.png)

모든 정보를 기억하는 것이 항상 좋지만은 않다. 적당히 잊어버리는 부분도 필요하다. 예를 들어, 하나의 문서를 다 읽고 다음 문서를 다루기 시작할 땐 기억을 싹 다 지우고 새로 시작하는 편이 더 정확할 수 있다. 이 LSTM의 핵심은 'forget' 이다.

가장 처음으로 거치게 되는 gate는 Forget gate이다. 입력값과 이전 hidden state 값을 보고 새로 들어오는 $$x_t$$ 값을 잊어버릴 지 가져갈 지 결정하게 된다. activation 함수로는 sigmoid를 사용해서 0 ~ 1 사이의 값을 가지게 한다. 연산된 값 $$f_t$$ 는 \*\(미분곱\) 해서 cell state로 올린다.

![Input Gate](../.gitbook/assets/image%20%28131%29.png)

어떤 정보를 cell state에 담을 지, 그 input 값을 결정하는 gate 이다.

* i\(t\) : sigmoid layer가 어떤 값을 업데이트 할 지 결정하고
* C'\(t\) : tanh layer가 새로운 후보 벡터값들을 만든다

sigmoid는 output 값의 범위가 \[0, 1\],  tanh는 \[-1, 1\]이기 때문에 각각 강도와 방향을 나타낸다?

a sigmoid layer called the “input gate layer” decides which values we’ll update.   
a tanh layer creates a vector of new candidate values

![](../.gitbook/assets/image%20%2824%29.png)

forget gate에서 나온 값과 input gate에서 나온 값을 더해준다. forget 하기로 했던 것을 forget 하고 각 state 값을 얼마나 갱신할 지 결정한 값을 cell state에 올려주는 것이다. 일반적인 RNN unit은 곱으로만 이루어져 있어서 vanishing gradient 문제가 발생했는데, LSTM에서는 더하기로 forget과 input 값을 잇고 있어서 그 문제를 해결하고자 했다.

![Output gate](../.gitbook/assets/image%20%28128%29.png)

Output gate에서는 어떤 값을 출력할 지를 결정한다. h\(t\) 값.

원래 우리가 내보내려고 했던 값들을 sigmoid 취하고, cell state에 tanh를 취해서 \[-1, 1\] 사이 값을 만들어 곱함으로써, 우리가 cell state를 통해 결정한 값들을 내보낼 수 있다.

### Variants on LSTM

![Variants on LSTM](../.gitbook/assets/image%20%28123%29.png)

 이후에 이 LSTM을 약간 변형해서 사용한 모델들도 있다.

* **peenhole connections** : peenhole을 만들어서 gate layer가 cell state를 지켜보도록 변형
* **coupled forget and input gates** : Forget gate와 Input gate를 합친 형태. 무언가를 입력하려고 할 땐 기존의 것을 잊는 경향이 있다,는 아이디어에서 나온 형태이다. 오래 된 어떤 것을 잊을 때만 새 값들을 상태에 입력.

그리고 가장 유명한 변형 모델은 GRU 이다.

## GRU \(Gated Recurrent Unit\)

![GRU](../.gitbook/assets/image%20%28280%29.png)

LSTM의 장점을 유지하면서 계산 복잡성을 낮춘 cell 구조이다. LSTM과 유사하지만, gate의 일부를 생략해, 2개의 gate로 만들었다.

* Update gate : 이전 메모리를 얼만큼 기억할 지 결정
* Reset gate : 새로운 입력을 이전 메모리와 어떻게 합칠지 결정

LSTM의 forget gate의 역할이 update, reset gate에 나뉘어져 있다. output 값 계산 할 때 추가적인 비선형 함수를 적용하지 않는다. 

* $$h$$ : 현 시점에서 기억해 둘 만한 정보. 현 시점 정보 W\(xt\)와 과거 정보 Uh\(t-1\)를 반영하되, 과거 정보를 얼마나 반영할 지는 reset gate 값에 의존한다.
* S : S\(t-1\)의 과거 정보, h는 현재 정보. 이를 어떻게 조합할 지는 update gate의 값에 의해 결정.

output gate를 생략한 형태라서 항상 메모리에서 결과를 출력한다.

