# 7. Policy Gradient

## Policy-Based RL

gradient decente를 이용해 쎄타 parameter들을 업데이트햇었다.

policy를 parameter들을 이용해 표현하는 것. 함수로 표현.

* value function을 이용해 policy를 만들었었다. 
* policy based는 value function을 배우지 않고 policy를 직접 learing 시키는 거다. \(policy hradient\)
* 그 두가지를 다 학습시키는 것이 Actor-Critic

    actor : policy

    critic : evaluation

Policy Based RL

* 수렴성이 더 좋다
* action의 가짓수가 많은 경우 \(continuous action space의 경우 action의 갯수가 무한대..\)

  이런 경우 value 로 학습하기 어렵다. Q 학습하기도 어렵고, 수많은 action 값중에서 Q를 maximize 하는 action을 골라야 하기 때문에 이 자체도 optimize 문제이다

* deterministic 한 환경에서만 value를 학습시켰었다. \(greedy하게 업데이트했으니까\) stochastic한 환경에서 배울수 있다.
* local optimum에 빠질 수 있다.
* value based는 공격적인 학습방법이다. max를 취하니까 policy가 확확 바뀐다. policy based는 gradient 만큼 조금씩 업데이트하니까 stable 하다. 하지만 효율성이 떨어진다.

주먹가위보

* deterministic 한 policy는 쉽게 exploit된다. 하나의 선택만 하는거.
* uniform policy는 optimal 하다. 항상 공평하게 낸다.  비슷한 확률로

Aliased Gridworld feature가 완전하지 않은 상황. partially observable한 상황

* determinisitc policy는 회색칸에 대해 같은 policy를 가진다. feature가 불안전해서 어떤칸에 있는지 모른다. stock상황에 올 수 있다.

  차라리 확률적으로 왼쪽이나 오른쪽을 선택하는 policy가 있어야 양쪽의 칸으로 이동할 수 있다. 

  fully observable 한 경우는 deterministic policy가 optimal 이 보장되나, partially observable한 경우는 optimal이 보장되지 않는다.

policy는 파이라는 함수를 사용한다. parameter 쎄타에 대해 action a를 뱉어주는 함수. best parameter를 찾는다. maximize하고자 하는 목적함수\(objective function\)은? 이 policy를 따랐을 때 총 reward값이 가장 높은 policy가 좋은 policy 1. episodic env에서는 start value를 사용.

* 첫 state가 정해져있다고 하면 policy 파이를 따랐을 때 value 의 기댓값. 고정된 start value 분포가 있을 때 그때의 value의 기댓값을 맥시마이즈
* 각 state에 있을 확률 \* 그 state에서의 value. 그거의 총합. 

  d파이쎄타 : 각 state 마다 머무는 확률 분포. 

* policy를 하고 얻게되는 리워드. state distiributionasㅇ리ㅏㅓㅣㅜ룯지ㅏㅜㅇ 세개 다 value를 최대화하는 objective function 목적함수를 정의하는 방법들. 뭘 하든 똑같이 동작한다.

  Policy Optimisation optimization문제 : 어떤 함수가 있을 때 그 함수의 값을 최대화 하는 input 값을 찾는 문제. J\(쎄타\)를 maximize 하는 인풋 쎼타를 찾는다.

  Policy Gradient 목적 함수 J가 있을 때 이 gradient를 구할 수 있는 상황이 있다 쎄타 값을 바꿨을 때 J\(쎄타\)의 방향이 가장 급격한 방향으로 a 만큼 update 해준다. 저번엔 loss를 minimize하기 위해 descent 했지만 이번엔 policy를 높이기 위해 ascending 한다. 파이가 아닌 J를 maximize 한다. J에 대한 gradient를 구한다.

  Finite Difference n dimension 을 할 때 n번을 evaluatnioㅎ ㅐ야한다. 싦플하지만 nosiy하고 비효율적이다. 어떤 policy에 대해서도 미분가능하지 않아도 사용 가능하다.

Score Function MC Policy gradient. policy 파이가 항상 미분가능하다. 파이에 대한 gradient를 우리가 안다. 두가지 가정이 필요. logx를 미분하면 x분의 1이니까, log분의 x를 미본하면 x분의 dx.. 파이/gradient 파이 = grdient log 파이 gradient log 파이 : scroe function

one-step MDP 어떤 분포를 따르는 initila state 에서. initial state의 확률분포가 있고, 거기에서 확률적으로 initial state를 설정한다. 거기에서 딱 한 스텝 가고 reward 받고 끝난다. 이런 MDP가 있을 때 policy graident를 어떻게 구할까/ 쎄타 업데이트하기 위해 J를 미분하면, gradient 파이가 gradient log 파이 랑 같으니까 \(likelihood ratio\) 앞쪽이 expectation으로 묶인다. J에 대한 gradient가 기댓값으로 표현이 된다. 이걸 이용해 쎄타를 업데이트 해주면 된다.

* R\_s, a : MDP의 term. 확률변수
* r : immditae reward. sample. 엄밀히 말하면 unbiased 한 sample을 얻는 거다. 수많이 반복하다보면 J의 gradient에 수렴한다. 파이는 우리가 직접 정하는거.

  Softmax Policy Gaussian Policy

  multi-step MDP 에서는? likelihood ratio를 이용한 policy graidn theorm 증명된 수식. R이 Q로 대체된다. one-step 은 R이 곧 ccumulative reward 였지만 multi-step에선 Q가 그것을 말하는 거니까. 목적함수 3가지가 다 된다.

  Monte-Carlo Policy Gradient \(REINFOCE 알고리즘\) Q는 신이 알려주는 Q. 우린모르는데 그럼 어떻게? return을 쓴다. Q자리에 return을 써주면 된다. Q 자리에 v를 \(v는 return. G\_t\)넣어서ㅠㅠ ㅠㅠㅠ ㅠㅠㅠ 쎄타를 임의로 initiali한다. 쎄타를 이용한 파이가 있다. 게임을 해. episode 끝나면 처음부터 마지막까지 쎄타를 업데이트해. 점차 좋은 policy가 되어갈거다.

  stable해서. leargning curve가 지그재그하지 않고 예쁘게 올라가 .value based는 지그재그해. variance 가 너무 커서 learning이 느리다. \(MC라서\)

Actor-Critic MC는 variacne가 높은데 이 variance를 줄이기 위해. Actor-Critic 사용 Q 자리에 아까는 reward를 넣엇는데 그러지말고 Q를 학습해서 그냥 넣자! estimate 해서 넣자. Functino approximation을 Q에도 쓰는거다. Q \(w\) 도 학습하고 파이 \(쎄타\)도 학습한다. Q는 파이에 종속적이라 파이와 함께 학습한다.

Q를 학습 -&gt; policy evaluation \(MC나 TD 람다나 써라\) 델타는 td error. w를 학습

GPI의 또 다른 형태. Evaluation와 Improvement을 반복.

그리고 variacne를 줄이는 또다른 방법론 Baseline action의 상대적인 차이가 중요한데, 그 값 자체가 큰 경우 baseline을 빼줘서 그 variance를 줄인다. baseline의 기댓값이 0이면 위의 식이 그대로 성립한다. Q 자리에 advantage function이 들어와도 식은 성립한다. advnataqge function = Q - V

policy gradient의 variacne를 줄일 수 있다. paratmer가 세 쌍이 필요!!

V에 대한 paramter들로 Q를 계산할 수 있다. TD error는 advantage fucntion의 unbiases estiatme 즉, sample이다. -&gt; Q를 학습할 필요가 없다. paramter v 만 학습하면 된다.

critic 학습할 때 아래와 같이 ㅓ거저거저거 쓸 수 있다. actor 학습할 때도 differne time-scael 에서 학스밯 ㄹ 수 있다.

