# Model Free Control

이번 단원에서는 환경 없이 최적의 policy를 찾는 방법에 대해서 배워볼 예정이다. 이전 강의에서는 policy를 estimate 했지만, 이번에는 optimal을 찾을거다. 이번 장에서 배울 control 문제는 여전히 table lookup 방식을 사용하고 있는데, state가 많아지만 이런 방식으로는 풀 수 없다. 전체적인 reinforcement learning의 흐름을 배운다고 보면 된다. 나중에는 수많은 state에 대해 대응할 수 있는 function approximate을 배울 예정이다.

## Introduction

![Uses of Model-Free Control](../.gitbook/assets/image%20%28411%29.png)

MDP 모델을 모르거나, MDP 모델을 알더라도 문제가 너무 커서 sampling 해서 풀어야 할 경우, model-free control 방식을 사용한다. 

![On and Off-Policy Learning](../.gitbook/assets/image%20%28413%29.png)

Model-Free Control 방식에는 On-policy 와 Off-policy 두가지 방식이 있다.

* On-policy 

On-policy는 최적화 하고자 하는 policy 와 환경에서 경험을 쌓는 behavior policy 두가지가 같은 것이고, Off-Policy는 다른 policy가 쌓아놓은 경험으로부터 배우는 것이다. 그래서 "Look over someone's shoulder" 라고 표현한다.

![Generalised Policy Iteration](../.gitbook/assets/image%20%28407%29.png)

이전에 optimal policy를 찾는 과정인 policy iteration에 대해서 배웠다. 위의 방식을 사용해서는 model-free에서 사용할 수는 없을까? Monte-carlo policy evaluation은 model free에서도 풀 수 있었는데. 

![Generalised Policy Iteration With Monte-Carlo Evaluation](../.gitbook/assets/image%20%28410%29.png)

$$V$$만 학습해서는 greedy 하게 선택을 할 수가 없다. model-free 환경이기 때문에 다음 state를 알 수 없기 때문이다. model을 알아야 $$V$$에 대한 greedy policy를 improvement 할 수 있다.

![Model-Free Policy Iteration Using Action-Value Function](../.gitbook/assets/image%20%28408%29.png)

그렇다면 Q에 대해서 greedy policy improvement 할 수 있을까? 그렇다. $$V(s)$$는 못구하더라도 $$Q(s, a)$$는 구할 수 있다. 어떤 state에서 action을 취해보고 나온 return 값들의 평균을 구하기만 하면 되기 때문에 $$Q$$에 대한 greedy policy improvement는 할 수 있다. 

![Generalised Policy Iteration with Action-Value Function](../.gitbook/assets/image%20%28405%29.png)

$$Q$$를 이용해서 하고 policy evaluation을 하고, policy improvement는 greedy 하게 선택하면 되는걸까? 아니다. 이 경우 greedy 하게만 움직이면 충분히 많은 곳을 가볼 수 없기 때문에 exploration이 되지 않는다.

![Example of Greedy Action Selection](../.gitbook/assets/image%20%28409%29.png)

두가지 문 중 하나의 문을 선택하는 문제가 있다고 하자. left로 갔을 때의 value 값은 0이고 right 로 갔을 때의 value 값을 받는다고 했을 때, Greedy Action Selection 을 하게 되면 계속해서 left 문을 다시 시도하지 않고 right 문만 선택하게 된다. 

![e-Greedy Exploration](../.gitbook/assets/image%20%28404%29.png)

그래서 나온 것이 $$\epsilon$$-greedy Exploration 이다. $$\epsilon$$의 확률로 랜덤하게 다른 action을 선택하는 거다. 그럼 모든 action을 exploration 할 수 있다. 

![](../.gitbook/assets/image%20%28412%29.png)

 $$\epsilon$$-greedy 를 사용해도 policy가 improve 되는 것을 증명하는 



입실론 그리디를 이용해도 policy가 improve 되는지 증명하는 슬라이드.

--

그래서 이제, Q를 평가, 그리고 greedy 대신 입실로 greedy imporevement.

이걸 더 효율적으로 할 수 있는 방법? monte carlo로 수렴할때까지 할 수도 있겠지만 evaluation때 끝까지 가지 않고 바로 improvement. 빨리 지그재그해서 빨리 imporevment 하기위해.

잘 수렴하기 위해선 몇가지 성질이 필요하다. 1. 모든 state action 페어들이 무한히 많이 반복되어야 한다. \(exploration\) 2. policy는 결국 greedy policy에 수렴해야 한다. \(explotation\) 두개는 trade-off 관계지만 이 두가지 조건을 만족시키고 싶다.

GLIE monte-carlo control 입실론을 1/k로 하게 되면 입실론 greedy가 optimal policy에 수렴하는게 증명되어 있다.

MC 대신 TD를 쓸 수도 있지 않을까?

S에서 A를 하면 R를 받고 S'에서 A'라는 액션을 하고. 리워드 R을 가지고 바로 예측할 수 있다.

한 step마다 Q를 업데이트하고 바로 policy improvement 하고, 그 policy로 또 이동하고. \(반복

Sarsa는 수렴한다. 다음 두 조건 만족시켜야 한다.

* glie 해야 한다. 모든 state-action pair에 방문해야하고, greedy policy에 수렴해야 한다.
* step size가 충분히 커야 한다. Q value를 수정하는게 점점 작아진다. 

  -&gt; practicaly 이런 조건을 만족시키지 않아도 잘 수렴한다.

Windy Gridworld

S에서 G 까지 최적의 길찾기 문제. 아래의 숫자만큼 강제로 위로 그 숫자만큼 올라간다. 에피소드가 쌓일수록 점점 더 빨리 발전한다. Q는 state \* action 갯수의 array가 필요하겠찌.

n-step sarsa

forward view sarsa\(lamba\)

backward view sarsa\(lambda\)

sarsa\(lambda\) 원래 sarsa는 action을 하고나면 그 칸만 업데이트 했는데, 람다는 그 액션에 대해 모든 칸을 업데이트한다. 계산량은 많지만 정보전파는 빨라진다.

one-step 살사는 도착하기 전 값만 업데이트한다. 위로 가는 것이 좋다고 반영하는 거다. 살사람다는 도착한 순간, 도착한 과정들의 eligibility trace를 봐서 그 책임만큼 다같이 업데이트를 해준다.

## off policy

begavior policy 뮤 : action을 sampling 하는 policy target policy 파이 :

다른 agent의 행동을 보고 배울 수 있는거다.\(예를 들어 사람\) supervised랑은 다르다. 그 agent를 보고 따라하는게 아니라, 그걸 보고 어떻게 행동할 지를 배우는 거다.

on policy는 한번 경험하면 그 경험은 버렸다. 그 경험으로 policy를 업데이트 하고 나면 그 policy로 새로운 action을 해야 의미가 있기 때문에. off policy는 경험한 policy가 그대로 남아있기 때문에 그 경험을 reuse 할 수 있다. 탐험적으로 움직이면서 안전하게 최적의 폴리시를 배워나갈 수 있다.

off poicy를 가능하게 하는 방법 2가지. importance sampoilng과 q-learning

X분포는 확률분포P 에서 샘플링 되는 것. f\(x\)는 어떤 함수. 어떤 값이 들어갈 때 어떤것이 나오는 함수. P를 이용해서 Fx의 기댓값을 구하는 것을 통해, 다른 Q 라는 것을 이용해 Fx의 기댓값을 구하고 싶은 것.

P분포에서의 확률과 Q분포에서의 확률 , 그 비율을 곱해주는 것. 뮤와 파이의 비율들이 action을 할 때마다 계쏙 나올 거고 그 비율을 곱해준다. 교정을 해주는 것. action 수만큼 교정 term이 있는거다.

근데 이 방법은 쓸 수 없다. term의 variance가 너무 커서 현실에서 쓸 수가 없다. 그래서 MC 대신 TD를 사용한다.

actoin 하나에 대해 target policy에서 behavior policy를 나눈 비율을 곱해준다. MC보다 variacne가 훨씬 적어서 괜찮다.

Q-learning

importance sampling 안쓰고 Q를 off pilicy를 훈련시키고 싶다. behavior policy에서 액션을 하나 뽑아서 behaivor policy대신 target policy를 쓴다. target policy로 action을 뽑고 그 value를 쓴다. 추측이니까 이렇게 해도 됨. 다음 state에 할 추측치 값을 사용해 Q를 update 해줘야 하는데, 그 추측치 값을 behavioir policy를 쓰지 않고 target policy를 쓰는거다. 이동은 behavior policyt로 해놓고.

off-policy control with q -learning

behavioir policy도 정해야한다. target policy처럼 imporevet 됏음 좋겠다. target policy는 greedy, behavior greedy는 입실로 greedy 를 사용한다. Q러닝이 이렇게 많이 쓴다. 행동은 입실론하기 하니까 다양하게 경험을 쌓고 실제 target policy는 greedy하게 선택하는 거다.

sarsa는 그 policy에서 하나의 action만 선택하는 거였는데 q learning에서는 action 중에서 max 값을 선택해 업데이트한다. sarsa max라고도 불린다.

