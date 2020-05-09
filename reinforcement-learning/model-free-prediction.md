# Model Free Prediction

## 

## Model Free Prediction

MDP를 모르는데 agent가 환경에 던져졌을 때 어떤 방식으로 prediction 하고 control 할 지 environment를 모르고 policy가 정해져있을 때 value를 찾는 문제. 즉 끝날 때 return을 알아내는 문제

monte carlo, td learning을 통해 문제를 풀거다.

### Introduction

### Monte carlo LEarning

MC : 직저 ㅂ구하기 어려운 것을 계속 시행하면서 나오는 값들을 통해 추정하는 것. 이 state에서 value의 기댓값이 어떻게 되나. 끝까지 했을 때 나오는 return 값듫이 다를텐데 그 return 값들로 평균 내는 것. 경험으로부터 직접 배운다. MDP 의 transition, rewarㅇ 몰라도 policy 대로 해보면서 value function을 알아가는 것. episode가 끝나야 return 이 나온다. episode 마다 다른 return을 얻는다. 모든 episode가 끝나야 MC 를 적용할 수 있다.

policy evaluation == prediction episode들을 다 해서 return갑승리 나오는데 그거의 기댓값을 사용

every visit : 해당 state에 방문할 때마다 count 해주는 것 first visit : 해당 state의 첫 방문때만 count 해주고 그 이후는 무시 episode에서 처음 방문할 때만 count 해주고 return에 더해준다.

뭘 써도 상관없음! 결국 같은 값이 나옴. 수렴하게 되니까. 모든 state를 다 방문해야한다는 것이 가정되어야 함. 그래야만 수렴함. 모든 state를 evaluate 하는게 목적이기 때문에, 그 족에 가지 않으면 그 state는 평가하지 못함.

v\(s\) = 리턴의 합 / 방문한횟수

#### Incremental Mean

MC는 여러개 해봐서 평균 내는 것. 그걸 좀 다르게 나타내면 Incremnetal Mean 이 전 k까지의 평균과 이번 state로 나눈것. episode 마다 state 값들을 다 저장해둬야 했다. 나중에 평균내려면. 이 방법을 사용하면ㄴ 새로운 값이 나올 때마다 교정해주면 된다. incremental 하게 MC를 업데이트 하는 것.

Gt-V\(st\) : 에러. 에러만큼 더해주는 것. N분의 1을 작은 값으로 고정시킬 수 있음. 오래된 기억은 잊어버리게 된다. non-stationary 문제에서 좋을 수 있다. 과거는 잊고 최신의 것만 기억하고 싶을 때 사용.

### Temporal-Difference LEarning

경험으로부터 직접적으로 배운다는건 같음. TD는 episode가 끝나지 않아도 배울수 있다. guess로 guess로 업데이트 하는 것 MC는 G방향으로 V를 업데이트 하는 것. TD는 R~~ \(TD 타겟\) 방향으로 v를 업데이트 하는 것 한 스텝을 가보고 거기서 예측하는 예측치를 보고 그방향으로 v 를 업데이트 이전에 예측한 것보다 더 정확할거아냐? 현실이 더 반영되어 있으니까? MC는 정확한 값으로 업데이트하는 건데 TD는 예측치로 예측하니까 에러가 더 있지 않을까?

### TD

final 에피소드가 나오기 전에 계산ㄱ ㅏ능. 끝나지 않는 non-terminating 환경에서 사용될 수 있다. MD - 에피소드가 다 끝나고 나온 return 값으로 계산 해야한다.

unbiased estimate . 편향되지 않았다. G\_t를 계쏙 평균내다보면 v\_pi로 결국 수렴하게 된다. 그래서 unbiased 한 estimate이 가능하다. True TD target. oracle이 v\_pi \(s\_t\)의 실제 값을 알려주게되면 unbiased 한 estiamte이 된다. bellman equation을 저 값을 보장해주기 때문ㅇ. 하지만 현재 추측치로 업데잍으 하기 때문에 biased 되어 있을 수 있다.

variance가 크고 bias 가 작음 - MC variance가 작고 bias가 크다 - TD / return 은 환경의 랜덤성을 제공

랜덤성과 랜덤성을 가지고 게임을 하면 variacne가 크다 TD는 한스텝한스텝마다 return이 나오기 때문에 랜덤성이 작아서 bias가 높고 variance가 작다.

MC : function approximation \(deep learning\) 을 써도 수렴을 잘 한다. 실제값을 업데이트하기 땜누에 initial value가 중요하지 않다.

TD : 보통 MC보다 효과적이지만.. initial value가 민감함. 추측치가 처음에 잘 정해져있어야 잘 수렴을 하기 때문.

### Random Walk Example

### Batch MC and TD

무한번 뽑아내면 MC나 TD나 v파이에 수렴하게 될거다. k개의 제한된 에피소드가 있을 때 TD가 잘 수렴할까? MC나 TD나 같은 값에 수렴을 할까?

AB Example MC로 하면 A는 0. TD로 하면 A는 0.75. V\(B\)로 A를 업데이트 하기 때문.

-&gt; TD는 Markov Property를 이용해서 value를 추측. markov env에서 더 효과적 MC는 그냥 mean squred error를 minimize 하는 것.

MC는 끝까지 가보고 St를 업데이트. \(monter carlo backup\) TD는 한스텝만 가고 추측해서 그 값으로 대체 \(bootstraping\) DP는 샘플링을 하지 않고 할 수 있는 모든 action에 대해 업데이트. full로 하고 끝까지 안간다.

Bootstraping은 추측치로 업데이트 하는 거라 예측치에 추측치가 포함된다. \(MC X\) - depth 관점으로 본거 sampling은 full 스윕을 안하고 샘플로 업데이트하는거. \(TD, DP\) - width 관점으로 본거

모델을 알 때는 DP가 가능하지만, 모델을 모를 때에는 sample backup을 해야 한다. \(TD, MC\) sampling은 어떻게 하느냐? agent가 policy를 따라 가는 것이 샘플링

--

TD의 변형들 TD를 몇번 후 구하는 거. MC와 TD 사이에 스펙트럼이 있는거

n만큼은 리워드를 넣고 그 이후는 추측치를 넣고. n의 값은 잘 찾아봐야함.

평균낸 걸 써도 됨 \(Average~\)

TD\(0\)~MC까지 모든걸 평균내서 써도됨 \(TD 람다\) 그냥 평균이 아니라 geometric mean. MC로 갈 수록 가중치가 적게 들어간다. Forward-view TD : 미래를 보는거니가. geometric mean 사용하는 이유는? computational efficient를 위해 TD\(0\) 같은 비용으로 td 람다를 계싼할 수 있다

TD 무한을 알아야 하기 때문에 에피소드가 끝나야 할 수 있다. TD\(0\)의 장점이 사라짐

backward view TD 람다

Eligibility Traces 책임이 큰 애한테 업데이트를 많이 해주는거.

* frquency : 많이 일어난 애한테 책임을 주는거
* recentcy : 가장 최근에 일어난 애한테 책임을 주는거

  eligibility trace를 곱해서 그 만큼만 업데이트 해주는거 td람다와 수학적으로 동일한 효과를 주는거 \(ㅠㅠ?

