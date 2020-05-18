# Model Free Prediction

## Introduction

MDP를 모르는 환경에 agent가 던져졌을 때 어떤 방식으로 prediction 하고 control 해야할까? 이것이 environment를 모르고 policy가 정해져있을 때 value를 찾는 문제이다. 즉, episode가 끝날 때 return 값을 알아내야 한다. 

이런 경우 경험으로부터 직접 배워가야 한다. Monte Carlo Learning, TD Learning 을 통해 위와 같은 문제를 풀려고 한다. value function을 추정하고 optimal policy를 찾는 방법들이다. 

DP처럼 모든 state transition probability를 알 필요 없다. DP는 MDP에 대한 정보로부터 value function을 계산\(compute\)했지만, 여기에서는 MDP의 return 표본들을 통해 value function을 학습\(learn\)한다.



## Monte-Carlo Learning

Monte-Carlo는 random하게 무언가를 취해보고 추정하는 방법이다. 

> MC는 sample return의 평균값을 통해 RL 문제를 푸는 방법이다. 경험이 여러개의 episode로 분할되고 언젠가는 종료된다고 가정했을 때, 하나의 episode가 완료된 후에만 value estimation과 policy가 변화한다.

직접 구하기 어려운 값들을 계속 시행하면서 나오는 값들을 통해 추정하는 방법이다. 어떤 state 에서 value의 기댓값이 어떻게 되는지 측정한다. 매 episode가 끝날 때 나오는 return 값들이 다를텐데 그 return 값들을 평균내어 사용한다. 

> 어떤 state의 value는 그 state를 시작점으로해서 계산된 expected return 값\(미래 discounted reward의 누적 기댓값\)이다. 그렇기 때문에 경험으로부터 state의 value를 추정하는 방법은, 그 state 이후에 방문한 모든 state의 return 값들의 평균을 구하는 것이다. 얻어지는 return 값이 많아질 수록 그 평균값은 기댓값 수렴하게 된다. \(MC의 기본 idea\)

* episode의 경험으로부터 직접적으로 학습한다
* MDP transition과 reward를 모르는 Model-free 방식이다
* episode가 끝나야 return 값이 나오고, 학습할 수 있다 \(no bootstrapping\)
* value = mean return
* episodic MDP에만 MC를 적용할 수 있다 \(모든 episode는 terminate 되어야 한다\)

### Monte-Carlo Policy Evaluation

{% hint style="info" %}
**\[Goal\]** policy가 $$ \pi $$일 때, episode의 경험로부터 $$ v_{\pi} $$를 학습시킨

$$
S_1, A_1, R_2, ..., S_k \sim \pi
$$
{% endhint %}

Policy Evaluation 이니 Prediction 문제이다.

Monte-Carlo policy evaluation은 expected return 값 대신에 empirical mean return 값을 이용한다.  episode를 끝까지 가본 후에 얻어지는 return 값들로 각 state의 value function을 거꾸로 계산해보는 방식이기 때문에 episode가 끝나지 않으면 사용할 수 없는 방법이다.

initial state 부터 terminal state 까지 policy $$ \pi $$를 따라서 가다보면 time step 마다 discounted reward를 받을텐데, 그 reward 값을 기억해뒀다가 $$ S_t $$가 되면 뒤돌아보며 각 state의 value function을 계산한다. 

Recall the `return` \(total discounted reward\)

 $$ G_t = R_{t+1} + \gamma R_{r+2} + ... + \gamma^{T-1} R_T $$

Recall the `value function` \(expected return\)

 $$ v_{\pi}(s) = \mathbb{E}_{\pi} [G_t | S_t = s] $$



#### First-Visit / Every-Visit

한 episode에서 state $$s$$를 방문했을 때 그것을 `visit` 이라고 한다. $$ s $$의 visit 이후에 발생하는 return 의 평균을 구함으로써 $$ v_{\pi}(s) $$를 추정한다.

`First-visit MC method` : 해당 state의 첫 방문때에만 count 해주고, 그 이후는 무시하고 return값 에 더해주기만 한다.

`Every-visit MC Method` : 해당 state에 방문할 때마다 count 해준다.

1. 방문한 횟수 증가 

   $$ N(s) \leftarrow N(s) + 1 $$

2. total return 값을 더해줌

   $$ S(s) \leftarrow S(s) + G_t $$

3. return 의 평균값 \(total return 의 합 / 방문한 횟수\) 으로 value를 estimate 한다

   $$ V(s) = S(s) / N(s) $$

4. 방문 횟수가 무한대로 갈 수록 return 평균값은  value 값에 수렴한다.$$ V(s) \rightarrow v_{\pi}(s)  \quad  as  \quad N(s) \rightarrow \infty $$

First-visit, every-visit 둘 중 무엇을 써도 상관은 없다. visit의 갯수\($$ N(s) $$\) 가 무한으로 갈 수록 수렴하게 된다. 다만 모든 state를 다 방문해야 한다는 것이 가정되어야 한다. 모든 state를 evaluate 하는 것이 목적이기 때문에, 어떤 state에 방문하지 않으면 해당 state는 평가하지 못한다.

### Incremental Monte-Carlo Updates

![Incremental Mean](../.gitbook/assets/image%20%28241%29.png)

MC는 여러개 시도해보고 episode가 끝난 후 그것을 평균내는 방식이다. 나중에 평균내기 위해서는 episode마다 이전의 state 값들을 전부 저장해놓고 값을 계산해줘야 하지만, Incremental Mean을 사용하면 그때그때 평균을 구하며 새로운 값이 있을 경우 그 값을 통해 교정해주기 때문에 이전에 계산된 모든 값들을 저장할 필요가 없다. Incremental 하게 평균값을 구해주는 것이다.

$$ x_1, x_2, .. $$의 연속된 입력값이 주어지고 이들의 평균 $$ \mu_1, \mu_2, ..$$을 구할 때 위와 같이 전개할 수 있다.

$$
\mu_k = \mu_{k-1} + \frac{1}{k}(x_k - \mu_{k-1})
$$

이를 이용해 MC에서 적용해보자. state $$ S_t $$ 에서 return 값이 $$G_t$$라고 하면,

$$
N(S_t) \leftarrow N(S_t) + 1 \\
V(S_t) \leftarrow V(S_t) +  \frac{1}{N(S_t)} G_t - V(S_t)
$$

이와 같이 나타낼 수 있다. $$ G_t - V(S_t) $$는 $$ s $$에서의 error term으로, error 만큼 더해주는 것으로 볼 수 있다. 방문하는 state 의 수가 증가함에 따라 $$ \frac{1}{N(S_t)} $$는 점점 작아지게 될텐데, 이 값을 작은 값 $$ \alpha $$로 고정할 수 있다. 그럼 오래된 기억은 잊게 된다.

$$
V(S_t) \leftarrow V(S_t) + \alpha(G_t = V(S_t))
$$

non-stationary 문제에서 과거의 기억은 잊고 최신의 경험만 기억하고 싶을 때 사용한다.

### Temporal-Difference Learning

TD는 MC 와 DP 방법을 결합한 것이다. MC 처럼 경험으로부터 직접적으로 학습하고, DP 처럼 episode가 끝나지 않더라도 학습할 수 있다. 

DP, MC, TD 모두 GPI \(Generalized policy iteration\)를 일부 변형한 것으로, prediction 문제에 대한 접근법의 차이다. 

* episode의 경험으로부터 직접적으로 학습한다
* MDP transition과 reward를 모르는 Model-free 방식이다
* episode가 끝나지 않더라도 학습한다 \(bootstrapping\)
* 예측을 통해 예측값을 업데이트 한다

{% hint style="info" %}
**\[Goal\]** policy가 $$ \pi $$일 때, episode의 경험로부터 online $$ v_{\pi} $$를 학습시킨다
{% endhint %}

Monte-Carlo 에서는 actual return인 $$G_t$$ 방향으로 $$V(S_t)$$를 업데이트 했다.

$$ V(S_t) \leftarrow V(S_t) + \alpha ({\color{RED} G_t} - V(S_t)) $$

TD에서는 \(TD\(0\)이라고 가정\) estimated return인 $$R_{t+1} + \gamma V(S_{t+1})$$방향으로 $$V(S_t)$$를 업데이트 한다. \(estimated return을 TD target이라고 부른다\) 즉, 현재 value function을 계산하는데 이전의 주변 state들의 value function을 사용하는 것이.

$$V(S_t) \leftarrow V(S_t) + \alpha( {\color{RED} R_{t+1} + \gamma V(S_{t+1})} - V(S_t))$$

* `TD target` : $$R_{t+1} + \gamma V(S_{t+1})$$
* `TD error` : $$ R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$







한 스텝을 가보고 거기서 예측하는 예측치를 보고 그방향으로 v 를 업데이트 이전에 예측한 것보다 더 정확할거아냐? 현실이 더 반영되어 있으니까? MC는 정확한 값으로 업데이트하는 건데 TD는 예측치로 예측하니까 에러가 더 있지 않을까?

### TD vs MC

<table>
  <thead>
    <tr>
      <th style="text-align:center"><b>TD</b>
      </th>
      <th style="text-align:center"><b>MC</b>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">
        <ul>
          <li>&#xB9E4; step &#xB9C8;&#xB2E4; online &#xD559;&#xC2B5;&#xD560; &#xC218;
            &#xC788;&#xB2E4;.</li>
          <li>incomplete sequences &#xC5D0;&#xC11C; &#xD559;&#xC2B5;&#xD560; &#xC218;
            &#xC788;&#xB2E4;.</li>
          <li>continuing (non-terminating) &#xD658;&#xACBD;&#xC5D0;&#xC11C; &#xC0AC;&#xC6A9;
            &#xAC00;&#xB2A5;&#xD558;&#xB2E4;.</li>
        </ul>
      </td>
      <td style="text-align:center">
        <ul>
          <li>episode&#xAC00; &#xB05D;&#xB098;&#xACE0; return &#xAC12;&#xC774; &#xB098;&#xC640;&#xC57C;
            &#xACC4;&#xC0B0;&#xD560; &#xC218; &#xC788;&#xB2E4;.</li>
          <li>complete sequences &#xC5D0;&#xC11C; &#xD559;&#xC2B5;&#xD560; &#xC218;
            &#xC788;&#xB2E4;.</li>
          <li>episodic (terminatinv) &#xD658;&#xACBD;&#xC5D0;&#xC11C;&#xB9CC; &#xC0AC;&#xC6A9;
            &#xAC00;&#xB2A5;&#xD558;&#xB2E4;.</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>* Return $$ G_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma ^ {T-1}R_t $$는 $$v_{\pi}(S_t)$$의 unbiased estimate 이다. 즉 편향되지 않았다는 것이다. $$G_t$$를 계속 평균내다 보면 $$v_{\pi}$$로 결국 수렴하게 되기 때문에 unbiased한 estimate 이 가능하다.
* True TD target $$R_{t+1} + \gamma v_{\pi}(S_{t+1})$$는 $$v_{\pi}(S_t)$$ 의 unbiased estimate 이다. 모든 것을 알고 있는 oracle이 $$v_{\pi}(S_t)$$의 실제값을 알려주게 되면 bellman equation이 저 값을 보장해주기 때문에 unbiased한 estimate 이 된다. 
* TD target $$R_{t+1} + \gamma v_{\pi}(S_{t+1})$$는 $$v_{\pi}(S_t)$$ 의  biased estimate 이다. 추측치로 업데이트하기 때문에 biased 되어 있을 수 있다.
* TD target은 return 보다 variance가 낮다. Return은 많은 random actions, transitions, rewards에 종속되지만, TD target은 하나의 random action, transition, reward에 종속되기 때문이다.

![](../.gitbook/assets/image%20%28103%29.png)

<table>
  <thead>
    <tr>
      <th style="text-align:center"><b>TD</b>
      </th>
      <th style="text-align:center"><b>MC</b>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">
        <ul>
          <li>variance&#xAC00; &#xC791;&#xACE0; bias&#xAC00; &#xD06C;</li>
          <li>&#xC77C;&#xBC18;&#xC801;&#xC73C;&#xB85C; MC &#xBCF4;&#xB2E4; &#xD6A8;&#xC728;&#xC801;&#xC774;&#xB2E4;.</li>
          <li>TD(0)&#xC740; &#xC5D0; &#xC218;&#xB834;&#xD55C;&#xB2E4;. &#xD558;&#xC9C0;&#xB9CC;
            function approximation&#xC744; &#xC37C;&#xC744; &#xB54C; &#xD56D;&#xC0C1;
            &#xC218;&#xB834;&#xD558;&#xB294; &#xAC83;&#xC740; &#xC544;&#xB2C8;&#xB2E4;.</li>
          <li>initial value&#xC5D0; &#xBBFC;&#xAC10;&#xD558;&#xB2E4;. &#xCD94;&#xCE21;&#xCE58;&#xAC00;
            &#xCC98;&#xC74C;&#xC5D0; &#xC798; &#xC815;&#xD574;&#xC838;&#xC788;&#xC5B4;&#xC57C;
            &#xC798; &#xC218;&#xB834;&#xC744; &#xD55C;&#xB2E4;.</li>
        </ul>
      </td>
      <td style="text-align:center">
        <ul>
          <li>variance&#xAC00; &#xD06C;&#xACE0; bias&#xAC00; &#xC791;&#xB2E4;.</li>
          <li>function approximation (deep learning)&#xC744; &#xC0AC;&#xC6A9;&#xD574;&#xB3C4;
            &#xC218;&#xB834;&#xC774; &#xC798; &#xB41C;&#xB2E4;.</li>
          <li>&#xC2E4;&#xC81C;&#xAC12;&#xC744; &#xC5C5;&#xB370;&#xC774;&#xD2B8; &#xD558;&#xAE30;
            &#xB54C;&#xBB38;&#xC5D0; initial value&#xAC00; &#xC911;&#xC694;&#xD558;&#xC9C0;
            &#xC54A;&#xB2E4;.</li>
          <li>&#xC2EC;&#xD50C;&#xD558;&#xAC8C; &#xC774;&#xD574;&#xD558;&#xACE0; &#xC0AC;&#xC6A9;&#xD558;&#xAE30;
            &#xC88B;&#xB2E4;.</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>> Bias : 데이터 내의 모든 정보를 고려하지 않아, 지속적으로 잘못된 것들을 학습하는 경향 \(underfitting에 연관있음\)
>
> Variance : 데이터 내의 에러나 노이즈까지 잘 잡아내는 model에 데이터를 fitting 시킴으로, 실제 현상과 관계없는 random한 것들까지 학습하는 알고리즘의 경향 \(overfitting에 연관있음\)
>
> 보통 이 두개의 값은 trade-off 관계에 있다.

Return은 환경의 랜덤성을 제공하는 것으로 볼 수 있다. 랜덤성과 랜덤성을 가지고 게임을 하면 variance가 크다. TD는 매 스텝마다 return이 나오기 때문에 랜덤성이 작아서 bias가 높고 variance가 작다.

TD는 한 episode 안에서 매 time-step 마다 업데이트를 하는데 보통 그 이전의 상태가 그 후의 상태에 영향을 많이 주기 때문에 학습이 한 쪽으로 치우칠 수 있다. \(bias가 높다\)  MC는 episode마다 학습하기 때문에 episode 가 전개됨에 따라 전혀 다른 경험을 가질 수가 있다. 하나의 state에서 다음 state로 넘어갈 때도 확률적으로 움직이기 때문에 random 성이 크다. \(variance가 높다\)

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

