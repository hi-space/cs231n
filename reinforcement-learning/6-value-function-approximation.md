# 6. Value Function Approximation

## Introduction

Value function은 state의 갯수만큼, Q 에서는 state-action fair의 갯수만큼 table 공간이 필요했다. 큰 MDP 문제에서는 메모리에 저장하기에 너무 많은 state와 action들이 존재했다. 그래서 value function을 function approximation을 해서 estimate 하는 방식을 사용하려고 한다.

$$
\hat{v}(s, w) \approx v_{\pi}(s) \\
\hat{q}(s, a, w) \approx q_{\pi}(s, a)
$$

* $$ v_{\pi}(s)$$ : 실제 value function
* $$\hat{v}(s, w)$$ : 실제 value 값을 모방하는 approximation function
* $$w$$: $$\hat{v}$$함수 안에 포함되어 있는 parameter 값
* $$\approx$$ : approximation

function approximation을 하게 되면 봤던 state 뿐만 아니라 보지 못한 state에 대해서도 generalize가 잘된다. MC나 TD learning을 통해 파라미터 $$w$$를 업데이트하며 학습해가는 방식이다.

![Types of Value Function Approximation](../.gitbook/assets/image%20%28437%29.png)

일반적인 함수의 모양을 나타내는 black box 이다. 어떤 input 값을 받았을 때 internal parameter인 $$w$$와 함께 연산되어 그에 해당하는 output을 산출한다.

1 그림\)  value function으로, $$s$$를 input으로 넣으면 $$w$$값들을 통해 $$\hat{v}(s, w)$$의 output이 나온다.

2, 3 그림\) Q 는 두가지 형태로 함수를 만들 수 있다.

* \(action in 형태\) $$s$$와 $$a$$를 input으로 넣으면 $$\hat{q}(s, a, w)$$이 output으로 나온다.
* \(action out 형태\) $$s$$를 input으로 넣으면 $$s$$에서 할 수 있는 모든 action에 대해서 output이 나온다.

![Which Function Approximator?](../.gitbook/assets/image%20%28438%29.png)

function approximation으로 사용할 수 있는 함수는 liner combinations, neural network, decision tree 등 여러가지가 될 수 있겠지만, 그 중에서도 미분가능한\(differentiable\) 함수를 사용할거다. 그래야만 그 상태의 gradient를 구해서 update 할 수 있기 때문이다. 

그래서 위의 함수들 중 Linear combinations of features와 Neural Network 를 차례로 알아볼거다.

> * **stationary**
>
>     $$ F(x_1, x_2, x_3) = F(x_1 + t, x_2 +t, x_3 + t)$$
>
> stationary에 대한 정의를 수식으로 나타내면 위와 같다. 어떤 값이 시간에 따라 바뀌더라도 그 확률분포는 변하지 않는다. 
>
> -&gt; non-stationary : 시간에 따라 랜덤변수의 성질이 변한다.
>
> * **iid**
>
> 랜덤변수가 같은 특징을 갖고있고\(identical\), 서로 독립 \(independent\) 이다. 예를 들면 동전던지기로 볼 수 있다. 앞이나 뒤가 나올 확률은 항상 $$\frac{1}{2}$$로 랜덤변수가 동일하지만, 던질때 마다 그 확률은 서로 독립이기 때문에 전에 던진 값이 이후에 던지는 확률에 영향을 주지 않는다.
>
> -&gt; non-iid : identical 이 아닐 수도 있고 independent가 아닐수도 있다.

non-stationary, non-iid data는 확분포가 시간에 따라 계속 바뀌는 random성을 가지고 있고 이전의 값이 이후의 값에 영향을 미칠 수 있는 데이터들을 말한다. 즉, 특징이 없는 이러한 데이터들에서도 general 하게 동작할 수 있는 training method를 찾고자 하는거다. 

## Incremental Methods

### Gradient Descent

![Gradient Descent](../.gitbook/assets/image%20%28434%29.png)

* $$J(w)$$ : $$w$$값이 input으로 들어가면 output이 나오는 어떤 함수
* $$w = [w_1, w_2, ..., w_n]$$ : n 차원 vector. 
* $$\alpha$$  : step size

$$J(w)$$ 라는 함수를 최소화하는 input값인 $$w$$ 값을 찾고 싶을 때 사용하는 것이 gradient descent 방법이다. $$J(w)$$를 $$w$$에 대해서 미분하면 gradient 값을 알 수 있게 된다. $$w$$가 vector 이니 결과값은 어떤 방향값이 나오게 되는데, 그 방향으로 $$\alpha$$값만큼 조금씩 움직인다. 가장 가파른 방향으로 조금씩 움직여 local minimum을 찾는 방법이다.

$$
\nabla_wJ(w) = 
\begin {pmatrix}
\frac {\partial J(w)} {\partial w_1} \\ \\
\vdots \\ \\
\frac {\partial J(w)} {\partial w_n} 
\end {pmatrix}
\\
\Delta w = -\frac{1}{2} \alpha \nabla_wJ(w)
$$

* $$\nabla_wJ(w)$$  :  $$J(w)$$ 의 gradient
* $$\Delta w$$            :  $$J(w)$$ 의 local minimum을 구하기 위해 업데이트 해야하는 $$w$$

### Value Function Approximation \(Stochastic Gradient Descent\)

Value Function을 잘 학습하기 위해서는 approximate value function인 $$\hat{v}(s, w)$$가 true value function인 $$v_{\pi}(s)$$와의 차이가 작게 나게하는 파라미터 vector $$w$$를 찾아야 한다.

#### Objective Function

$$
J(w) = \mathbb{E}[(v_{\pi}(S) - \hat{v}(S, w))^2]
$$

그래서 목적함수 $$J(w)$$를 위와 같이 정의할 수 있다. true value function 과 approximate value function의 오차를 최대한 작게 하는 것\(MSE\)을 목표로 한다. \(loss값이라고 볼 수 있음\) 이 목적함수를 가지고 $$J(w)$$를 최소화하는 방향으로 $$w$$를 업데이트 해야 한다.

#### Update Parameters

$$
\Delta w
= \alpha \mathbb{E}_{\pi}[(v_{\pi}(S) - \hat{v}(S, w)) \nabla_w \hat{v}(S, w)]
$$

위의 수식은 $$\Delta w = -\frac{1}{2} \alpha \nabla_wJ(w) $$의 $$\nabla_wJ(w)$$ 자리에 목적함수를 두고 전개하여 나타낸 것이다. gradient descent를 이용해 local minimum을 찾기 위해 저 값만큼 update 해주면 된다.

$$
\Delta w = \alpha (v_{\pi}(S) - \hat{v}(S, w)) \nabla_w \hat{v}(S, w)
$$

Stochastic Gradient Descent는 Gradient의 sample을 뽑아서 사용하는거다. 방문했었던 state들을 input값으로 넣어준다. 이 과정을 여러번 반복하게 되면 expected update와 full gradient update와 동일하게 되는 것이 증명되어 있다.

### Linear Function Approximation

이번엔 Approximation을 위한 함수로 Linear Function을 사용하는 경우에 대한 이야기이다.

#### Feature Vector

$$
x(S) =   \begin{pmatrix}
x_1(S)
\\ 
\vdots
\\ 
x_n(S)
\end{pmatrix}
$$

어떤 state $$S$$가 있다고 하면 그 state에 따른 n 개의 feature가 있을 수 있고, 그 feature vector 위와 같이 표현할 수 있다. 여기서 말하는  feature를 예를 들면 로봇의 각 관절마다의 토크값으로 볼 수 있다.

#### Approximate Value Function

$$
\hat{v}(S, w) = x(S)^\intercal w = \sum_{j=1}^{n} x_j(S)w_j
$$

feature vector를 transpose 해서 $$w$$ 값을 내적곱 해주면 linear combination of features의 value function이 나온다. \(transpose 해주는 이유는 $$w$$과 연산해주기 위함\)

#### Objective Function

$$
J(w) = \mathbb{E}_{\pi} [ (v_{\pi}(S) - x(S)^\intercal  w)^2]
$$

Objective Function은 마찬가지로 true value function과 approximate value function 간의 MSE 의 기댓값으로 둘 수 있다. 

함수의 모양이 Linear 하기 때문에 minimum 값이 하나 밖에 없어서 Stochastic Gradient Descent는 global optimum으로 수렴한다. 

#### Update Parameters 

$$
\nabla_w \hat{v}(S, w) = x(S) \\
\Delta w = \alpha( v_{\pi}(S) - \hat{v}(S, w)) x(S)
$$

* $$\Delta w$$ : update
* $$\alpha$$ : step-size
* $$v_{\pi}(S) - \hat{v}(S, w)$$: prediction error
* $$x(S)$$ : feature value

#### Table Lookup Feature

![Table Lookup Features](../.gitbook/assets/image%20%28433%29.png)

기존에 배웠던 table lookup 도 linear value function의 하나의 예시라고 볼 수 있다. state의 값들을 하나의 feature로 보고 vector로 표현해 feature vector 처럼 만들 수 있다. 그리고 n개의 $$w$$를 내적곱 해서 approximate value function을 표현할 수 있다. \($$w$$는 feature의 갯수만큼 존재\)

### Incremental Prediction Algorithm

지금까지 true value function과 approximate value function의 차이를 줄이는 것을 목표로 해서 w 값을 update 하는 방법을 봤다. 하지만 실제 문제에서는 supervisor가 없기 때문에 true value function은 알 수가 없고 immediate reward 만 볼 수 있다. 그래서 이 true value function 대신 이전에 배운 MC나 TD를 사용하려고 한다.

![target for value function](../.gitbook/assets/image%20%28435%29.png)

true value function 대신 MC나 TD를 쓴다는 것은 결국 cumulative reward에 대해 예측을 어떻게 하는지 알아내는 prediction 문제이다. 

#### MC

value 는 return 의 기댓값니다. sampling 할 때마다 다른 episode가 나오게 되고 그 episode에 나오는 return 값들도 상이하다. 이 데이터들을 training data로 사용해도 된다.

* $$G_t$$는 true value $$v_{\pi}(S_t)$$의 noisy 하고 unbiased 한 sample이다.
* 이 sample 데이터들을 training data로 사용할 수 있다. $$ <S_1, G_1>, <S_2, G_2>, ... , <S_t, G_t>$$
* Monte-Carlo Evaluation은 local optimum에 수렴한다. \(non-linear value function approximation을 사용하더라도\)

#### TD

* TD- target인 $$R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$$ 는 true value $$v_{\pi}(S_t)$$의 biased sample 이다.
* 이 sample 데이터들을 training data로 사용할 수 있다. $$<S_1, R_2 + \gamma \hat{v}(S_2, w)>, <S_2, R_3 + \gamma \hat{v}(S_3, w)>, ... , <S_{T-1}, R_T>$$
* Linear TD\(0\) 은 global optimum에 수렴한다.

MC는 unbiased 하기 때문에 variance가 크더라도 정답값에 비슷하게 맞출 수 있다. 반면에 TD는 variance가 작고 bias가 크기 때문에 global optimum에 수렴해갈 수 있다.

#### TD\($$\lambda$$\)

* $$\lambda$$-return인 $$G_t^{\lambda}$$는 true value $$v_{\pi}(S_t)$$의 biased sample 이다.
* 이 sample 데이터들을 training data로 사용할 수 있다. $$ <S_1, G_1^{\lambda}>, <S_2, G_2^{\lambda}>, ... , <S_{T-1}, G_{T-1}^{\lambda}>$$

### Action-Value Function Approximation

![Control with Value Function Approximation](../.gitbook/assets/image%20%28436%29.png)

Model Free가 되려면 Value Function 대신 Action Value Function을 사용해야 한다. Action-Value Function Approximation도 비슷한 형식으로 이뤄진다. Evaluation은 parameter $$w$$를 update 해가고, Improvement 는 그렇게 update된 action-value function에서 $$\epsilon-greedy$$로 action을 취하며 improve 된다.

#### action-value function approximation

$$
\hat{q}(S, A, w) \approx q_{\pi}(S, A)
$$

Q function에서 parameter $$w$$를 포함시켜 approximation 함수를 정의한다.

#### Objective Function

$$
J(w) = \mathbb{E}_{\pi}[(q_{\pi}(S, A) - \hat{q}(S, A, w))^2]
$$

true action-value function과 approximate action-value function의 MSE 를 최소화한다.

#### Stochastic Gradient Descent

$$
-\frac{1}{2}\nabla_wJ(w) = (q_{\pi}(S, A) - \hat{q}(S, A, w)) \nabla_w \hat{q}(S, A, w)
\\
\Delta w = \alpha(q_{\pi}(S, A) - \hat{q}(S, A, w)) \nabla_w \hat{q}(S, A, w)
$$

Stochastic Gradient Descent를 이용해 local minimum을 찾는다.

#### MC & TD

![](../.gitbook/assets/image%20%28439%29.png)

마찬가지로 true action-value function은 MC와 TD로 대체될 수 있다.

### Convergences

![Convergence of Prediction Algorithms](../.gitbook/assets/image%20%28441%29.png)

Off-Policy에서는 MC에서만 수렴한다고 되어 있는데 실제로는 TD\(0\)이나 TD\($$\lambda$$\)나 잘 수렴한다.

## Batch Methods

Incremental Methods는 sample을 하나 뽑아서 Stochastic Gradient Descent를 이용해 parameter를 조금씩 update 하고, 또 그것으로 policy update 까지 한다. 뽑은 sample로 policy 까지 update 하기 때문에 그 이후에는 그 sample이 버려지게 되어, 효과적으로 사용되지 않는다. 

Batch Methods는 sample 데이터\(경험\)들을 쌓아놓고 있다가 re-use 하면서 학습을 진행한다. SGD 처럼 차근차근 학습을 진행하지않고 training data를 모아놨다가 한꺼번에 update 하는 방식이다.

$$D = \{<s_1, v_1^{\pi}>, <s_2, v_2^{\pi}>, ..., <s_T, v_T^{\pi}> \}$$

value function approximation $$\hat{v}(s, w) \approx v_{\pi}(s)$$ 가 주어졌을 때 $$D$$는 &lt;state, value&gt; pair 이다. value function approximation은 경험들의 집합인 $$D$$를 이용해 parameter를 update 한다. Incremental Methods는 $$\pi$$를 따라가며 parameter를 update 했지만 Batch Methods는 $$D$$를 따라가며 update 한다. 

value function approximation$$v_t^{\pi}$$와 target value인 $$v_t^{\pi}$$사이의 sum-squared error를 minimize 하는 parameter vector를 찾기 위해서 Least Squares 알고리즘을 사한다. 

$$
\begin {matrix}
LS(w) 
&=& \sum_{t=1}^{T}(v_t^{\pi} - \hat{v}(s_t, w))^2
\\ \\
&=& \mathbb{E}_D[(v^{\pi} - \hat{v}(s, w))^2]


\end{matrix}
$$

### Experience Replay

Experience Reply는 transition 들을 replay memory에 쌓고 랜덤하게 mini-batch를 뽑아서, 그것을 training data로 사용해 학습을 하는 방법이다. Experience Replay를 이용하면 sample efficient 하다는 장점도 있지만, 데이터들 사이의 correlation 으로 인해 학습이 잘 안되는 문제도 해결 할 수 있다.

\(1\)$$D$$에서 state와 value를 sampling 하고 경험이 쌓이고나면  
$$<s, v^{\pi}> \sim  D$$

\(2\) Stochastic Descent Update 로 parameter들을 update 한다.   
$$\Delta w = \alpha(v^{\pi} - \hat{v}(s, w)) \nabla_w \hat{v}(s, w)$$

이 과정을 반복하면 되는데 이를 experience replay 라고 한다. off-policy 할 때 많이 사용되는 방법이다. Least Squares는 수렴한다. $$ w^{\pi} = \underset{w}{argmin}  LS(w)$$

결국 Batch Method는 experience data를 한번만 사용하는 것이 문제였던 SGD를 해결하기 위해 사용하는 방법이다. \(experience reply\)







Non-Linear 할 때 수렴성을 높이기 위해 experience replay와 fixed Q-targets를 이요한다.

experience reply는 transition 들을 replay memory에 쌓고 랜덤하게 mini-batch를 뽑아서, 그것을 training data로 사용해 학습을 하는 방법이다.

fixed Q-targets : TD target을 계산할 때 parameter 들을 고정시켜놓고 학습킨다. target network를 두개를 이용.

non-lenear할 때 수렴성을 높이기 위해 experien replay : transitoin들을 replay memory에 쌓고 랜덤하게 미니배치를 뽑아서, 그걸로 학습을 한다. fixed Q-targets : TD target을 계산할 때 파라미터를 고정시켜놓고 하다가, 업데이트. target network를 두개를 이용하는 거.

