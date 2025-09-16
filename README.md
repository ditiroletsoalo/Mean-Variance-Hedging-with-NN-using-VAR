### Preface

Let $(\Omega, \mathcal{F}, \mathbb{F}, P)$ be a filtered probability space, and let $X = \{(X_t^1, \ldots, X_t^d) : t \geq 0\}$ be a $d$-dimensional stochastic process representing the discounted prices of $d$ risky assets. Consider a company with an obligation to make a payment at some future time $T > 0$, represented by the random variable $H$. This payment could be, for example, an option on a share.

Ideally, the company aims to construct a portfolio of assets that will match the value of $H$ at time $T$ under all circumstances. If perfect replication is not feasible, the goal is to find a portfolio whose value is as close as possible to $H$ at time $T$. Traditional models like the Black-Scholes model in continuous time and the Binomial model in discrete time allow for such perfect replication. However, we seek to generalize these ideas to find portfolios that replicate a payoff as closely as possible by minimizing a quadratic utility functional.

Specifically, if $V_T(\phi)$ denotes the value of a trading strategy $\phi$ at time $T$, we aim to choose a strategy that minimizes the expected squared replication error:
$$
\mathbb{E}[(V_T(\phi) - H)^2]
$$
over all admissible trading strategies.

To achieve this, we leverage the universal approximation theorem to estimate the trading strategies $\phi$ using $\phi_\theta$ with a Long Short-Term Memory (LSTM)-like recurrent neural network. This approach is defined as follows:
$$
C_n^\theta = g_{NN}^\theta(C_{n-1}^\theta, X_n) \quad \text{and} \quad \phi_{n+1}^\theta = f_{NN}^\theta(C_n^\theta)
$$
for $n = 0, 1, \ldots, T-1$, where $f_{NN}$ and $g_{NN}$ are neural networks and $\theta$ is the vector of parameters for these networks. This formulation ensures that $\phi$ is a predictable process, allowing the terminal value to be written as:
$$
V_T^\theta(\phi) = v_0 + (\phi_\theta \cdot X)_T.
$$

We then find parameter estimates $\hat{\theta}$ such that:
$$
\mathbb{E}[(V_T^{\hat{\theta}} - H)^2] = \min_\theta \mathbb{E}[(V_T^\theta - H)^2].
$$

A significant innovation in this research is the use of Vector Autoregression (VAR) models as the underlying sample path generators. VAR models are powerful tools in econometrics, capable of capturing linear interdependencies among multiple time series. By fitting a VAR model to historical data, we can simulate realistic future scenarios, providing a robust framework for training our neural network-based hedging strategies.

By integrating the statistical rigor of VAR models with the approximation power of neural networks, this research aims to contribute to the field of quantitative finance, offering new insights and tools for effective risk management. The outcomes of this study have the potential to enhance the robustness and accuracy of hedging strategies, ultimately benefiting financial practitioners and advancing academic understanding in this domain.





### Sigma algebra
$\pmb {Definition}.$ Let $\Omega$ be a (non-empty) set. A collection $A\subseteq 2 ^\Omega$ of subsets of Ω is called an
algebra on $\Omega$ if 
$\mathcal{A} 1. \ \emptyset \in \mathcal{A};$ 

$\mathcal{A} 2.$ For every $A \subseteq \Omega,$ if $A \in \mathcal{A},$ then $A^c \in \mathcal{A}$ 

$\mathcal{A}3.$ For every $A,\ B \subseteq \Omega,$ if $A, \ B \in \mathcal{A},$ then $A \cup B \in \mathcal{A}$ 


An algebra $\mathcal F$ is called a $\sigma-algebra$ or $\sigma-field$ if it also satisfies the following additional condition



$\sigma \mathcal{A}4.$ For every countable collection $\{A_n : n=1,2,3,\ldots\}$ of subsets of $\Omega$ with $A_n \in \mathcal{F}$ for each $n \geq 1,$

$$
\begin{align*}
\bigcup_{n=1}^{\infty} A_n &\in \mathcal{F} \\
\end{align*}
$$

### Measure space
If $\Omega$ is a set and $\mathcal F$ is a sigma algebra on $\Omega$, we will call the pair ($\Omega$, $\mathcal F$) a measurable space and
the elements of $\mathcal F$ will be called measurable sets.


### stochastic processes
$\pmb {Definition}.$ Let ($\Omega$, $\mathcal F$, $\mathbb {P}$) be a probability space and $\emptyset \neq \mathbb I \subseteq [0, \infty)$ be a set. A stochastic
process $X= \{ X_t : t \in \mathbb I})$ is a collection of random variables. We call I the index set. If $\mathbb I \subseteq \mathbb N,$
we call X a discrete-time stochastic process, while if $\mathbb I = [0,\infty)$ or $\mathbb I = [0, T]$ for some $T > 0$, we
call X a continuous-time stochastic process.


### Filtration 

$\pmb {Definition}.$ Let ($\Omega$, $\mathcal F$, $\mathbb {P}$) be a probability space and $\mathbb I = [0, \infty)$ be an index set. A filtration $\mathbb {F} = \{ \mathcal F_t:t \in \mathbb I \}$ is an increasing collection of $sub-\sigma-algebras$ of $\mathcal F$; i.e., $\mathcal F_s \subseteq \mathcal F_t \subseteq \mathcal F$ for $s \leq t,$ $s, t \in \mathbb I$. A stochastic process $X = {X_t: t \in \mathbb I}$ is adapted to a filtration $\mathbb F$ if for every $t \in \mathbb I,$
$X_t$ is $\mathcal F_t-measurable$
