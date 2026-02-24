---
type: exercise
course: RL
week: 1
source: "Exercise Set 1"
concepts:
  - "[[Multi-Armed Bandit]]"
  - "[[Markov Decision Process]]"
  - "[[Dynamic Programming]]"
  - "[[Bellman Equation]]"
  - "[[Policy Iteration]]"
  - "[[Value Iteration]]"
status: complete
---

# RL Exercise Set Week 1: Prerequisites, Intro & MDPs, Dynamic Programming

This exercise set covers the mathematical foundations required for RL, an introduction to the agent-environment interface, and the basics of solving MDPs using Dynamic Programming.

---

## 0. Prerequisites

### 0.1 Multi-armed Bandits - Introduction Lab
*Download the notebook `RL_WC1_bandit.ipynb` from Canvas and follow the instructions.*

**Concepts tested:** `[[Multi-Armed Bandit]]`, `[[Exploration-Exploitation Trade-off]]`.

---

### 0.2 Prior knowledge self-test

#### 0.2.1 Linear algebra and multivariable derivatives
**Concepts tested:** `[[Linear Algebra]]`, `[[Vector Calculus]]`.

Consider the following matrices and vectors:
$$A = \begin{pmatrix} a_{11} & 0 \\ 0 & a_{22} \end{pmatrix}, \quad B = \begin{pmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{pmatrix}, \quad c = \begin{pmatrix} y - x^2 \\ \ln x \\ y \end{pmatrix}, \quad d = \begin{pmatrix} d_1 \\ d_2 \end{pmatrix}, \quad e = \begin{pmatrix} x \\ y \end{pmatrix}$$

**1. Compute $AB$, $AB^T$, and $d^T B d$.**

> [!tip] Matrix Multiplication Reminder
> For $A \in \mathbb{R}^{N \times K}$ and $B \in \mathbb{R}^{K \times M}$, the product $(AB)_{nm} = \sum_{k=1}^K A_{nk} B_{km}$.
> For a quadratic form $d^T B d$, where $d \in \mathbb{R}^N$ and $B \in \mathbb{R}^{N \times N}$, the result is $\sum_{i=1}^N \sum_{j=1}^N B_{ij} d_i d_j$.

**Solution:**
$$AB = \begin{pmatrix} a_{11}b_{11} & a_{11}b_{12} \\ a_{22}b_{21} & a_{22}b_{22} \end{pmatrix}$$
$$AB^T = \begin{pmatrix} a_{11}b_{11} & a_{11}b_{21} \\ a_{22}b_{12} & a_{22}b_{22} \end{pmatrix}$$
$$d^T B d = d_1 b_{11} d_1 + d_1 b_{12} d_2 + d_2 b_{21} d_1 + d_2 b_{22} d_2$$

**2. Find the inverses of $A$ and $B$.**

**Solution:**
For a diagonal matrix $A$, the inverse is $A^{-1}_{ii} = 1/A_{ii}$:
$$A^{-1} = \begin{pmatrix} 1/a_{11} & 0 \\ 0 & 1/a_{22} \end{pmatrix}$$
For a $2 \times 2$ matrix $M$, $M^{-1} = \frac{1}{\det M} \text{adj}(M)$:
$$B^{-1} = \frac{1}{b_{11}b_{22} - b_{12}b_{21}} \begin{pmatrix} b_{22} & -b_{12} \\ -b_{21} & b_{11} \end{pmatrix}$$

**3. Compute $\frac{\partial c}{\partial x}$ and $\frac{\partial c}{\partial e}$.**

**Solution:**
We use the **numerator layout** (Jacobian formulation): if $v$ is an $n$-vector and $w$ is an $m$-vector, $\frac{\partial v}{\partial w}$ is an $n \times m$ matrix where entry $(i, j)$ is $\frac{\partial v_i}{\partial w_j}$.

$$\frac{\partial c}{\partial x} = \begin{pmatrix} \frac{\partial (y-x^2)}{\partial x} \\ \frac{\partial \ln x}{\partial x} \\ \frac{\partial y}{\partial x} \end{pmatrix} = \begin{pmatrix} -2x \\ 1/x \\ 0 \end{pmatrix}$$
$$\frac{\partial c}{\partial e} = \begin{pmatrix} \frac{\partial (y-x^2)}{\partial x} & \frac{\partial (y-x^2)}{\partial y} \\ \frac{\partial \ln x}{\partial x} & \frac{\partial \ln x}{\partial y} \\ \frac{\partial y}{\partial x} & \frac{\partial y}{\partial y} \end{pmatrix} = \begin{pmatrix} -2x & 1 \\ 1/x & 0 \\ 0 & 1 \end{pmatrix}$$

**4. Consider the function $f(x) = \sum_{i=1}^N i x_i$, which maps an $N$-dimensional vector $x$ to a real number. Find an expression for $\frac{\partial f}{\partial x}$ in terms of integers $1$ to $N$.**

**Solution:**
$$\frac{\partial f}{\partial x} = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_N} \right]$$
For a single term $\frac{\partial f}{\partial x_j}$:
$$\frac{\partial f}{\partial x_j} = \frac{\partial}{\partial x_j} \sum_{i=1}^N i x_i = \sum_{i=1}^N i \frac{\partial x_i}{\partial x_j} = \sum_{i=1}^N i \delta_{ij} = j$$
where $\delta_{ij}$ is the Kronecker delta. Thus:
$$\frac{\partial f}{\partial x} = (1, 2, \dots, N)$$

---

#### 0.2.2 Probability theory
**Concepts tested:** `[[Bias-Variance Trade-off]]`, `[[Probability Theory]]`.

Assume $X$ and $Y$ are two independent random variables with means $\mu, \nu$ and variances $\sigma^2, \tau^2$.

**1. What is the expected value of $X + \alpha Y$, where $\alpha$ is some constant?**
**Solution:** By linearity of expectation: $E[X + \alpha Y] = E[X] + \alpha E[Y] = \mu + \alpha \nu$.

**2. What is the variance of $X + \alpha Y$?**
**Solution:** For independent variables: $\text{Var}(aX + bY) = a^2 \text{Var}(X) + b^2 \text{Var}(Y)$.
Thus, $\text{Var}(X + \alpha Y) = \text{Var}(X) + \alpha^2 \text{Var}(Y) = \sigma^2 + \alpha^2 \tau^2$.

**3. Explain the terms in the bias-variance decomposition of squared error:**
$$E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$

**Solution:**
- **Bias**: Error from simplifying assumptions. Large when the model is too simple (underfitting).
- **Variance**: Sensitivity of the model to the specific training set. High when the model is too complex and fits noise (overfitting).
- **Irreducible Error ($\sigma^2$)**: The noise in the data itself. Cannot be reduced by improving the model.

**4. Explain why this is a "trade-off".**
**Solution:** Reducing bias usually requires increasing model complexity, which increases variance (as the model starts fitting spurious correlations/noise in the training data). Conversely, reducing variance (e.g., via regularization or simpler models) often increases bias by making stronger assumptions.

---

#### 0.2.3 OLS, linear projection, and gradient descent
**Concepts tested:** `[[Ordinary Least Squares|OLS]]`, `[[Gradient Descent]]`, `[[Linear Algebra]]`.

Given training set $X$ ($n \times m$) and labels $y$ ($n \times 1$). Fit $f_\beta(X) = X\beta$ by minimizing $\|y - X\beta\|_2^2$.

**1. What is the dimensionality of $\beta$?**
**Solution:** $\beta \in \mathbb{R}^m$ (one parameter for each feature).

**2. Show by differentiation that the OLS estimator $\hat{\beta} = (X^T X)^{-1} X^T y$.**
**Solution:**
Define $L(\beta) = (y - X\beta)^T (y - X\beta)$. Set $\frac{\partial L}{\partial \beta} = 0$:
$$\frac{\partial}{\partial \beta} (y^T y - 2y^T X \beta + \beta^T X^T X \beta) = 0$$
$$-2X^T y + 2X^T X \beta = 0$$
$$X^T X \beta = X^T y \implies \beta = (X^T X)^{-1} X^T y$$

**3-5. Geometric Interpretation (Figure 1)**

> [!example] Figure 1: Geometric representation of OLS
> ```text
>          y (Target vector)
>          ^
>          |
>          |   ε (Residual vector, perpendicular to plane)
>          |   :
>          |  /|
>          | / |
>          |/  |
>    ______●---+------------------  <- Subspace spanned by columns of X (plane "col X")
>    \    /   /
>     \  /   X1 (Regressor 1)
>      \. (Origin)
>       \
>        X2 (Regressor 2)
> ```
> **Description:** $y$ lies outside the plane spanned by $X_i$. The OLS prediction $X\hat{\beta}$ is the orthogonal projection of $y$ onto the plane (the point ●). The residual $\epsilon = y - X\hat{\beta}$ is the shortest distance from $y$ to the plane, hence it must be orthogonal to the plane.

**Solution (3-5):**
- **Minimizing L2 norm** is equivalent to finding the shortest distance from $y$ to the subspace. This is achieved by the orthogonal projection $P(y)$.
- **Orthogonality** means the residual $\epsilon_\beta$ is perpendicular to every column in $X$, hence $X^T \epsilon_\beta = 0$.
- **Derivation**: $X^T(y - X\beta) = 0 \implies X^T y - X^T X \beta = 0 \implies \hat{\beta} = (X^T X)^{-1} X^T y$.

**6. What is the loss function for OLS?**
**Solution:** $L_\beta(y, X) = \|y - f_\beta(X)\|_2^2$ (Squared $L_2$ norm).

**7. Write the gradient descent update rule for $\beta$.**
**Solution:** $\beta_{t+1} = \beta_t - \alpha \frac{\partial L}{\partial \beta_t} = \beta_t + 2\alpha X^T(y - X\beta_t)$.

---

## 1. Introduction & MDPs

### 1.1 Introduction
**Concepts tested:** `[[Course of Dimensionality]]`, `[[State Space]]`.

**1. Explain the "curse of dimensionality".**
**Solution:** Computational requirements (and the amount of data needed) grow exponentially with the number of state variables.

**2. Predator-Prey on $5 \times 5$ toroidal grid.**
- (a) **Naive state space**: $(x_p, y_p, x_q, y_q) \implies 5^4 = 625$ states.
- (b) **Reduced representation**: Relative distance $(\Delta x, \Delta y) = (x_p - x_q, y_p - y_q) \pmod 5$.
- (c) **New size**: $5^2 = 25$ states.
- (d) **Advantage**: Alleviates the curse of dimensionality, making the problem easier to solve.
- (e) **Tic-Tac-Toe**: Exploiting symmetries (rotational, reflectional) to reduce the value function representation.

**3. Greedy vs. Non-greedy agent.**
**Solution:** Non-greedy (exploratory) agent usually performs better long-term. It discovers better strategies that the greedy agent might miss by settling for a sub-optimal "local" maximum too early.

**4. Annealing exploration ($\epsilon$).**
- (a) **Method**: Start with high $\epsilon$ (e.g., 1.0) and decrease it over time (e.g., $\epsilon_t = \frac{1}{\sqrt{t}}$ or linear decay) as the agent learns.
- (b) **Non-stationary environments**: If the opponent changes strategies, time-based annealing fails. The agent will be "locked in" to an old strategy. **Heuristic suggestion**: Increase $\epsilon$ when the TD-error becomes large again, indicating the environment model is no longer accurate.

---

### 1.2 Exploration
**Concepts tested:** `[[Exploration-Exploitation Trade-off]]`, `[[Epsilon-Greedy]]`, `[[Optimistic Initial Values]]`.

**1. Probability of selecting the greedy action in $\epsilon$-greedy?**
**Solution:** $P(a^*) = (1 - \epsilon) + \frac{\epsilon}{n}$, where $n$ is the number of actions. (Probability from exploitation + probability of picking it randomly during exploration).

**2. 3-armed bandit sequence (Start $Q=[0,0,0]$).**
- $A_0=1, R_1=-1 \implies Q=[−1, 0, 0]$
- $A_1=2, R_2=1 \implies Q=[−1, 1, 0]$
- $A_2=2, R_3=-2 \implies Q=[−1, \frac{1-2}{2}, 0] = [−1, −0.5, 0]$
- $A_3=2, R_4=2 \implies Q=[−1, \frac{-1+2}{3}, 0] = [−1, 0.333, 0]$
- $A_4=3, R_5=1 \implies Q=[−1, 0.333, 1]$
**Result:** $A_3$ and $A_4$ were non-greedy (exploratory) because $Q$ for action 2 was not the maximum when they were selected.

**3-6. Pessimistic vs. Optimistic Initialization.**
- Arm 1 ($+1$), Arm 2 ($-1$).
- **Optimistic ($+5$)**: $A_0$ (Arm 1) $\to Q=[1, 5]$. $A_1$ (Arm 2) $\to Q=[1, -1]$. $A_2$ (Arm 1) $\to Q=[1, -1]$. Return = $1-1+1 = 1$.
- **Pessimistic ($-5$)**: $A_0$ (Arm 1) $\to Q=[1, -5]$. $A_1$ (Arm 1) $\to Q=[1, -5]$. $A_2$ (Arm 1) $\to Q=[1, -5]$. Return = $1+1+1 = 3$.
- **Comparison**:
    - **Return**: Pessimistic had higher return (stayed with the first good arm).
    - **Estimation**: Optimistic leads to better Q-value estimates because it **forced** exploration of all arms.
    - **Exploration**: Optimistic initialization is a "trick" for exploration; the high initial value makes all unexplored arms look better than explored ones, forcing the agent to try everything.

---

### 1.3 Markov Decision Processes
**Concepts tested:** `[[Markov Decision Process|MDP]]`, `[[Return]]`, `[[Discount Factor]]`.

**1. MDP Definitions.**
- **Chess**: State = board config; Action = legal moves; Reward = $+1$ (win), $-1$ (loss).
- **Robot Maze**: State = position, velocity, and variables like "has key".
- **Driving**:
    - Low-level (accelerator/brake): Fine control but hard to learn long sequences.
    - High-level (navigate to X): Easier planning but assumes sub-skills already exist.
    - **Hybrid**: `[[Hierarchical Reinforcement Learning|HRL]]` (Low-level skills for "how to drive", High-level for "where to go").

**2. Return and Geometric Series.**
- (a) **Episodic Return**: $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$.
- (b) **Proof of $\sum_{k=0}^\infty \gamma^k = \frac{1}{1-\gamma}$**:
    Let $S = 1 + \gamma + \gamma^2 + \dots$
    $\gamma S = \gamma + \gamma^2 + \gamma^3 + \dots$
    $S - \gamma S = 1 \implies S(1-\gamma) = 1 \implies S = \frac{1}{1-\gamma}$ (for $|\gamma| < 1$).
- (c-e) **Robot in escape room**: If reward is only $+1$ at exit, and no discount, $G_t$ is always $1$ regardless of time. Agent has no incentive to be fast.
    - **Fix 1**: Use $\gamma < 1$. Then $G_t = \gamma^{T-t}$, which is maximized when $T-t$ (steps to exit) is smallest.
    - **Fix 2**: Use time penalty $R_t = -c$ per step.

---

## 2. Dynamic Programming

### 2.1 Dynamic programming
**Concepts tested:** `[[Value Iteration]]`, `[[Optimal Policy]]`.

> [!example] Figure 2: MDP with 3 States
> ```text
>        Action A: -2
>     (1) ----------> (2)
>      ^               |
>      |               | Action D: -10.5
>      | Action C: -3  |
>      +--------------(3) <--- Action E: 0 (Terminal)
>      |
>      | Action B: -5
>      | (Prob 1/3 to 2, 2/3 to 3)
> ```
> **Detailed MDP Transitions:**
> - **State 1**: $A \to 2$ ($r=-2$); $B \to \{2 \text{ w.p. } 1/3, 3 \text{ w.p. } 2/3\}$ ($r=-5$).
> - **State 2**: $C \to 1$ ($r=-3$); $D \to 3$ ($r=-10.5$).
> - **State 3**: $E \to 3$ ($r=0$, Terminal).

**Value Iteration Walkthrough ($\gamma = 1$):**
Initialize $v_0(s) = [0, 0, 0]$.
Update: $v_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]$

**Iteration 1:**
- $v_1(1) = \max(-2+0, -5 + \frac{1}{3}(0) + \frac{2}{3}(0)) = -2$ (Action A)
- $v_1(2) = \max(-3+0, -10.5+0) = -3$ (Action C)
- $v_1(3) = 0$ (Action E)
$\mathbf{v_1 = [-2, -3, 0]}$

**Iteration 2:**
- $v_2(1) = \max(-2 + v_1(2), -5 + \frac{1}{3}v_1(2) + \frac{2}{3}v_1(3)) = \max(-2-3, -5-1) = -5$
- $v_2(2) = \max(-3 + v_1(1), -10.5) = \max(-3-2, -10.5) = -5$
$\mathbf{v_2 = [-5, -5, 0]}$

**Convergence:**
Continuing until convergence yields $V^* = [-8.5, -10.5, 0]$.

---

### 2.2 * Exam question: Dynamic programming
**Concepts tested:** `[[Value Iteration]]`, `[[Policy Iteration]]`, `[[Bellman Equation|Bellman Optimality Equation]]`.

**1. True/False:**
- (a) **False**: Value Iteration and Policy Iteration both converge to **optimal** policies.
- (b) **True**: Value Iteration effectively does one step of policy evaluation followed by policy improvement in each sweep.

**2. Why does the Bellman Optimality Equation hold at stabilization?**
**Solution:**
When Policy Iteration stabilizes:
1. **Policy Improvement** yields no change: $\pi(s) = \arg\max_a \sum_{s', r} p(s', r | a, s) [r + \gamma v_\pi(s')]$.
2. **Policy Evaluation** is consistent: $v_\pi(s) = \sum_{s', r} p(s', r | \pi(s), s) [r + \gamma v_\pi(s')]$.
Substituting (1) into (2):
$$v_\pi(s) = \max_a \sum_{s', r} p(s', r | a, s) [r + \gamma v_\pi(s')]$$
This is the Bellman Optimality Equation, meaning $v_\pi$ is $v^*$.
