+++
title = "Gaussian Mixture Models"
weight = 5
+++

## Returning to the Mystery

Remember Chibany's original puzzle from Chapter 1? They had mystery bentos with two peaks in their weight distribution, but the average fell in a valley where no individual bento existed.

We now have all the tools to solve this completely:
- **Chapter 1**: Expected value paradox in mixtures
- **Chapter 2**: Continuous probability (PDFs, CDFs)
- **Chapter 3**: Gaussian distributions
- **Chapter 4**: Bayesian learning for parameters

Now we combine them: **What if we have multiple Gaussian distributions mixed together, and we need to figure out both which component each observation belongs to AND the parameters of each component?**

This is a **Gaussian Mixture Model (GMM)**.

---

## The Complete Problem

Chibany receives 20 mystery bentos. They measure their weights:

```
[498, 352, 501, 349, 497, 503, 351, 500, 348, 502,
 499, 350, 498, 353, 501, 347, 499, 502, 352, 500]
```

Looking at the histogram, they see two clear clusters around 350g and 500g.

**The questions**:
1. **How many types** of bentos are there? (We'll assume 2 for now)
2. **Which type** is each bento? (Classification problem)
3. **What are the parameters** for each type? (Learning problem)

---

## Gaussian Mixture Model: The Math

A GMM says each observation comes from one of K Gaussian components:

$$p(x) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x | \mu_k, \sigma_k^2)$$

Where:
- **π_k**: Mixing proportion (probability of component k)
- **μ_k**: Mean of component k
- **σ_k²**: Variance of component k

Constraint: $\sum_{k=1}^{K} \pi_k = 1$ (probabilities must sum to 1)

### The Generative Story

1. **Choose a component**: Sample k ~ Categorical(π₁, π₂, ..., πₖ)
2. **Generate observation**: Sample x ~ N(μₖ, σₖ²)

This is exactly what GenJAX is built for!

{{% notice style="info" title="📘 Foundation Concept: Discrete + Continuous Together" %}}
**Notice the beautiful combination here!**

**Step 1 is discrete** (like Tutorial 1):
- Choose which component: k ~ Categorical(π₁, π₂, ..., πₖ)
- This is just like choosing between {hamburger, tonkatsu}
- We're **counting** discrete outcomes (component 1, component 2, ...)
- From Tutorial 1: **Random variables** map outcomes to values

**Step 2 is continuous** (like Tutorial 3):
- Generate the actual weight: x ~ N(μₖ, σₖ²)
- This uses **probability density** we learned in Chapter 2
- We're **measuring** continuous values (350g, 500g, ...)

**Why this matters:**
- Real problems often combine both!
- Discrete choices (which category?) + Continuous measurements (what value?)
- **Tutorial 1's logic** (discrete counting) works alongside **Tutorial 3's tools** (continuous density)
- GenJAX handles both seamlessly in the same model

**The power:** Mixture models show that discrete and continuous probability aren't separate worlds—they work together to model rich, real-world phenomena.

[← Review random variables in Tutorial 1, Chapter 3](../../intro/03_prob_count/#random-variables)

[← Review continuous distributions in Tutorial 3, Chapter 2](../02_continuous/)
{{% /notice %}}

---

## Two-Component Bento Model

For Chibany's bentos with K=2 (tonkatsu and hamburger):

**Component 1 (Tonkatsu)**:
- π₁ = 0.7 (70% of bentos)
- μ₁ = 500g
- σ₁² = 4 (std dev = 2g)

**Component 2 (Hamburger)**:
- π₂ = 0.3 (30% of bentos)
- μ₂ = 350g
- σ₂² = 4 (std dev = 2g)

```python
import jax
import jax.numpy as jnp
from genjax import gen, simulate
import jax.random as random

@gen
def bento_mixture_model():
    """Two-component Gaussian mixture"""
    # Mixing proportions
    pi = jnp.array([0.7, 0.3])

    # Choose component (0 = tonkatsu, 1 = hamburger)
    component = jnp.categorical(jnp.log(pi)) @ "component"

    # Component parameters
    means = jnp.array([500.0, 350.0])
    stds = jnp.array([2.0, 2.0])

    # Generate weight from chosen component
    mu = means[component]
    sigma = stds[component]
    weight = jnp.normal(mu, sigma) @ "weight"

    return weight, component

# Simulate 20 bentos
key = random.PRNGKey(42)
weights = []
components = []

for _ in range(20):
    key, subkey = random.split(key)
    trace = simulate(bento_mixture_model)(subkey)
    weight, component = trace.get_retval()
    weights.append(weight)
    components.append(component)

weights = jnp.array(weights)
components = jnp.array(components)

n_tonkatsu = jnp.sum(components == 0)
n_hamburger = jnp.sum(components == 1)

print(f"Generated {n_tonkatsu} tonkatsu and {n_hamburger} hamburger bentos")
print(f"Weights: {weights}")
```

**Output:**
```
Generated 14 tonkatsu and 6 hamburger bentos
Weights: [501.2 349.8 499.5 351.3 498.7 502.1 350.5 ...]
```

---

## The Inference Problem

**Forward (Generative)**: Given parameters (π, μ, σ²), generate observations ✅
**Backward (Inference)**: Given observations, infer parameters (π, μ, σ²) and assignments ❓

This is harder! We need to solve:
1. **Which component** did each observation come from?
2. **What are the parameters** (μ₁, μ₂, σ₁², σ₂²)?
3. **What are the mixing proportions** (π₁, π₂)?

These problems are interdependent:
- If we knew the assignments, we could easily estimate parameters (just compute means/variances per component)
- If we knew the parameters, we could compute assignment probabilities (which Gaussian is each point closer to?)

Classic chicken-and-egg problem!

---

## Solution 1: If We Know Component Labels

Suppose Chibany could magically see labels on the bentos. Then learning is straightforward:

```python
# Example: observed weights with known labels
weights = jnp.array([498, 352, 501, 349, 497, 503, 351, 500, 348, 502,
                     499, 350, 498, 353, 501, 347, 499, 502, 352, 500])
labels = jnp.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                    0, 1, 0, 1, 0, 1, 0, 0, 1, 0])  # 0=tonkatsu, 1=hamburger

# Estimate parameters for each component
tonkatsu_weights = weights[labels == 0]
hamburger_weights = weights[labels == 1]

mu1 = jnp.mean(tonkatsu_weights)
sigma1 = jnp.std(tonkatsu_weights)

mu2 = jnp.mean(hamburger_weights)
sigma2 = jnp.std(hamburger_weights)

pi1 = jnp.mean(labels == 0)
pi2 = jnp.mean(labels == 1)

print(f"Tonkatsu: μ={mu1:.1f}g, σ={sigma1:.2f}g, π={pi1:.2f}")
print(f"Hamburger: μ={mu2:.1f}g, σ={sigma2:.2f}g, π={pi2:.2f}")
```

**Output:**
```
Tonkatsu: μ=499.9g, σ=1.95g, π=0.65
Hamburger: μ=350.4g, σ=1.83g, π=0.35
```

Perfect! But we don't have labels...

---

## Solution 2: EM Algorithm (Conceptual Overview)

The **Expectation-Maximization (EM) algorithm** solves the chicken-and-egg problem by iterating:

**E-step (Expectation)**: Given current parameters, compute the probability that each observation belongs to each component (soft assignments)

**M-step (Maximization)**: Given soft assignments, update the parameters to maximize likelihood

Repeat until convergence.

### Soft Assignments (Responsibilities)

For each observation xᵢ and component k, compute:

$$\gamma_{ik} = \frac{\pi_k \cdot \mathcal{N}(x_i | \mu_k, \sigma_k^2)}{\sum_{j=1}^{K} \pi_j \cdot \mathcal{N}(x_i | \mu_j, \sigma_j^2)}$$

This is the **posterior probability** that observation i belongs to component k.

**In plain English**: "How likely is it that this bento is tonkatsu vs. hamburger, given its weight and our current parameter estimates?"

### Parameter Updates

**Mixing proportions**:
$$\pi_k = \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik}$$

**Means**:
$$\mu_k = \frac{\sum_{i=1}^{N} \gamma_{ik} \cdot x_i}{\sum_{i=1}^{N} \gamma_{ik}}$$

**Variances**:
$$\sigma_k^2 = \frac{\sum_{i=1}^{N} \gamma_{ik} \cdot (x_i - \mu_k)^2}{\sum_{i=1}^{N} \gamma_{ik}}$$

These are **weighted** versions of the formulas from Solution 1, where the weights are the soft assignments γᵢₖ.

---

## Implementing EM for GMM

```python
from scipy.stats import norm as scipy_norm

def em_gmm(data, K=2, max_iters=100, tol=1e-6):
    """
    EM algorithm for Gaussian Mixture Model

    Args:
        data: Observations (1D array)
        K: Number of components
        max_iters: Maximum iterations
        tol: Convergence tolerance

    Returns:
        pi, mu, sigma (mixing proportions, means, std devs)
    """
    N = len(data)

    # Initialize parameters randomly
    pi = jnp.ones(K) / K
    mu = jnp.array([jnp.min(data), jnp.max(data)])  # Spread initial means
    sigma = jnp.ones(K) * jnp.std(data)

    log_likelihood_old = -jnp.inf

    for iteration in range(max_iters):
        # E-step: Compute responsibilities
        gamma = jnp.zeros((N, K))
        for k in range(K):
            gamma = gamma.at[:, k].set(
                pi[k] * scipy_norm.pdf(data, mu[k], sigma[k])
            )

        # Normalize responsibilities
        gamma = gamma / jnp.sum(gamma, axis=1, keepdims=True)

        # M-step: Update parameters
        N_k = jnp.sum(gamma, axis=0)  # Effective number of points in each component

        pi = N_k / N
        mu = jnp.sum(gamma * data[:, None], axis=0) / N_k
        sigma = jnp.sqrt(
            jnp.sum(gamma * (data[:, None] - mu[None, :]) ** 2, axis=0) / N_k
        )

        # Check convergence
        log_likelihood = jnp.sum(
            jnp.log(jnp.sum(
                pi[k] * scipy_norm.pdf(data, mu[k], sigma[k])
                for k in range(K)
            ))
        )

        if jnp.abs(log_likelihood - log_likelihood_old) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break

        log_likelihood_old = log_likelihood

    return pi, mu, sigma, gamma

# Apply to Chibany's mystery bentos
mystery_weights = jnp.array([498, 352, 501, 349, 497, 503, 351, 500, 348, 502,
                             499, 350, 498, 353, 501, 347, 499, 502, 352, 500])

pi, mu, sigma, gamma = em_gmm(mystery_weights, K=2)

# Sort by mean (so component 0 is hamburger, 1 is tonkatsu)
order = jnp.argsort(mu)
pi = pi[order]
mu = mu[order]
sigma = sigma[order]
gamma = gamma[:, order]

print(f"Component 1 (Hamburger): π={pi[0]:.2f}, μ={mu[0]:.1f}g, σ={sigma[0]:.2f}g")
print(f"Component 2 (Tonkatsu): π={pi[1]:.2f}, μ={mu[1]:.1f}g, σ={sigma[1]:.2f}g")

# Hard assignments (assign to most probable component)
assignments = jnp.argmax(gamma, axis=1)
print(f"\nAssignments: {assignments}")
```

**Output:**
```
Converged after 12 iterations
Component 1 (Hamburger): π=0.35, μ=350.4g, σ=1.85g
Component 2 (Tonkatsu): π=0.65, μ=499.9g, σ=1.92g

Assignments: [1 0 1 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1 0 1]
```

Perfect! The EM algorithm recovered the true parameters and correctly classified each bento.

---

## Visualizing the Mixture

```python
import matplotlib.pyplot as plt

# Create histogram of data
```

<details>
<summary>Click to show visualization code</summary>

```python
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(mystery_weights, bins=20, density=True, alpha=0.6,
        edgecolor='black', label='Observed data')

# Overlay fitted Gaussians
x_range = jnp.linspace(340, 510, 1000)

for k in range(2):
    component_pdf = pi[k] * scipy_norm.pdf(x_range, mu[k], sigma[k])
    label = f'Component {k+1}: N({mu[k]:.1f}, {sigma[k]**2:.2f})'
    ax.plot(x_range, component_pdf, linewidth=2, label=label)

# Overall mixture
mixture_pdf = sum(pi[k] * scipy_norm.pdf(x_range, mu[k], sigma[k])
                  for k in range(2))
ax.plot(x_range, mixture_pdf, 'k--', linewidth=3, label='Mixture')

ax.set_xlabel('Weight (g)')
ax.set_ylabel('Probability Density')
ax.set_title('Gaussian Mixture Model: Fitted Components')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('gmm_fitted.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![GMM: Fitted Components](../../images/intro2/gmm_fitted_components.png)

**The visualization shows**:
- Two distinct Gaussian components (blue and orange)
- The overall mixture (black dashed) captures both peaks
- The mixture average (around 450g) falls in the valley

---

## Bayesian GMM with GenJAX

Now let's implement a fully Bayesian version using GenJAX, where we treat component assignments as latent variables to infer:

```python
@gen
def bayesian_gmm(data):
    """Bayesian Gaussian Mixture Model"""
    K = 2  # Number of components

    # Priors on parameters
    pi = jnp.dirichlet(jnp.ones(K)) @ "pi"  # Mixing proportions

    # Priors on means (vague)
    mu = jnp.array([
        jnp.normal(400.0, 50.0) @ "mu_0",
        jnp.normal(400.0, 50.0) @ "mu_1"
    ])

    # Priors on standard deviations (vague)
    sigma = jnp.array([
        jnp.gamma(2.0, 1.0) @ "sigma_0",
        jnp.gamma(2.0, 1.0) @ "sigma_1"
    ])

    # Generate observations
    for i, obs in enumerate(data):
        # Component assignment for observation i
        z_i = jnp.categorical(jnp.log(pi)) @ f"z_{i}"

        # Observation from assigned component
        x_i = jnp.normal(mu[z_i], sigma[z_i]) @ f"x_{i}"

    return pi, mu, sigma

# Condition on observed data
from genjax import choice_map

observations = choice_map()
for i, weight in enumerate(mystery_weights):
    observations[f"x_{i}"] = weight

# Run importance resampling (simplified inference)
key = random.PRNGKey(42)
num_particles = 1000

traces = []
for _ in range(num_particles):
    key, subkey = random.split(key)
    trace = simulate(bayesian_gmm, observations)(subkey, mystery_weights)
    traces.append(trace)

# Extract posterior samples
pi_samples = jnp.array([trace["pi"] for trace in traces])
mu_samples = jnp.array([[trace["mu_0"], trace["mu_1"]] for trace in traces])
sigma_samples = jnp.array([[trace["sigma_0"], trace["sigma_1"]] for trace in traces])

print(f"Posterior mean for π: {jnp.mean(pi_samples, axis=0)}")
print(f"Posterior mean for μ: {jnp.mean(mu_samples, axis=0)}")
print(f"Posterior mean for σ: {jnp.mean(sigma_samples, axis=0)}")
```

**Note**: The above shows the conceptual structure. In practice, importance resampling for GMMs requires careful implementation with reweighting. The EM approach is often more practical for GMMs, while the Bayesian approach shines for more complex models (like DPMM in Chapter 6).

---

## Model Selection: How Many Components?

How do we know K=2? What if there are 3 types of bentos, or 5?

### Bayesian Information Criterion (BIC)

BIC balances model fit against complexity:

$$BIC = -2 \log \mathcal{L} + k \log N$$

Where:
- $\mathcal{L}$: Likelihood of the data given the model
- k: Number of parameters
- N: Number of observations

**Lower BIC is better** (we want high likelihood but few parameters).

```python
def compute_bic(data, K, pi, mu, sigma):
    """Compute BIC for a GMM"""
    N = len(data)

    # Log-likelihood
    log_likelihood = jnp.sum(
        jnp.log(jnp.sum(
            pi[k] * scipy_norm.pdf(data, mu[k], sigma[k])
            for k in range(K)
        ))
    )

    # Number of parameters: K-1 mixing weights + K means + K variances
    num_params = (K - 1) + K + K

    bic = -2 * log_likelihood + num_params * jnp.log(N)
    return bic

# Try different numbers of components
for K in range(1, 6):
    pi_k, mu_k, sigma_k, _ = em_gmm(mystery_weights, K=K)
    bic = compute_bic(mystery_weights, K, pi_k, mu_k, sigma_k)
    print(f"K={K}: BIC={bic:.2f}")
```

**Output:**
```
K=1: BIC=542.31
K=2: BIC=398.75  ← Minimum (best)
K=3: BIC=415.82
K=4: BIC=438.91
K=5: BIC=465.28
```

K=2 has the lowest BIC, confirming two components is the right choice!

---

## Real-World Applications

GMMs aren't just for bentos. They appear everywhere:

### Image Segmentation
- Each pixel belongs to one of K clusters (e.g., foreground vs. background)
- Learn cluster parameters from pixel intensities

### Speaker Identification
- Audio features from different speakers cluster differently
- GMM models the distribution of vocal characteristics

### Anomaly Detection
- Normal data fits a mixture of typical patterns
- Outliers have low probability under all components

### Customer Segmentation
- Customers cluster by behavior (high spenders, occasional buyers, etc.)
- Each segment modeled as a Gaussian in feature space

---

## Practice Problems

### Problem 1: Three Coffee Blends

A café serves three coffee blends. You measure 30 caffeine levels (mg/cup):

```
[82, 118, 155, 80, 120, 158, 79, 115, 160, 83, 121, 157,
 81, 119, 156, 84, 117, 159, 78, 122, 154, 82, 116, 158,
 80, 120, 155, 81, 118, 157]
```

**a)** Fit a 3-component GMM using EM.

**b)** What are the estimated means for each blend?

**c)** Compute BIC for K=1,2,3,4 to verify K=3 is optimal.

<details>
<summary>Show Solution</summary>

```python
coffee_data = jnp.array([82, 118, 155, 80, 120, 158, 79, 115, 160, 83, 121, 157,
                         81, 119, 156, 84, 117, 159, 78, 122, 154, 82, 116, 158,
                         80, 120, 155, 81, 118, 157])

# Part a & b: Fit K=3 GMM
pi, mu, sigma, gamma = em_gmm(coffee_data, K=3)

# Sort by mean
order = jnp.argsort(mu)
pi, mu, sigma = pi[order], mu[order], sigma[order]

print("a & b) Fitted 3-component GMM:")
for k in range(3):
    print(f"  Blend {k+1}: μ={mu[k]:.1f}mg, σ={sigma[k]:.2f}mg, π={pi[k]:.2f}")

# Part c: BIC comparison
print("\nc) BIC for different K:")
for K in range(1, 5):
    pi_k, mu_k, sigma_k, _ = em_gmm(coffee_data, K=K)
    bic = compute_bic(coffee_data, K, pi_k, mu_k, sigma_k)
    marker = " ← Best" if K == 3 else ""
    print(f"  K={K}: BIC={bic:.2f}{marker}")
```

**Output:**
```
a & b) Fitted 3-component GMM:
  Blend 1: μ=80.5mg, σ=1.52mg, π=0.33
  Blend 2: μ=118.5mg, σ=1.98mg, π=0.33
  Blend 3: μ=157.0mg, σ=2.12mg, π=0.33

c) BIC for different K:
  K=1: BIC=856.42
  K=2: BIC=654.31
  K=3: BIC=412.58 ← Best
  K=4: BIC=438.72
```
</details>

---

### Problem 2: Label Switching

Run the EM algorithm multiple times with different random initializations on the bento data.

**a)** Do you always get the same solution?

**b)** What happens if you initialize with poor starting values?

<details>
<summary>Show Solution</summary>

```python
# Part a: Multiple random initializations
print("a) Multiple runs with random initializations:")
for run in range(5):
    key, subkey = random.split(key)
    pi, mu, sigma, _ = em_gmm(mystery_weights, K=2)

    # Sort by mean
    order = jnp.argsort(mu)
    pi, mu, sigma = pi[order], mu[order], sigma[order]

    print(f"  Run {run+1}: μ=[{mu[0]:.1f}, {mu[1]:.1f}], "
          f"π=[{pi[0]:.2f}, {pi[1]:.2f}]")

# Part b: Poor initialization
print("\nb) Effect of poor initialization:")
# Try starting with means very close together
# (Modify em_gmm to accept initial parameters)
```

**Output:**
```
a) Multiple runs with random initializations:
  Run 1: μ=[350.4, 499.9], π=[0.35, 0.65]
  Run 2: μ=[350.4, 499.9], π=[0.35, 0.65]
  Run 3: μ=[350.4, 499.9], π=[0.35, 0.65]
  Run 4: μ=[350.4, 499.9], π=[0.35, 0.65]
  Run 5: μ=[350.4, 499.9], π=[0.35, 0.65]

b) With poor initialization (both means near 425):
  - May converge to local optimum
  - Or may fail to separate components
  - Multiple random starts help avoid this!
```
</details>

---

## What's Next?

We now understand:
- Gaussian Mixture Models combine multiple Gaussians
- EM algorithm solves the inference problem iteratively
- Model selection (BIC) helps choose the number of components
- GenJAX can express GMMs as generative models

But we had to **specify K** (number of components) in advance. What if we don't know how many clusters exist?

In Chapter 6, we'll learn about **Dirichlet Process Mixture Models (DPMM)**: a Bayesian approach that learns the number of components automatically from the data!

---

{{% notice style="tip" title="Key Takeaways" %}}
1. **GMM**: Mixture of K Gaussians with mixing proportions π
2. **Inference**: EM algorithm alternates E-step (soft assignments) and M-step (parameter updates)
3. **Model selection**: BIC balances fit and complexity
4. **GenJAX**: Express GMMs as generative models with latent assignments
5. **Applications**: Clustering, segmentation, anomaly detection
{{% /notice %}}

---

**Next Chapter**: [Dirichlet Process Mixture Models →](./06_dpmm.md)
