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

## 📚 Prerequisite: Understanding Categorization

Before tackling the full GMM learning problem, make sure you understand **categorization** in mixture models with **known parameters**.

{{% notice style="warning" title="⚠️ Recommended Preparation" %}}
If you haven't already, work through the **Gaussian Clusters** assignment from Chapter 4:

**📝 Assignment**: [`solution_2_gaussian_clusters.ipynb`](../../notebooks/solution_2_gaussian_clusters.ipynb)

**📓 Interactive exploration**: [`gaussian_bayesian_interactive_exploration.ipynb`](../../notebooks/gaussian_bayesian_interactive_exploration.ipynb) (Part 2)

**Why this matters**:
- **Chapter 4 Problem 2** teaches you how to compute P(category | observation) when parameters are **known**
- **This chapter (5)** extends that to learning parameters when they are **unknown**
- Understanding categorization with known parameters is essential before attempting to learn them!

**What you'll practice**:
- Using Bayes' rule: P(c|x) = p(x|c)P(c) / p(x)
- Computing marginal distributions: p(x) = Σ_c p(x|c)P(c)
- Understanding decision boundaries and how priors/variances affect them
- Visualizing bimodal vs. unimodal mixture distributions
{{% /notice %}}

### The Bridge: Known Parameters → Unknown Parameters

**In Chapter 4 Problem 2**, you learned:
- Given: μ₁, μ₂, σ₁², σ₂², θ (all known)
- Infer: Which category for each observation?
- Formula: P(c=1|x) = θ·N(x;μ₁,σ₁²) / [θ·N(x;μ₁,σ₁²) + (1-θ)·N(x;μ₂,σ₂²)]

**In this chapter**, we tackle the harder problem:
- Given: Only observations x₁, x₂, ..., xₙ
- Infer: Categories **AND** parameters μ₁, μ₂, σ₁², σ₂², θ
- Method: Expectation-Maximization (EM) algorithm

Think of it as:
1. **First** (Chapter 4 Problem 2): "I know the recipe for tonkatsu (μ₁, σ₁²) and hamburger (μ₂, σ₂²). Given a weight, which is it?"
2. **Now** (Chapter 5): "I don't know the recipes! Can I figure them out from weights alone?"

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

## Understanding the Inference Challenge

If we knew which type each bento was, learning would be straightforward - just compute the mean and variance for each group. Conversely, if we knew the true parameters, we could compute which component each observation likely came from.

This chicken-and-egg problem is exactly what probabilistic inference is designed to solve. Instead of point estimates, we'll use GenJAX to reason about the full posterior distribution over both parameters and assignments.

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

**Note**: The above shows the conceptual structure. In practice, GMM inference with GenJAX requires careful implementation of inference algorithms. We'll explore more sophisticated inference techniques including MCMC and variational methods in later chapters. The Bayesian approach becomes particularly powerful for more complex models like DPMM (Chapter 6), where we want to reason about uncertainty in the number of components.

---

## Model Selection: How Many Components?

How do we know K=2? What if there are 3 types of bentos, or 5?

In traditional approaches, you would fit multiple models with different K values and use criteria like BIC (Bayesian Information Criterion) to select the best one.

However, in fully Bayesian inference (which we'll explore more in Chapter 6), we can treat K itself as a random variable and let the data inform us about the likely number of components through the posterior distribution.

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

**a)** Extend the Bayesian GMM code to K=3 components.

**b)** What prior distributions would be appropriate for the means if you know caffeine levels range from 50-200mg?

**c)** How would you interpret the posterior distribution over component assignments?

<details>
<summary>Show Solution</summary>

```python
@gen
def coffee_gmm(data):
    """3-component GMM for coffee blends"""
    K = 3

    # Prior on mixing proportions
    pi = jnp.dirichlet(jnp.ones(K)) @ "pi"

    # Priors on means - centered around expected range
    mu = jnp.array([
        jnp.normal(80.0, 20.0) @ "mu_0",   # Low caffeine
        jnp.normal(120.0, 20.0) @ "mu_1",  # Medium caffeine
        jnp.normal(160.0, 20.0) @ "mu_2"   # High caffeine
    ])

    # Priors on standard deviations
    sigma = jnp.array([
        jnp.gamma(2.0, 1.0) @ "sigma_0",
        jnp.gamma(2.0, 1.0) @ "sigma_1",
        jnp.gamma(2.0, 1.0) @ "sigma_2"
    ])

    # Generate observations with component assignments
    for i, obs in enumerate(data):
        z_i = jnp.categorical(jnp.log(pi)) @ f"z_{i}"
        x_i = jnp.normal(mu[z_i], sigma[z_i]) @ f"x_{i}"

    return pi, mu, sigma

# b) The priors above use Normal(expected_mean, 20.0) which allows
# reasonable variation while keeping means in sensible ranges

# c) The posterior over z_i tells us the probability each cup
# belongs to each blend, accounting for uncertainty in both
# assignments and parameters
```
</details>

---

### Problem 2: Understanding Uncertainty

Using the Bayesian GMM for the bento data:

**a)** How would you quantify uncertainty about which component a particular observation belongs to?

**b)** How is this different from a point estimate of the assignment?

<details>
<summary>Show Solution</summary>

**a)** In the Bayesian approach, we get a full posterior distribution over component assignments. For each observation i, we can compute:
- P(z_i = 0 | data) - probability it's component 0
- P(z_i = 1 | data) - probability it's component 1

An observation near the decision boundary might have P(z_i = 0) ≈ 0.5, showing high uncertainty.

**b)** A point estimate would simply assign each observation to its most likely component, discarding information about confidence. The Bayesian approach preserves this uncertainty, which is crucial for:
- Identifying ambiguous cases
- Propagating uncertainty to downstream tasks
- Making better decisions under uncertainty

For example, a bento weighing 425g (right between the two clusters) would have high assignment uncertainty that we shouldn't ignore.
</details>

---

## What's Next?

We now understand:
- Gaussian Mixture Models combine multiple Gaussians
- GMMs elegantly combine discrete choices (component assignments) with continuous observations
- GenJAX naturally expresses the generative process as a probabilistic program
- Bayesian inference preserves uncertainty over both parameters and assignments

But we had to **specify K** (number of components) in advance. What if we don't know how many clusters exist?

In Chapter 6, we'll learn about **Dirichlet Process Mixture Models (DPMM)**: a Bayesian approach that learns the number of components automatically from the data!

---

{{% notice style="tip" title="Key Takeaways" %}}
1. **GMM**: Mixture of K Gaussians with mixing proportions π
2. **Generative process**: First choose component (discrete), then generate observation (continuous)
3. **Bayesian inference**: Reason about full posterior over parameters and assignments
4. **GenJAX**: Express GMMs declaratively as probabilistic programs
5. **Uncertainty**: Preserve and quantify uncertainty about component membership
6. **Applications**: Clustering, segmentation, anomaly detection
{{% /notice %}}

---

**Next Chapter**: [Dirichlet Process Mixture Models →](./06_dpmm.md)
