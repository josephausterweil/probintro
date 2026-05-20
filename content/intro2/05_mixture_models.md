+++
date = "2026-05-20"
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

**📝 Assignment**: [Open in Colab: `solution_2_gaussian_clusters.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/solution_2_gaussian_clusters.ipynb)

**📓 Interactive exploration**: [Open in Colab: `gaussian_bayesian_interactive_exploration.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/gaussian_bayesian_interactive_exploration.ipynb) (Part 2)

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
import jax.random as random
from genjax import gen, flip, normal

@gen
def bento_mixture_model():
    """Two-component Gaussian mixture for Chibany's mystery bentos."""

    # Choose component with flip(0.7): True = tonkatsu (70%), False = hamburger (30%).
    # We use flip() because the choice is binary; flip(p) takes a probability directly.
    is_tonkatsu = flip(0.7) @ "component"

    # Pick the chosen component's parameters. jnp.where() selects without an if/else,
    # which keeps the model JAX-traceable.
    mu = jnp.where(is_tonkatsu, 500.0, 350.0)
    sigma = jnp.where(is_tonkatsu, 2.0, 2.0)   # both stds are 2.0 here

    # Generate the weight from the chosen component's Gaussian.
    weight = normal(mu, sigma) @ "weight"

    return weight, is_tonkatsu

# Simulate 20 bentos. GenJAX runs a model with model.simulate(key, args);
# here args is the empty tuple () because bento_mixture_model takes no arguments.
key = random.PRNGKey(42)
keys = random.split(key, 20)

# jax.vmap runs simulate once per key, in parallel.
traces = jax.vmap(lambda k: bento_mixture_model.simulate(k, ()))(keys)
weights, is_tonkatsu = traces.get_retval()

n_tonkatsu = jnp.sum(is_tonkatsu)
n_hamburger = jnp.sum(~is_tonkatsu)

print(f"Generated {n_tonkatsu} tonkatsu and {n_hamburger} hamburger bentos")
print(f"Weights: {weights}")
```

**Output (your numbers will differ — this is random):**
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

Now let's implement a Bayesian version using GenJAX, where we treat both the
component means and each observation's assignment as latent variables to infer.

To keep the focus on the *mixture* structure, we treat the two standard
deviations as **known** (σ = 2 for each component) and learn only the means
and the assignments. Learning the variances too is a straightforward extension
(add a prior for each), but fixing them keeps this first model readable.

```python
import jax
import jax.numpy as jnp
import jax.random as random
from genjax import gen, flip, normal, ChoiceMap

# Mystery bento weights from earlier.
mystery_weights = jnp.array([
    498.0, 352.0, 501.0, 349.0, 497.0, 503.0, 351.0, 500.0, 348.0, 502.0,
    499.0, 350.0, 498.0, 353.0, 501.0, 347.0, 499.0, 502.0, 352.0, 500.0
])

# The number of observations is a *structural* constant of the model — the @gen
# function below loops `range(N_OBS)` to lay out one assignment + one observation
# per data point. It must be a plain Python int (not a traced model argument),
# because JAX cannot trace through `range()` of a traced value. We close over it.
N_OBS = len(mystery_weights)
SIGMA_KNOWN = 2.0  # known standard deviation, shared by both components

@gen
def bayesian_gmm():
    """Bayesian 2-component Gaussian Mixture Model.

    Latents: two component means (mu_0, mu_1) and one assignment per observation.
    The standard deviation is the fixed constant SIGMA_KNOWN. The number of
    observations is the fixed constant N_OBS — both are closed over, not passed
    as arguments, so the loop length is known at trace time.
    """
    # Priors on the two component means (vague Normal priors).
    mu_0 = normal(400.0, 50.0) @ "mu_0"
    mu_1 = normal(400.0, 50.0) @ "mu_1"

    # Generate each observation: pick a component, then sample from its Gaussian.
    for i in range(N_OBS):
        # Assignment for observation i: flip(0.5) — True = component 1, False = component 0.
        z_i = flip(0.5) @ f"z_{i}"

        # Pull the assigned component's mean with jnp.where (keeps the model traceable).
        mu_i = jnp.where(z_i, mu_1, mu_0)

        # The observation itself.
        x_i = normal(mu_i, SIGMA_KNOWN) @ f"x_{i}"

    return mu_0, mu_1

# Condition on the observed weights by building a ChoiceMap that fixes each "x_i".
# ChoiceMap.d({...}) builds a choice map from a plain Python dict.
observations = ChoiceMap.d({
    f"x_{i}": mystery_weights[i] for i in range(N_OBS)
})

# generate() runs the model with those choices forced, and returns a trace plus
# a log-importance-weight. Running it for many keys gives weighted posterior samples.
# The model takes no arguments, so the args tuple is empty: ().
key = random.PRNGKey(42)
num_particles = 1000
keys = random.split(key, num_particles)

def one_particle(k):
    trace, log_weight = bayesian_gmm.generate(k, observations, ())
    choices = trace.get_choices()
    return choices["mu_0"], choices["mu_1"], log_weight

mu_0_samples, mu_1_samples, log_weights = jax.vmap(one_particle)(keys)

# Convert log-weights to normalized weights (log-sum-exp trick for stability).
weights = jnp.exp(log_weights - jnp.max(log_weights))
weights = weights / jnp.sum(weights)

# Weighted posterior means.
post_mu_0 = jnp.sum(weights * mu_0_samples)
post_mu_1 = jnp.sum(weights * mu_1_samples)

print(f"Posterior mean for mu_0: {post_mu_0:.1f}")
print(f"Posterior mean for mu_1: {post_mu_1:.1f}")
```

**Note**: This is *importance sampling* — the simplest inference method, and
deliberately so. Don't expect the printed posterior means to land neatly on 350
and 500: with 20 observations and a vague prior, most randomly-drawn particles
explain the data poorly and get a near-zero weight, so the weighted average is
dragged toward the prior and the estimate is noisy. That is not a bug — it is
*exactly why* importance sampling alone is not enough. Real GMM inference uses
smarter algorithms (EM, MCMC, variational methods) that we'll meet in later
chapters; they concentrate computation on the parameter settings that actually
fit the data. The Bayesian framing becomes especially powerful for the DPMM
(Chapter 6), where even the *number* of components is uncertain.

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
import jax
import jax.numpy as jnp
from genjax import gen, categorical, normal

# Coffee caffeine data.
coffee_data = jnp.array([
    82.0, 118.0, 155.0, 80.0, 120.0, 158.0, 79.0, 115.0, 160.0, 83.0,
    121.0, 157.0, 81.0, 119.0, 156.0, 84.0, 117.0, 159.0, 78.0, 122.0,
    154.0, 82.0, 116.0, 158.0, 80.0, 120.0, 155.0, 81.0, 118.0, 157.0
])

# Structural constants — closed over by the @gen function, NOT passed as
# arguments, so the loop length is a concrete int at trace time. (See the
# bento GMM above for why this matters.)
COFFEE_N_OBS = len(coffee_data)
COFFEE_SIGMA = 5.0  # known standard deviation, shared by all components

@gen
def coffee_gmm():
    """3-component GMM for coffee blends.

    a) Extends the 2-component model to K=3 by using categorical() for the
       component choice instead of flip().
    Latents: three component means; one assignment per observation.
    The standard deviation is fixed (COFFEE_SIGMA) — same simplification as the
    bento GMM above; learning the variances is a straightforward extension.
    """
    K = 3

    # Equal mixing proportions, fixed. categorical(probs) takes probabilities
    # directly (not log-probabilities).
    mixing_probs = jnp.full(K, 1.0 / K)

    # Priors on the three means — centered on the expected low/medium/high range.
    mu_0 = normal(80.0, 20.0) @ "mu_0"    # Low caffeine
    mu_1 = normal(120.0, 20.0) @ "mu_1"   # Medium caffeine
    mu_2 = normal(160.0, 20.0) @ "mu_2"   # High caffeine
    means = jnp.array([mu_0, mu_1, mu_2])

    # Generate observations with component assignments.
    for i in range(COFFEE_N_OBS):
        z_i = categorical(mixing_probs) @ f"z_{i}"   # 0, 1, or 2
        mu_i = means[z_i]
        x_i = normal(mu_i, COFFEE_SIGMA) @ f"x_{i}"

    return mu_0, mu_1, mu_2

# Conditioning + importance sampling follows the same pattern as the bento GMM:
# build a ChoiceMap fixing each "x_i" to coffee_data[i], then call
# coffee_gmm.generate(key, observations, ())  — the model takes no arguments.

# b) The priors above use Normal(expected_mean, 20.0), which allows reasonable
#    variation while keeping the means inside the plausible 50-200 mg range.

# c) The posterior over each z_i tells us the probability that cup i belongs to
#    each blend, accounting for uncertainty in both the assignments and the means.
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
