+++
title = "Glossary - All Tutorials"
weight = 100
+++

## How to Use This Glossary

This glossary covers all three tutorials in the Probability with GenJAX series. Terms are tagged to show which tutorial introduces them:

- 📘 **Tutorial 1** (Discrete Probability) - Sets and counting approach
- 💻 **Tutorial 2** (GenJAX Programming) - Probabilistic programming basics
- 📊 **Tutorial 3** (Continuous Probability) - Advanced topics and Bayesian learning

Click on any term to expand its definition with examples and code.

---

## Core Concepts (Tutorial 1)

### Bayes Theorem 📘
{{% expand "Bayes Theorem" %}}
*Bayes Theorem* (or Bayes' rule) is a formula for reversing the order that variables are conditioned — how to go from $P(A \mid B)$ to $P(B \mid A)$.

**Formula:** $P(H \mid D) = \frac{P(D \mid H) P(H)}{P(D)}$

**Components:**
- $P(H \mid D)$ = posterior (updated belief after seeing data)
- $P(D \mid H)$ = likelihood (how well data fits hypothesis)
- $P(H)$ = prior (belief before seeing data)
- $P(D)$ = evidence (total probability of data)

**Application:** Updating beliefs with new information

**See also**: Prior, Posterior, Likelihood
{{% /expand %}}

### Cardinality 📘
{{% expand "Cardinality" %}}
The *cardinality* or *size* of a set is the number of elements it contains. If $A = \\{H, T\\}$, then the cardinality of $A$ is $|A|=2$.

**Notation:** $|A|$ means "the size of set $A$"

**In programming**: This is like `len(A)` in Python or counting array elements
{{% /expand %}}

### Conditional Probability 📘
{{% expand "Conditional Probability" %}}
The *conditional probability* is the probability of an event conditioned on knowledge of another event. Conditioning on an event means that the possible outcomes in that event form the set of possibilities or outcome space. We then calculate probabilities as normal within that *restricted* outcome space.

**Formally:** $P(A \mid B) = \frac{|A \cap B|}{|B|}$, where everything to the left of the $\mid$ is what we're interested in knowing the probability of and everything to the right of the $\mid$ is what we know to be true.

**Alternative formula:** $P(A \mid B) = \frac{P(A,B)}{P(B)}$ (assuming $P(B) > 0$)

**In GenJAX 💻**: We condition using `ChoiceMap` to specify observed values
{{% /expand %}}

### Dependence 📘
{{% expand "Dependence" %}}
When knowing the outcome of one random variable or event influences the probability of another, those variables or events are called *dependent*. This is denoted as $A \not\perp B$.

When they do not influence each other, they are called *independent*. This is denoted as $A \perp B$.

**Formal definition of independence:** $P(A \mid B) = P(A)$, or equivalently, $P(A, B) = P(A) \times P(B)$

**Example**: Coin flips are independent (one doesn't affect the next). Drawing cards without replacement is dependent (first draw affects second).
{{% /expand %}}

### Event 📘
{{% expand "Event" %}}
An *event* is a set that contains none, some, or all of the possible outcomes. In other words, an event is any subset of the outcome space $\Omega$.

**Example:** "At least one tonkatsu" is the event $\\{HT, TH, TT\\} \subseteq \Omega$.

**In programming**: Events correspond to filtering/counting samples that satisfy a condition
{{% /expand %}}

### Generative Process 📘💻
{{% expand "Generative Process" %}}
A *generative process* defines the probabilities for possible outcomes according to an algorithm with random choices. Think of it as a recipe for producing outcomes.

**Example:** "Flip two coins: first for lunch (H or T), second for dinner (H or T). Record the pair."

**In GenJAX 💻**: We write generative processes as `@gen` decorated functions

```python
@gen
def chibany_day():
    lunch = flip(0.5) @ "lunch"
    dinner = flip(0.5) @ "dinner"
    return (lunch, dinner)
```

This connects probabilistic thinking to actual executable code!
{{% /expand %}}

### Joint Probability 📘
{{% expand "Joint Probability" %}}
The *joint probability* is the probability that multiple events all occur. This corresponds to the intersection of the events (outcomes that are in all the events).

**Notation:** $P(A, B)$ or $P(A \cap B)$

**Intuition:** "What's the probability that both $A$ and $B$ happen?"

**Example**: $P(\text{lunch}=T, \text{dinner}=T) = P(TT)$
{{% /expand %}}

### Marginal Probability 📘
{{% expand "Marginal Probability" %}}
A *marginal probability* is the probability of a random variable that has been calculated by summing over the possible values of one or more other random variables.

**Formula:** $P(A) = \sum_{b} P(A, B=b)$

**Intuition:** "What's the probability of $A$ regardless of what $B$ is?"

**Example**: $P(\text{lunch}=T) = P(TH) + P(TT)$ (tonkatsu for lunch, regardless of dinner)
{{% /expand %}}

### Outcome Space 📘
{{% expand "Outcome Space" %}}
The *outcome space* (denoted $\Omega$, the Greek letter omega) is the set of all possible outcomes for a random process. It forms the foundation for calculating probabilities.

**Example:** For Chibany's two daily meals, $\Omega = \\{HH, HT, TH, TT\\}$.

**In GenJAX 💻**: We generate outcomes from the outcome space by running `simulate()` many times
{{% /expand %}}

### Probability 📘
{{% expand "Probability" %}}
The *probability* of an event $A$ relative to an outcome space $\Omega$ is the ratio of their sizes: $P(A) = \frac{|A|}{|\Omega|}$.

When outcomes are weighted (not equally likely), we sum the weights instead of counting.

**Interpretation:** "What fraction of possible outcomes are in event $A$?"

**In code**: We approximate this by simulation: run the process many times and compute the fraction of runs where the event occurs.
{{% /expand %}}

### Random Variable 📘
{{% expand "Random Variable" %}}
A *random variable* is a function that maps from the set of possible outcomes to some set or space. The output or range of the function could be the set of outcomes again, a whole number based on the outcome (e.g., counting the number of Tonkatsu), or something more complex.

Technically the output must be *measurable*. You shouldn't worry about that distinction unless your random variable's output gets really, really big (like continuous). We'll talk more about probabilities over continuous random variables in Tutorial 3 📊.

**Key insight:** It's called "random" because its value depends on which outcome occurs, but it's really just a function!

**Example**: $X(\omega)$ = number of tonkatsu meals in outcome $\omega$
{{% /expand %}}

### Set 📘
{{% expand "Set" %}}
A *set* is a collection of elements or members. Sets are defined by the elements they do or do not contain. The elements are listed with commas between them and "$\\{$" denotes the start of a set and "$\\}$" the end of a set. Note that the elements of a set are unique.

**Example:** $\\{H, T\\}$ is a set containing two elements: H and T.

**In programming**: Like a Python set `{0, 1}` or a list of unique elements
{{% /expand %}}

---

## GenJAX Programming (Tutorial 2)

### @gen Decorator 💻
{{% expand "@gen Decorator" %}}
The `@gen` decorator in GenJAX marks a Python function as a *generative function* that can make addressed random choices and be used for probabilistic inference.

**Usage**:
```python
@gen
def my_model():
    x = bernoulli(0.5) @ "x"  # Random choice at address "x"
    return x
```

**What it does**:
- Tracks all random choices made
- Allows conditioning on observations
- Enables inference (importance sampling, MCMC, etc.)

**See also**: Generative Function, Trace, ChoiceMap
{{% /expand %}}

### Bernoulli Distribution 💻
{{% expand "Bernoulli Distribution" %}}
A probability distribution representing a single binary trial (success/failure, 1/0, true/false). Named after mathematician Jacob Bernoulli.

**Parameter**: $p$ = probability of success (returning 1)

**What it represents**: A single yes/no outcome. Think of it as a biased coin flip where the coin comes up heads with probability $p$ and tails with probability $1-p$.

**In GenJAX**:
```python
@gen
def coin_flip():
    is_heads = flip(0.5) @ "coin"  # 50% chance of 1 (heads)
    return is_heads
```

**Note**: In GenJAX, we use `flip(p)` instead of `bernoulli(p)` — the name reflects the coin flip metaphor!

**Returns**: `True`/`1` (success) or `False`/`0` (failure)

**Example uses**: Coin flips, yes/no questions, on/off states, binary decisions

**See also**: flip(), Categorical distribution (generalization to multiple outcomes)
{{% /expand %}}

### flip() 💻
{{% expand "flip()" %}}
GenJAX's function for sampling from a Bernoulli distribution. The name reflects the coin flip metaphor.

**Signature**: `flip(p)`

**Parameter**:
- `p` - probability of returning `True`/`1` (like getting heads)

**Returns**: `True` or `False` (represented as `1` or `0` in JAX arrays)

**In GenJAX**:
```python
@gen
def coin_flip():
    result = flip(0.7) @ "coin"  # 70% chance of True (heads)
    return result
```

**Common values**:
- `flip(0.5)` - Fair coin flip (50/50)
- `flip(0.8)` - Biased toward True (80% chance)
- `flip(0.2)` - Biased toward False (80% chance of False)

**Why "flip" instead of "bernoulli"?** GenJAX has both functions, but they take different arguments:
- `flip(p)` - takes a **probability** (0 to 1) - more intuitive for most users
- `bernoulli(logit)` - takes a **logit** (log-odds, -∞ to +∞) - inherited from TensorFlow conventions

Most users should use `flip()` as it works the way you'd expect from probability theory (pass in 0.7 for 70% chance of true).

**See also**: Bernoulli Distribution
{{% /expand %}}

### Categorical Distribution 💻📊
{{% expand "Categorical Distribution" %}}
Probability distribution over discrete outcomes with specified probabilities.

**Parameters**: Array of probabilities that sum to 1.0

**In GenJAX**:
```python
@gen
def roll_die(probs):
    outcome = categorical(probs) @ "roll"  # Returns 0,1,2,3,4, or 5
    return outcome
```

**Example**: `categorical([0.25, 0.25, 0.25, 0.25])` for fair 4-sided die

**Returns**: Integer index (0, 1, 2, ..., k-1)

**Connection to Tutorial 1 📘**: Generalizes the discrete outcome spaces you learned with sets

**Used in 📊**: Cluster assignment in mixture models, DPMM
{{% /expand %}}

### ChoiceMap 💻
{{% expand "ChoiceMap" %}}
GenJAX's way of specifying observed values for random choices. A dictionary-like structure that maps addresses (names) to values.

**Used for**:
- Recording what random choices were made (from traces)
- Specifying observations for inference
- Constraining random choices

**In code**:
```python
from genjax import ChoiceMap

# Observe x=2.5
observations = ChoiceMap.d({"x": 2.5})

# Or use builder pattern
cm = ChoiceMap.empty()
cm = cm.set("x", 2.5)
cm = cm.set("y", 1.0)
```

**Think of it as**: A way to name and track all the random decisions

**See also**: Trace, Target
{{% /expand %}}

### Generative Function 💻
{{% expand "Generative Function" %}}
In GenJAX, a generative function is a Python function decorated with `@gen` that can make addressed random choices. It represents a probability distribution over its return values.

**Structure**:
```python
@gen
def model(params):
    # Random choices with addresses
    x = distribution(params) @ "address"
    y = another_distribution(x) @ "another_address"
    return result
```

**Key features**:
- Makes random choices at named addresses
- Can condition on observations
- Supports inference operations

**See also**: @gen decorator, Trace, ChoiceMap
{{% /expand %}}

### Importance Sampling 💻📊
{{% expand "Importance Sampling" %}}
An inference method that approximates the posterior distribution by:
1. Generating samples from a proposal distribution
2. Weighting each sample by how well it matches observations
3. Using weighted samples to approximate the posterior

**In GenJAX**:
```python
trace, log_weight = target.importance(key, choicemap)
```

**Key concept**: Effective sample size (ESS) measures how well the weights are distributed. ESS close to the number of samples is good; ESS of 1 means only one sample has meaningful weight (bad).

**Formula**: $\text{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2}$

**Used in 📊**: Posterior inference for Bayesian models, DPMM

**See also**: Target, Weight degeneracy
{{% /expand %}}

### JAX Key 💻
{{% expand "JAX Key" %}}
JAX uses explicit random keys to control randomness (unlike NumPy's global random state). Think of it like a seed that you explicitly pass around.

**Why**: Enables reproducibility and functional programming patterns

**Usage**:
```python
import jax

# Create a key
key = jax.random.key(42)  # 42 is the seed

# Split into multiple keys
keys = jax.random.split(key, num=100)  # Get 100 independent keys

# Use a key
trace = model.simulate(keys[0], ())
```

**Best practice**: Always split keys, never reuse the same key twice

**See also**: vmap (often used together)
{{% /expand %}}

### Monte Carlo Simulation 📘💻
{{% expand "Monte Carlo Simulation" %}}
A computational method for approximating probabilities by generating many random samples and counting outcomes. Named after the Monte Carlo casino.

**Process:**
1. Generate many random outcomes (e.g., 10,000 simulated days)
2. Count how many satisfy your event
3. Calculate the ratio

**In GenJAX**:
```python
# Generate 10,000 samples
keys = jax.random.split(key, 10000)
samples = jax.vmap(lambda k: model.simulate(k, ()).get_retval())(keys)

# Count event occurrences
event_count = jnp.sum(samples >= threshold)
probability = event_count / 10000
```

**When useful:** When outcome spaces are too large to enumerate by hand

**See also**: vmap, Trace
{{% /expand %}}

### Normal Distribution 💻📊
{{% expand "Normal Distribution" %}}
See **Gaussian Distribution** (same thing)
{{% /expand %}}

### simulate() 💻
{{% expand "simulate()" %}}
The `simulate()` method generates one random execution of a generative function.

**Signature**:
```python
trace = model.simulate(key, args)
```

**Parameters**:
- `key`: JAX random key
- `args`: Tuple of arguments to the generative function
- Optional: `observations` (ChoiceMap) to condition on

**Returns**: A trace containing all random choices and the return value

**Example**:
```python
@gen
def coin_flip():
    return bernoulli(0.5) @ "flip"

trace = coin_flip.simulate(key, ())
result = trace.get_retval()  # 0 or 1
```

**See also**: Trace, importance(), JAX Key
{{% /expand %}}

### Target 💻
{{% expand "Target" %}}
In GenJAX, a `Target` is created by conditioning a generative function on observations. It represents the posterior distribution.

**Creating a target**:
```python
from genjax import Target

# Observe some data
observations = ChoiceMap.d({"x_0": 2.5, "x_1": 3.0})

# Create target (posterior)
target = Target(model, (params,), observations)
```

**Using for inference**:
```python
# Importance sampling
trace, log_weight = target.importance(key, ChoiceMap.empty())
```

**Key concept**: The target represents $P(\text{latent variables} \mid \text{observations})$

**See also**: ChoiceMap, Importance Sampling, Posterior
{{% /expand %}}

### Trace 💻
{{% expand "Trace" %}}
In probabilistic programming, a *trace* records all random choices made during one execution of a generative function, along with their addresses (names) and the return value.

**Think of it as:** A complete record of "what happened" during one run of a probabilistic program

**Structure**:
```python
trace = model.simulate(key, args)

# Access components
retval = trace.get_retval()         # Return value
choices = trace.get_choices()        # ChoiceMap with all random choices
log_prob = trace.get_score()         # Log probability of this trace
```

**Example**:
```python
@gen
def example():
    x = flip(0.5) @ "x"
    y = normal(0, 1) @ "y"
    return x + y

trace = example.simulate(key, ())
print(trace.get_choices()["x"])  # e.g., True or False
print(trace.get_choices()["y"])  # e.g., 0.234
print(trace.get_retval())        # e.g., 1.234
```

**Used in:** GenJAX and other probabilistic programming systems

**See also**: ChoiceMap, Generative Function
{{% /expand %}}

### vmap 💻
{{% expand "vmap" %}}
JAX's "vectorized map" - applies a function to many inputs in parallel (very fast!).

**Concept**: Instead of a for-loop running sequentially, vmap runs operations in parallel on the GPU/CPU.

**Usage**:
```python
import jax

# Regular loop (slow)
results = []
for key in keys:
    results.append(model.simulate(key, ()).get_retval())

# vmap (fast!)
def run_once(key):
    return model.simulate(key, ()).get_retval()

results = jax.vmap(run_once)(keys)
```

**Think of it as**: "Do this function 10,000 times, but do them all at once"

**Why it's fast**: Leverages parallel hardware (GPU, vectorized CPU operations)

**See also**: JAX Key, Monte Carlo Simulation
{{% /expand %}}

---

## Continuous Probability (Tutorial 3)

### Beta Distribution 📊
{{% expand "Beta Distribution" %}}
A continuous probability distribution on the interval [0,1], parameterized by two shape parameters $\alpha$ and $\beta$.

**Parameters**:
- $\alpha$ (alpha) - shape parameter
- $\beta$ (beta) - shape parameter

**PDF**: $p(x \mid \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}$

**In GenJAX**:
```python
@gen
def stick_breaking(alpha):
    # Beta(1, alpha) for stick-breaking
    beta_k = beta(1.0, alpha) @ f"beta_{k}"
    return beta_k
```

**Special cases**:
- Beta(1,1) = Uniform(0,1)
- Beta(α,α) is symmetric around 0.5

**Used in 📊**:
- Stick-breaking construction for Dirichlet Process
- Modeling probabilities and proportions
- Conjugate prior for Bernoulli/Binomial

**See also**: Dirichlet distribution, Stick-breaking
{{% /expand %}}

### Chinese Restaurant Process (CRP) 📊
{{% expand "Chinese Restaurant Process" %}}
A metaphor and algorithm for understanding the Dirichlet Process. Imagine customers entering a restaurant with infinite tables:
- First customer sits at table 1
- Next customer: sit at an occupied table with probability proportional to its occupancy, OR sit at a new table with probability proportional to α

**Parameters**: α (concentration parameter)

**Properties**:
- "Rich get richer" - popular tables attract more customers
- But always a chance to start new tables
- α controls tendency to create new clusters

**Connection to DPMM**: Each table = a cluster. CRP determines cluster assignments, then each cluster has its own Gaussian distribution.

**Not used directly in code**: Stick-breaking construction is mathematically equivalent but more practical for implementation

**See also**: Dirichlet Process, DPMM, Stick-breaking
{{% /expand %}}

### Concentration Parameter (α) 📊
{{% expand "Concentration Parameter (α)" %}}
The parameter α in the Dirichlet Process and related models controls the tendency to create new clusters vs. reusing existing ones.

**Effect**:
- **Small α** (e.g., 0.1): Few clusters, strong preference for existing clusters
- **Medium α** (e.g., 1-5): Balanced exploration/exploitation
- **Large α** (e.g., 10+): Many clusters, high probability of creating new ones

**In stick-breaking**:
```python
beta_k = beta(1.0, alpha) @ f"beta_{k}"
```

**Intuition**: α is like a "prior strength" for new clusters. Higher α = more willing to explain data with new clusters rather than fitting to existing ones.

**Typical range**: 0.1 to 10 for most applications

**See also**: Dirichlet Process, DPMM, Stick-breaking
{{% /expand %}}

### Conjugate Prior 📊
{{% expand "Conjugate Prior" %}}
A prior distribution is *conjugate* to a likelihood when the posterior distribution is in the same family as the prior.

**Why useful**: Enables closed-form posterior calculation (no need for sampling)

**Classic examples**:
- **Beta-Binomial**: Beta prior × Binomial likelihood = Beta posterior
- **Gamma-Poisson**: Gamma prior × Poisson likelihood = Gamma posterior
- **Gaussian-Gaussian**: Normal prior × Normal likelihood = Normal posterior

**Example (Gaussian-Gaussian)**:
```python
# Prior: μ ~ Normal(μ₀, σ₀²)
# Likelihood: x | μ ~ Normal(μ, σ²)
# Posterior: μ | x ~ Normal(μ_post, σ_post²)  # Still Gaussian!

# Posterior parameters:
# μ_post = (σ²·μ₀ + σ₀²·x) / (σ² + σ₀²)
# σ_post² = (σ²·σ₀²) / (σ² + σ₀²)
```

**Trade-off**: Mathematical convenience vs. modeling flexibility

**Tutorial 3, Chapter 4** covers Gaussian-Gaussian conjugacy in detail

**See also**: Prior, Posterior, Bayesian Learning
{{% /expand %}}

### Cumulative Distribution Function (CDF) 📊
{{% expand "Cumulative Distribution Function (CDF)" %}}
For a continuous random variable, the CDF gives the probability that the variable is less than or equal to a value:

$$F(x) = P(X \leq x) = \int_{-\infty}^x p(t) \, dt$$

**Key properties**:
- Always increasing (or flat)
- Ranges from 0 to 1
- $F(-\infty) = 0$ and $F(\infty) = 1$
- Derivative of CDF = PDF: $\frac{dF}{dx} = p(x)$

**Interpretation**: "What's the probability of getting a value this small or smaller?"

**Example (Standard Normal)**:
- CDF(0) ≈ 0.5 (50% chance of being ≤ 0)
- CDF(1.96) ≈ 0.975 (97.5% chance of being ≤ 1.96)

**In code**: Usually not needed directly in GenJAX (we sample instead), but useful for understanding quantiles and probabilities

**See also**: PDF, Quantile
{{% /expand %}}

### Dirichlet Distribution 📊
{{% expand "Dirichlet Distribution" %}}
The multivariate generalization of the Beta distribution. Produces probability vectors that sum to 1.

**Parameters**: α = (α₁, α₂, ..., αₖ) - concentration parameters

**Output**: Vector (p₁, p₂, ..., pₖ) where all pᵢ > 0 and Σpᵢ = 1

**In GenJAX**:
```python
@gen
def mixture_weights(alpha_vector):
    # Returns a probability distribution over K categories
    probs = dirichlet(alpha_vector) @ "probs"
    return probs
```

**Special case**: Dirichlet(1,1,1,...,1) = Uniform over probability simplex

**Intuition**: Like rolling a weighted die where the weights themselves are random

**Used in**:
- Prior for mixture weights in GMM
- ~~DPMM~~ (not directly - stick-breaking is used instead)
- Topic modeling (LDA)

**See also**: Beta distribution, Categorical distribution
{{% /expand %}}

### Dirichlet Process (DP) 📊
{{% expand "Dirichlet Process" %}}
A distribution over distributions. It's a *prior* for mixture models when you don't know how many clusters/components you need.

**Parameters**:
- α (concentration parameter) - controls cluster formation
- G₀ (base distribution) - the "prototype" distribution for clusters

**Key properties**:
- **Infinite mixture**: Can have arbitrarily many clusters
- **Automatic model selection**: Data determines effective number of clusters
- **Clustering property**: Enforces that some samples share the same parameter values (clusters)

**Two representations**:
1. **Chinese Restaurant Process** (CRP) - metaphorical, sequential
2. **Stick-breaking** - constructive, practical for implementation

**Why "Dirichlet Process"**: It's a generalization of the Dirichlet distribution to infinite dimensions

**In practice**: Used via DPMM for clustering without specifying K

**Tutorial 3, Chapter 6** covers DP in detail

**See also**: DPMM, Stick-breaking, Chinese Restaurant Process
{{% /expand %}}

### Dirichlet Process Mixture Model (DPMM) 📊
{{% expand "Dirichlet Process Mixture Model (DPMM)" %}}
An infinite mixture model that automatically determines the number of clusters from data.

**Structure**:
```
1. Generate cluster parameters using stick-breaking:
   - β₁, β₂, ... ~ Beta(1, α)
   - π₁ = β₁, π₂ = β₂(1-β₁), π₃ = β₃(1-β₁)(1-β₂), ...

2. For each data point:
   - z ~ Categorical(π)  # Assign to cluster
   - x | z ~ Normal(μ_z, σ²)  # Generate from that cluster's Gaussian
```

**Parameters**:
- α - controls number of clusters
- μ₀, σ₀ - prior for cluster means
- σ - observation noise

**In GenJAX**:
```python
@gen
def dpmm(alpha, mu0, sig0, sigx):
    # Stick-breaking for mixture weights
    pis = stick_breaking_construction(alpha, K)

    # Cluster means
    mus = [normal(mu0, sig0) @ f"mu_{k}" for k in range(K)]

    # Assign data points and generate observations
    for i in range(N):
        z_i = categorical(pis) @ f"z_{i}"
        x_i = normal(mus[z_i], sigx) @ f"x_{i}"
```

**Advantages**:
- No need to specify K in advance
- Principled Bayesian uncertainty
- Automatic model complexity control

**Challenges**:
- Requires truncation (approximate with K clusters)
- Inference can be slow for large datasets
- Sensitive to α choice

**Tutorial 3, Chapter 6** has full implementation and interactive notebook

**See also**: GMM, Dirichlet Process, Stick-breaking
{{% /expand %}}

### Expected Value 📊
{{% expand "Expected Value" %}}
The average value of a random variable, weighted by probabilities. Also called the *mean* or *expectation*.

**For discrete**: $E[X] = \sum_{x} x \cdot P(X=x)$

**For continuous**: $E[X] = \int_{-\infty}^{\infty} x \cdot p(x) \, dx$

**In GenJAX** (approximation by sampling):
```python
# Generate many samples
samples = [model.simulate(key_i, ()).get_retval() for key_i in keys]

# Expected value ≈ average of samples
expected_value = jnp.mean(samples)
```

**Properties**:
- Linearity: $E[aX + bY] = aE[X] + bE[Y]$
- For independent variables: $E[XY] = E[X]E[Y]$

**Interpretation**: "If I repeated this experiment many times, what would the average outcome be?"

**Tutorial 3, Chapter 1** covers expected value with the "mystery bento" paradox

**See also**: Variance, Law of Iterated Expectation
{{% /expand %}}

### Gaussian Distribution 📊
{{% expand "Gaussian Distribution" %}}
Also called the *Normal distribution*. The famous bell curve, ubiquitous in statistics and machine learning.

**Parameters**:
- μ (mu) - mean (center of the bell)
- σ² (sigma squared) - variance (width of the bell)

**PDF**:
$$p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**In GenJAX**:
```python
@gen
def gaussian_model():
    x = normal(mu, sigma) @ "x"  # Note: sigma, not sigma²
    return x
```

**The 68-95-99.7 Rule**:
- 68% of data within μ ± σ
- 95% of data within μ ± 2σ
- 99.7% of data within μ ± 3σ

**Why so common**:
- Central Limit Theorem (sums converge to Gaussian)
- Maximum entropy distribution for given mean and variance
- Mathematically tractable (conjugate priors!)

**Tutorial 3, Chapter 3** covers Gaussians in detail

**See also**: Normal distribution (same thing), Standard Normal
{{% /expand %}}

### Gaussian Mixture Model (GMM) 📊
{{% expand "Gaussian Mixture Model (GMM)" %}}
A mixture of multiple Gaussian distributions, each with its own mean, variance, and mixing weight.

**Structure**:
```
1. Choose cluster k with probability πₖ
2. Sample from Normal(μₖ, σₖ²)
```

**Parameters**:
- K - number of components (must be specified)
- π₁, ..., πₖ - mixing weights (sum to 1)
- μ₁, ..., μₖ - component means
- σ₁², ..., σₖ² - component variances

**In GenJAX**:
```python
@gen
def gmm(pis, mus, sigmas):
    # Choose component
    z = categorical(pis) @ "z"

    # Sample from chosen component
    x = normal(mus[z], sigmas[z]) @ "x"
    return x
```

**Use cases**:
- Clustering data with multiple groups
- Modeling multimodal distributions
- Density estimation

**Limitation**: Must specify K in advance (DPMM fixes this!)

**Tutorial 3, Chapter 5** covers GMM

**See also**: DPMM, Mixture Model
{{% /expand %}}

### Likelihood 📊
{{% expand "Likelihood" %}}
The probability of observing the data given specific parameter values: $P(D \mid \theta)$

**Key distinction**:
- As a function of data (θ fixed): **Probability**
- As a function of parameters (data fixed): **Likelihood**

**In Bayes' Theorem**:
$$P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}$$
- $P(D \mid \theta)$ is the **likelihood**
- $P(\theta)$ is the **prior**
- $P(\theta \mid D)$ is the **posterior**

**Example**:
```python
# Observed data: x = [2.5, 3.0, 2.8]
# Model: x[i] ~ Normal(μ, 1.0)

# Likelihood of μ = 3.0:
likelihood = product([
    normal_pdf(2.5, mu=3.0, sigma=1.0),
    normal_pdf(3.0, mu=3.0, sigma=1.0),
    normal_pdf(2.8, mu=3.0, sigma=1.0)
])
```

**In GenJAX**: The trace log probability includes the likelihood

**See also**: Posterior, Prior, Bayes' Theorem
{{% /expand %}}

### Mixture Model 📊
{{% expand "Mixture Model" %}}
A probability model that combines multiple component distributions, each active with some probability.

**General form**:
$$p(x) = \sum_{k=1}^K \pi_k \cdot p_k(x)$$

where:
- πₖ = mixing weights (probabilities, sum to 1)
- pₖ(x) = component distributions

**Generative process**:
1. Choose component k with probability πₖ
2. Sample from component pₖ

**Common types**:
- **Gaussian Mixture Model (GMM)**: Components are Gaussians
- **DPMM**: Infinite mixture (K → ∞)

**Why useful**:
- Model complex, multimodal distributions
- Perform soft clustering
- Represent heterogeneous populations

**Tutorial 3, Chapter 5** covers finite mixtures (GMM)
**Tutorial 3, Chapter 6** covers infinite mixtures (DPMM)

**See also**: GMM, DPMM, Categorical distribution
{{% /expand %}}

### Posterior Distribution 📊
{{% expand "Posterior Distribution" %}}
The updated probability distribution over parameters after observing data: $P(\theta \mid D)$

**Via Bayes' Theorem**:
$$P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}$$

- $P(\theta)$ = **prior** (before seeing data)
- $P(D \mid \theta)$ = **likelihood** (how well θ explains data)
- $P(D)$ = **evidence** (normalizing constant)
- $P(\theta \mid D)$ = **posterior** (after seeing data)

**In GenJAX**:
```python
# Specify observations
observations = ChoiceMap.d({"x_0": 2.5, "x_1": 3.0})

# Create posterior target
target = Target(model, (params,), observations)

# Sample from posterior
trace, log_weight = target.importance(key, ChoiceMap.empty())
```

**Interpretation**: "Given what I observed, what parameter values are most plausible?"

**Tutorial 3, Chapter 4** covers Bayesian learning and posteriors

**See also**: Prior, Likelihood, Bayes' Theorem
{{% /expand %}}

### Predictive Distribution 📊
{{% expand "Predictive Distribution" %}}
The distribution over new, unobserved data given the data we've already seen.

**Posterior Predictive**: $P(x_{\text{new}} \mid D) = \int P(x_{\text{new}} \mid \theta) \cdot P(\theta \mid D) \, d\theta$

**In words**:
1. Consider all possible parameter values θ
2. Weight each by posterior probability P(θ | D)
3. Average their predictions for new data

**In GenJAX** (via sampling):
```python
# 1. Get posterior samples for θ
posterior_samples = []
for key in keys:
    trace, _ = target.importance(key, ChoiceMap.empty())
    theta = trace.get_choices()["theta"]
    posterior_samples.append(theta)

# 2. For each θ, generate predictions
predictions = []
for theta in posterior_samples:
    x_new = generate_new_data(theta)
    predictions.append(x_new)

# predictions is now a sample from the predictive distribution!
```

**Why important**: Captures uncertainty in both parameters AND new data

**Tutorial 3, Chapter 4** shows predictive distributions for Bayesian learning

**See also**: Posterior, Prior
{{% /expand %}}

### Prior Distribution 📊
{{% expand "Prior Distribution" %}}
The probability distribution over parameters *before* seeing any data: $P(\theta)$

**In Bayes' Theorem**:
$$P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}$$

- $P(\theta)$ = **prior** (our initial belief)
- $P(\theta \mid D)$ = **posterior** (updated belief after seeing data D)

**Types of priors**:
- **Informative**: Strong beliefs (e.g., Normal(0, 0.1²) says μ is near 0)
- **Weakly informative**: Gentle guidance (e.g., Normal(0, 10²))
- **Uninformative/Flat**: No preference (e.g., Uniform(-∞, ∞))

**In GenJAX**:
```python
@gen
def bayesian_model(mu0, sigma0):
    # Prior: μ ~ Normal(mu0, sigma0)
    mu = normal(mu0, sigma0) @ "mu"

    # Likelihood: x | μ ~ Normal(μ, 1.0)
    x = normal(mu, 1.0) @ "x"
    return x
```

**Controversy**: Subjectivity of priors is both a feature (encode knowledge) and criticism (bias results) of Bayesian methods

**Tutorial 3, Chapter 4** discusses priors in Bayesian learning

**See also**: Posterior, Likelihood, Conjugate Prior
{{% /expand %}}

### Probability Density Function (PDF) 📊
{{% expand "Probability Density Function (PDF)" %}}
For continuous random variables, the PDF describes the *density* of probability at each value.

**Key insight**: $p(x)$ is NOT a probability! It's a **density**.

**Why**:
- Probability of any exact value is 0 (infinitely many possible values)
- Probability is the **area under the PDF curve** over an interval:
  $$P(a \leq X \leq b) = \int_a^b p(x) \, dx$$

**Properties**:
- $p(x) \geq 0$ (non-negative)
- $\int_{-\infty}^{\infty} p(x) \, dx = 1$ (total area = 1)
- $p(x)$ can be > 1! (it's density, not probability)

**Example (Gaussian)**:
$$p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**In GenJAX**: We usually sample from PDFs rather than compute them directly

**Connection to discrete 📘**: PDF is the continuous analog of probability mass function (PMF)

**Tutorial 3, Chapter 2** introduces PDFs

**See also**: CDF, Continuous Random Variable
{{% /expand %}}

### Standard Normal 📊
{{% expand "Standard Normal" %}}
The Gaussian distribution with μ=0 and σ²=1.

**PDF**:
$$p(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)$$

**Notation**: $X \sim \mathcal{N}(0,1)$

**Why special**:
- Reference distribution (z-scores)
- Any Normal(μ, σ²) can be standardized: $Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0,1)$
- Tables and functions often use standard normal

**In GenJAX**:
```python
z = normal(0.0, 1.0) @ "z"  # Standard normal
```

**See also**: Gaussian Distribution, Z-score
{{% /expand %}}

### Stick-Breaking Construction 📊
{{% expand "Stick-Breaking Construction" %}}
A way to construct the infinite mixture weights in a Dirichlet Process by "breaking sticks."

**Metaphor**: Start with a stick of length 1. Repeatedly:
1. Break off a fraction (β) of the remaining stick
2. That piece becomes the weight for the next cluster
3. Continue with the remaining stick

**Mathematical process**:
```
β₁, β₂, β₃, ... ~ Beta(1, α)

π₁ = β₁
π₂ = β₂ · (1 - β₁)
π₃ = β₃ · (1 - β₁) · (1 - β₂)
...
πₖ = βₖ · ∏(1 - βⱼ) for j < k
```

**Properties**:
- All πₖ > 0
- Σ πₖ = 1 (sum to 1)
- πₖ decreases (on average) as k increases

**In GenJAX**:
```python
@gen
def stick_breaking(alpha, K):
    betas = []
    pis = []

    for k in range(K):
        beta_k = beta(1.0, alpha) @ f"beta_{k}"
        betas.append(beta_k)

    # Convert betas to pis
    remaining = 1.0
    for k in range(K):
        pis.append(betas[k] * remaining)
        remaining *= (1.0 - betas[k])

    return jnp.array(pis)
```

**Used in**: DPMM implementation

**Tutorial 3, Chapter 6** explains stick-breaking in detail

**See also**: Dirichlet Process, DPMM, Beta Distribution
{{% /expand %}}

### Truncation (in DPMM) 📊
{{% expand "Truncation" %}}
The Dirichlet Process is theoretically infinite, but in practice we approximate it by limiting to K components.

**Why necessary**:
- Can't actually implement infinite dimensions in code
- After K components, remaining weights are negligibly small

**How it works**:
```python
# Truncated stick-breaking
K_max = 20  # Truncation level

# First K-1 components use stick-breaking
for k in range(K_max - 1):
    beta_k = beta(1.0, alpha) @ f"beta_{k}"
    pis[k] = beta_k * remaining
    remaining *= (1.0 - beta_k)

# Last component gets all remaining weight
pis[K_max - 1] = remaining
```

**Choosing K**:
- Too small: Can't capture true number of clusters
- Too large: Slower inference, but mathematically fine
- Rule of thumb: K = 2-3× expected clusters

**Quality check**: If highest cluster indices have significant weight, increase K

**Tutorial 3, Chapter 6** discusses truncation in DPMM

**See also**: DPMM, Stick-breaking
{{% /expand %}}

### Uniform Distribution 📊
{{% expand "Uniform Distribution" %}}
A continuous distribution where all values in a range [a, b] are equally likely.

**Parameters**:
- a - minimum value
- b - maximum value

**PDF**:
$$p(x) = \begin{cases}
\frac{1}{b-a} & \text{if } a \leq x \leq b \\\\
0 & \text{otherwise}
\end{cases}$$

**In GenJAX**:
```python
@gen
def uniform_example():
    x = uniform(a, b) @ "x"
    return x
```

**Properties**:
- Mean: (a + b) / 2
- Variance: (b - a)² / 12

**Example uses**:
- Random initialization
- Uninformative prior on bounded parameters
- Modeling "complete ignorance" in a range

**Connection to discrete 📘**: Continuous analog of "all outcomes equally likely"

**Tutorial 3, Chapter 2** introduces uniform distribution

**See also**: PDF, Continuous Random Variable
{{% /expand %}}

### Variance 📊
{{% expand "Variance" %}}
A measure of spread/variability in a distribution. The expected squared deviation from the mean.

**Formula**: $\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$

**Notation**:
- Var(X) or σ²
- Standard deviation: σ = √(Var(X))

**In GenJAX** (approximation by sampling):
```python
# Generate samples
samples = jnp.array([model.simulate(key_i, ()).get_retval() for key_i in keys])

# Variance ≈ sample variance
variance = jnp.var(samples)
std_dev = jnp.sqrt(variance)
```

**Properties**:
- Always non-negative
- Var(aX + b) = a² · Var(X)
- For independent X, Y: Var(X + Y) = Var(X) + Var(Y)

**Interpretation**: "How spread out is the data?"

**See also**: Expected Value, Standard Deviation, Gaussian
{{% /expand %}}

### Weight Degeneracy 📊
{{% expand "Weight Degeneracy" %}}
A problem in importance sampling where most samples have negligible weight, so only one or a few samples contribute meaningfully.

**Symptom**: Effective sample size (ESS) << number of samples

**Example**:
```python
# Generate 100 samples with importance weights
samples = [...100 samples...]
weights = [...100 weights...]

# Compute ESS
normalized_weights = weights / sum(weights)
ESS = 1.0 / sum(normalized_weights**2)

# ESS ≈ 1.0 out of 100 = severe weight degeneracy!
```

**Causes**:
- Prior and posterior very different
- Proposal distribution poor match for posterior
- Model misspecification

**Solutions**:
- Use more samples
- Better proposal distribution
- Different inference method (MCMC)
- Fix model (e.g., remove extra randomization)

**Tutorial 3, Chapter 6**: The DPMM notebook had weight degeneracy (ESS=1/10) due to double randomization bug, which was fixed

**See also**: Importance Sampling, Effective Sample Size
{{% /expand %}}

---

## Navigation

**By Tutorial**:
- [Tutorial 1: Discrete Probability](../intro/) - 📘 Tagged terms
- [Tutorial 2: GenJAX Programming](../genjax/) - 💻 Tagged terms
- [Tutorial 3: Continuous Probability](../intro2/) - 📊 Tagged terms

**By Topic**:
- **Probability Basics**: Set, Outcome Space, Event, Probability, Conditional Probability
- **Programming**: @gen, Trace, ChoiceMap, simulate(), importance(), vmap
- **Distributions**: Bernoulli, Categorical, Normal/Gaussian, Beta, Uniform
- **Bayesian Learning**: Prior, Likelihood, Posterior, Predictive Distribution
- **Advanced Models**: GMM, DPMM, Dirichlet Process, Stick-breaking

---

*This glossary is designed to grow with the tutorials. If a term is missing, please let us know!*
