+++
date = "2026-06-24"
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
The *cardinality* or *size* of a set is the number of elements it contains. If $A = \{H, T\}$, then the cardinality of $A$ is $|A|=2$.

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

**Example:** "At least one tonkatsu" is the event $\{HT, TH, TT\} \subseteq \Omega$.

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

### Markov Equivalence Class 📘
{{% expand "Markov Equivalence Class" %}}
A *Markov equivalence class* is the set of all directed acyclic graphs (DAGs) that encode the **exact same set of conditional independencies**. Two graphs in the same class are called *Markov equivalent*: they impose identical constraints on the joint distribution, so **no amount of observational data can distinguish them** — the data is equally compatible with every graph in the class.

**Intuition:** reversing some arrows can leave the *statistical* content of a graph completely unchanged. For two variables, $T \to C$ and $C \to T$ are Markov equivalent — both factorize to the same joint $P(T,C)$ and both say only "$T$ and $C$ are dependent." Telling them apart requires *intervention* (the do-operator), not observation.

**The exception that breaks equivalence:** a *collider* $A \to B \leftarrow C$ asserts an independence ($A \perp C$) that its reversed cousins do not, so a collider is generally *not* equivalent to the corresponding chain or fork. (Same skeleton, different "v-structures" ⇒ different class.)

**Appears in:** [Tutorial 3, Chapter 9: Conditional Independence](../intro2/09_conditional_independence/) (chain / fork / collider, d-separation) and [Chapter 10: Causal Bayes Nets](../intro2/10_causal_bayes_nets/#the-same-statistics-three-different-stories) (the do-operator, intervention).
{{% /expand %}}

### Outcome Space 📘
{{% expand "Outcome Space" %}}
The *outcome space* (denoted $\Omega$, the Greek letter omega) is the set of all possible outcomes for a random process. It forms the foundation for calculating probabilities.

**Example:** For Chibany's two daily meals, $\Omega = \{HH, HT, TH, TT\}$.

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
A *set* is a collection of elements or members. Sets are defined by the elements they do or do not contain. The elements are listed with commas between them and "$\{$" denotes the start of a set and "$\}$" the end of a set. Note that the elements of a set are unique.

**Example:** $\{H, T\}$ is a set containing two elements: H and T.

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

**Key concept**: the [effective sample size](#effective-sample-size-) measures how well the weights are distributed. It is close to the number of samples when the proposal matches the target, and near 1 when a single sample dominates. The [importance weight](#importance-weight-) $w = p/q$ is the core correction.

**Used in 📊**: Posterior inference for Bayesian models, DPMM; introduced in full in [Chapter 16: Monte Carlo](../intro2/16_monte_carlo/#importance-sampling-sample-the-wrong-distribution) (self-normalized form, likelihood weighting), and the sequential version drives the [particle filter](#particle-filter-) of [Chapter 17](../intro2/17_particle_filtering/).

**See also**: [Importance Weight](#importance-weight-), [Effective Sample Size](#effective-sample-size-), [Proposal Distribution](#proposal-distribution-), [Target](#target-), [Weight Degeneracy](#weight-degeneracy-)
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

**Developed in 📊**: [Chapter 16: Monte Carlo](../intro2/16_monte_carlo/#the-monte-carlo-estimator) builds the estimator $\hat\mu_n$ from the ground up (die rolls, $\pi$-by-darts), with its $1/\sqrt{n}$ error rate, [rejection sampling](#rejection-sampling-), and [importance sampling](#importance-sampling-).

**See also**: [vmap](#vmap-), [Trace](#trace-), [Importance Sampling](#importance-sampling-), [Rejection Sampling](#rejection-sampling-), [Effective Sample Size](#effective-sample-size-)
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

$$F(x) = P(X \leq x) = \int_{-\infty}^x p(t)   dt$$

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

**For continuous**: $E[X] = \int_{-\infty}^{\infty} x \cdot p(x)   dx$

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

**Posterior Predictive**: $P(x_{\text{new}} \mid D) = \int P(x_{\text{new}} \mid \theta) \cdot P(\theta \mid D)   d\theta$

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
  $$P(a \leq X \leq b) = \int_a^b p(x)   dx$$

**Properties**:
- $p(x) \geq 0$ (non-negative)
- $\int_{-\infty}^{\infty} p(x)   dx = 1$ (total area = 1)
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
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
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
<!-- validate: skip -->
```python
# Suppose 100 importance-sampling weights, but one dominates all the rest:
weights = [0.97] + [0.03 / 99] * 99   # one huge weight, 99 tiny ones

# Compute the effective sample size (ESS)
total = sum(weights)
normalized_weights = [w / total for w in weights]
ESS = 1.0 / sum(w**2 for w in normalized_weights)

# ESS ≈ 1.06 out of 100 — severe weight degeneracy!
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

**Tutorial 3, Chapter 6**: The DPMM notebook had weight degeneracy (ESS=1/10) due to double randomization bug, which was fixed. In the streaming setting it is the reason a [particle filter](#particle-filter-) must [resample](#resampling-) every step ([Chapter 17](../intro2/17_particle_filtering/#resampling-and-degeneracy)).

**See also**: [Importance Sampling](#importance-sampling-), [Effective Sample Size](#effective-sample-size-), [Resampling](#resampling-), [Particle Filter](#particle-filter-)
{{% /expand %}}

### Surprise (Information Content) 📊
{{% expand "Surprise (Information Content)" %}}
The *surprise* (or *information content*) of an outcome $x$ is $-\log_2 P(x)$, measured in **bits**. The less probable an outcome was, the more surprised you should be when it occurs; the logarithm makes surprise **additive over independent events** (since independent probabilities multiply).

**Formula:** $\text{surprise}(x) = -\log_2 P(x)$

**Example:** an event you gave $P=0.01$ has surprise $-\log_2 0.01 \approx 6.6$ bits; one you gave $P=0.99$ has surprise $\approx 0.014$ bits.

**Appears in:** [Tutorial 3, Chapter 11: Information Theory](../intro2/11_information_theory/#surprise--log-px)

**See also:** [Entropy](#entropy-)
{{% /expand %}}

### Entropy 📊
{{% expand "Entropy" %}}
The *entropy* of a random variable $X$ is its **expected surprise** — the average uncertainty in its outcome, in bits. It is zero for a deterministic variable and maximal for a uniform one (1 bit for a fair coin).

**Formula:** $H(X) = -\sum_x P(x) \log_2 P(x) = \mathbb{E}\bigl[-\log_2 P(X)\bigr]$

**Intuition:** "How surprised do I expect to be by $X$, on average?" Equivalently, the average number of yes/no questions needed to pin down $X$.

**Examples:** fair coin → 1 bit; $\text{Bernoulli}(0.7)$ → 0.881 bits; certain outcome → 0 bits.

**Appears in:** [Tutorial 3, Chapter 11: Information Theory](../intro2/11_information_theory/#entropy--expected-surprise)

**See also:** [Surprise](#surprise-information-content-), [Joint Entropy](#joint-entropy-), [Conditional Entropy](#conditional-entropy-), [Mutual Information](#mutual-information-)
{{% /expand %}}

### Joint Entropy 📊
{{% expand "Joint Entropy" %}}
The *joint entropy* of two variables is the entropy of the pair $(X, Y)$ treated as a single combined outcome — the total uncertainty in both at once.

**Formula:** $H(X, Y) = -\sum_{x,y} P(x, y) \log_2 P(x, y)$

**Key identity (chain rule):** $H(X, Y) = H(X) + H(Y \mid X)$ — total uncertainty = uncertainty in $X$ + leftover uncertainty in $Y$ given $X$. (This is the logarithm of the probability chain rule $P(x,y)=P(x)P(y\mid x)$.)

**Appears in:** [Tutorial 3, Chapter 11: Information Theory](../intro2/11_information_theory/#joint-and-conditional-entropy)

**See also:** [Entropy](#entropy-), [Conditional Entropy](#conditional-entropy-), [Mutual Information](#mutual-information-)
{{% /expand %}}

### Conditional Entropy 📊
{{% expand "Conditional Entropy" %}}
The *conditional entropy* $H(Y \mid X)$ is the uncertainty that **remains** in $Y$ once you know $X$ — the entropy of $P(Y \mid X = x)$, averaged over $X$.

**Formula:** $H(Y \mid X) = -\sum_{x,y} P(x, y) \log_2 P(y \mid x) = H(X, Y) - H(X)$

**Limits:** if $X$ determines $Y$, then $H(Y \mid X) = 0$ (nothing left to learn); if $X$ is independent of $Y$, then $H(Y \mid X) = H(Y)$ (knowing $X$ didn't help).

**Appears in:** [Tutorial 3, Chapter 11: Information Theory](../intro2/11_information_theory/#joint-and-conditional-entropy)

**See also:** [Joint Entropy](#joint-entropy-), [Mutual Information](#mutual-information-)
{{% /expand %}}

### Mutual Information 📊
{{% expand "Mutual Information" %}}
The *mutual information* $I(X; Y)$ is how much learning one variable reduces your uncertainty about the other — the number of **bits the two variables share**. It is **symmetric** and **zero exactly when $X$ and $Y$ are independent**.

**Equivalent formulas:**
$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y).$$

**Independence:** $X \perp Y \iff I(X; Y) = 0$; conditionally, $X \perp Y \mid Z \iff I(X; Y \mid Z) = 0$.

**Picture:** if $H(X)$ and $H(Y)$ are overlapping circles, $I(X;Y)$ is their *overlap* (inclusion–exclusion with the joint entropy as the union).

**Appears in:** [Tutorial 3, Chapter 11: Information Theory](../intro2/11_information_theory/#mutual-information). Conditioning on a **collider** drives it *above* zero — explaining away, measured in bits.

**See also:** [Entropy](#entropy-), [Conditional Entropy](#conditional-entropy-), [Conditional Independence (d-separation)](#markov-equivalence-class-)
{{% /expand %}}

### Cross-Entropy 📊
{{% expand "Cross-Entropy" %}}
The *cross-entropy* $H(P, Q)$ is your **average surprise when reality is $P$ but you predicted with $Q$** — the surprise you actually feel using the wrong model.

**Formula:** $H(P, Q) = -\sum_x P(x) \log_2 Q(x)$

**Key identity:** $H(P, Q) = H(P) + D_{\text{KL}}(P \parallel Q)$ — total surprise = the irreducible part $H(P)$ + the penalty for being wrong. Since $H(P)$ doesn't depend on $Q$, **minimizing cross-entropy = minimizing KL divergence**, which is why "cross-entropy loss" trains classifiers and language models.

**Note:** $H(P, P) = H(P)$ — a perfect model's cross-entropy collapses to plain entropy.

**Appears in:** [Tutorial 3, Chapter 11: Information Theory](../intro2/11_information_theory/#cross-entropy-and-kl-divergence)

**See also:** [Entropy](#entropy-), [KL Divergence](#kl-divergence-)
{{% /expand %}}

### KL Divergence 📊
{{% expand "KL Divergence" %}}
The *Kullback–Leibler divergence* $D_{\text{KL}}(P \parallel Q)$ measures how far a model $Q$ is from the truth $P$, in **extra bits of surprise** — the cost of believing $Q$ when reality is $P$.

**Formula:** $D_{\text{KL}}(P \parallel Q) = \sum_x P(x) \log_2 \frac{P(x)}{Q(x)} = H(P, Q) - H(P)$

**Properties:** $D_{\text{KL}}(P \parallel Q) \ge 0$, with equality iff $Q = P$ (**Gibbs' inequality**) — the wrong distribution can only *increase* your average surprise. It is **not symmetric** ($D_{\text{KL}}(P \parallel Q) \ne D_{\text{KL}}(Q \parallel P)$ in general), so it's a *divergence*, not a true distance.

**Appears in:** [Tutorial 3, Chapter 11: Information Theory](../intro2/11_information_theory/#cross-entropy-and-kl-divergence)

**See also:** [Cross-Entropy](#cross-entropy-), [Entropy](#entropy-)
{{% /expand %}}

### Markov Property 📊
{{% expand "Markov Property" %}}
A process has the *Markov property* when **the future is independent of the past, given the present**: $P(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = P(X_{t+1} \mid X_t)$. Once you know the current state $X_t$, the entire earlier history adds nothing to your prediction of $X_{t+1}$.

**Intuition:** the present is a complete summary of the past *for the purpose of predicting the future* — history left all its mark on the current state.

**Appears in:** [Tutorial 3, Chapter 13: Markov Chains](../intro2/13_markov_chains/#naming-it-the-markov-property)

**See also:** [Markov Chain](#markov-chain-), [Transition Matrix](#transition-matrix-)
{{% /expand %}}

### Markov Chain 📊
{{% expand "Markov Chain" %}}
A *Markov chain* is a sequence of states $X_0, X_1, X_2, \dots$ that has the [Markov property](#markov-property-): each state depends only on the one before it. A chain over a finite set of states is fully described by its [transition matrix](#transition-matrix-).

**Example:** Chibany choosing tonkatsu or hamburger each day, where today's choice depends only on yesterday's.

**Appears in:** [Tutorial 3, Chapter 13: Markov Chains](../intro2/13_markov_chains/) (the chain machinery), and as a [random walk](#random-walk-) on a network in [Chapter 14](../intro2/14_random_walks_networks/) and a model of memory in [Chapter 15](../intro2/15_memory_search/).

**See also:** [Markov Property](#markov-property-), [Stationary Distribution](#stationary-distribution-), [Random Walk](#random-walk-)
{{% /expand %}}

### Transition Matrix 📊
{{% expand "Transition Matrix" %}}
The *transition matrix* $P$ of a Markov chain collects every one-step probability: $P_{ij} = P(X_{t+1} = j \mid X_t = i)$, the probability of moving *to* state $j$ given you are *in* state $i$. Each **row** is a probability distribution over next states, so it sums to 1 — such a matrix is called **row-stochastic**.

**Why it matters:** the matrix is also a *sampler* — pairing it with a stream of random numbers generates the whole sequence — and multiplying a distribution by $P$ steps it forward one unit of time.

**Appears in:** [Tutorial 3, Chapter 13: Markov Chains](../intro2/13_markov_chains/#two-views-of-the-same-chain). In [Chapter 14](../intro2/14_random_walks_networks/#from-a-graph-to-a-transition-matrix) it is built by row-normalizing a network's [adjacency matrix](#adjacency-matrix-and-degree-).

**See also:** [Markov Chain](#markov-chain-), [Stationary Distribution](#stationary-distribution-), [Adjacency Matrix and Degree](#adjacency-matrix-and-degree-)
{{% /expand %}}

### Stationary Distribution 📊
{{% expand "Stationary Distribution" %}}
The *stationary distribution* $\pi$ of a Markov chain is the long-run fraction of time the chain spends in each state — equivalently, the one distribution that a single step leaves unchanged: $\pi P = \pi$. If your belief about the current state is already $\pi$, it stays $\pi$ forever.

**How to find it:** by [power iteration](#power-iteration-) (just run the chain) or as the **left eigenvector of $P$ with eigenvalue 1** (every row-stochastic matrix has one). For a random walk on an undirected network it has the simple form $\pi_i \propto \deg(i)$.

**Appears in:** [Tutorial 3, Chapter 13: Markov Chains](../intro2/13_markov_chains/#the-stationary-distribution); the degree form and **PageRank** in [Chapter 14](../intro2/14_random_walks_networks/#the-stationary-distribution-of-a-walk).

**See also:** [Power Iteration](#power-iteration-), [Ergodicity](#ergodicity-), [PageRank](#pagerank-)
{{% /expand %}}

### Power Iteration 📊
{{% expand "Power Iteration" %}}
*Power iteration* finds a chain's [stationary distribution](#stationary-distribution-) by starting from any distribution $\mathbf{v}$ and multiplying by the [transition matrix](#transition-matrix-) repeatedly: $\mathbf{v}, \mathbf{v}P, \mathbf{v}P^2, \dots \to \pi$. The sequence converges to $\pi$ regardless of where it started (for an [ergodic](#ergodicity-) chain).

**Appears in:** [Tutorial 3, Chapter 13: Markov Chains](../intro2/13_markov_chains/#finding-π-just-run-it-power-iteration)

**See also:** [Stationary Distribution](#stationary-distribution-), [Ergodicity](#ergodicity-)
{{% /expand %}}

### Ergodicity 📊
{{% expand "Ergodicity" %}}
A Markov chain is *ergodic* when you can reach any state from any other (possibly in several steps) and it does not get trapped in a fixed cycle. An ergodic chain **mixes**: it converges to the *same* [stationary distribution](#stationary-distribution-) from every starting point, so $\pi$ is a property of the chain, not of where you began.

**Useful fact:** any chain can be made ergodic by adding a small probability $\varepsilon$ of jumping to any state — the trick that makes [PageRank](#pagerank-) well-defined (its "teleport" / damping term).

**Appears in:** [Tutorial 3, Chapter 13: Markov Chains](../intro2/13_markov_chains/#why-the-start-doesnt-matter-ergodicity); the ε-trick reused in [Chapter 14](../intro2/14_random_walks_networks/#pagerank-the-same-π-at-web-scale); reused as the forgetting-the-start basis of MCMC **mixing** in [Chapter 18](../intro2/18_markov_chain_monte_carlo/#mixing-burn-in-and-the-multimodal-trap).

**See also:** [Stationary Distribution](#stationary-distribution-), [PageRank](#pagerank-), [Mixing](#mixing-), [Burn-in](#burn-in-)
{{% /expand %}}

### Random Walk 📊
{{% expand "Random Walk" %}}
A *random walk* on a network is a [Markov chain](#markov-chain-) whose states are the **nodes** of a graph: at each step the walker moves to a uniformly random neighbour. Its [transition matrix](#transition-matrix-) is the [adjacency matrix](#adjacency-matrix-and-degree-) with each row normalized to sum to 1.

**Key result:** on an undirected, unweighted network the walk's [stationary distribution](#stationary-distribution-) is $\pi_i \propto \deg(i)$ — more-connected nodes are visited more often.

**Appears in:** [Tutorial 3, Chapter 14: Random Walks on Networks](../intro2/14_random_walks_networks/); a **censored** random walk models memory recall in [Chapter 15](../intro2/15_memory_search/).

**See also:** [Markov Chain](#markov-chain-), [Adjacency Matrix and Degree](#adjacency-matrix-and-degree-), [PageRank](#pagerank-), [Censoring Function](#censoring-function-)
{{% /expand %}}

### Adjacency Matrix and Degree 📊
{{% expand "Adjacency Matrix and Degree" %}}
A graph $G = (V, E)$ has **nodes** $V$ joined by **edges** $E$. Its *adjacency matrix* $L$ records the edges: $L_{ij} = 1$ when nodes $i$ and $j$ are connected, else $0$ (symmetric for an undirected graph). The *degree* of a node, $\deg(i)$, is the number of edges touching it — equivalently, the sum of its row in $L$.

**Why it matters:** row-normalizing $L$ gives the [transition matrix](#transition-matrix-) of a [random walk](#random-walk-), and for an undirected walk $\pi_i \propto \deg(i)$ — the degree *is* the long-run visit frequency.

**Appears in:** [Tutorial 3, Chapter 14: Random Walks on Networks](../intro2/14_random_walks_networks/#chibanys-animal-network)

**See also:** [Random Walk](#random-walk-), [Transition Matrix](#transition-matrix-)
{{% /expand %}}

### PageRank 📊
{{% expand "PageRank" %}}
*PageRank* — the algorithm behind the original Google search engine — ranks the nodes of a directed graph by the [stationary distribution](#stationary-distribution-) of a [random walk](#random-walk-) over it: a "random surfer" who follows links, with a small probability $\varepsilon$ of teleporting to a random node (the [ergodicity](#ergodicity-) fix; Google's *damping factor* is $1 - \varepsilon$). Important nodes are the ones a random walker visits often.

**Cognitive connection:** Griffiths, Steyvers & Firl (2007) showed PageRank over a *semantic* network predicts which words people produce in a fluency task.

**Appears in:** [Tutorial 3, Chapter 14: Random Walks on Networks](../intro2/14_random_walks_networks/#pagerank-the-same-π-at-web-scale)

**See also:** [Stationary Distribution](#stationary-distribution-), [Ergodicity](#ergodicity-), [Semantic Network](#semantic-network-)
{{% /expand %}}

### Semantic Network 📊
{{% expand "Semantic Network" %}}
A *semantic network* represents knowledge as a graph: **concepts** are nodes and **associations** are edges (e.g. *dog*–*cat*). Such networks are often estimated from word-association data — asking many people what comes to mind for a cue word.

**Why it matters:** treating semantic memory as a network lets a single [random walk](#random-walk-) on it model how people *recall* — the basis of the memory-search model in Chapter 15.

**Appears in:** [Tutorial 3, Chapter 14: Random Walks on Networks](../intro2/14_random_walks_networks/#whats-a-graph) and [Chapter 15: Memory Search](../intro2/15_memory_search/).

**See also:** [Random Walk](#random-walk-), [Censoring Function](#censoring-function-), [PageRank](#pagerank-)
{{% /expand %}}

### Censoring Function 📊
{{% expand "Censoring Function" %}}
In the random-walk model of memory search (Abbott, Austerweil & Griffiths 2012), the *censoring function* maps the latent walk onto the observed list: you **report a word only the first time the walk reaches it, and only if it is in the target category**; revisits and off-category nodes are *censored* (hidden). The borrowed statistics term means "happened but unrecorded."

**Consequence:** the gap between successive *first-hitting times* $\tau(k)$ — when the walk first reaches the $k$-th reported item — drives the **inter-item response time** $\text{IRT}(k) = \tau(k) - \tau(k-1) + \text{word length}$, reproducing the human "switch-cost" curve with no explicit switch rule.

**Appears in:** [Tutorial 3, Chapter 15: Memory Search](../intro2/15_memory_search/#the-censoring-function)

**See also:** [Random Walk](#random-walk-), [Semantic Network](#semantic-network-)
{{% /expand %}}

### Importance Weight 📊
{{% expand "Importance Weight" %}}
When you sample from a **proposal** $q$ instead of the **target** $p$ you care about, the *importance weight* $w(x) = p(x)/q(x)$ corrects the mismatch, so that a weighted average under $q$ estimates an expectation under $p$: $\mathbb{E}_p[f] = \mathbb{E}_q[f \cdot w]$.

**Good vs. bad weights:** a proposal that resembles the target gives weights near 1 (even, healthy); a mismatched proposal gives a few huge weights and many near-zero ones (the estimate gets noisy). The [effective sample size](#effective-sample-size-) measures this.

**Appears in:** [Tutorial 3, Chapter 16: Monte Carlo](../intro2/16_monte_carlo/#importance-sampling-sample-the-wrong-distribution)

**See also:** [Importance Sampling](#importance-sampling-), [Proposal Distribution](#proposal-distribution-), [Effective Sample Size](#effective-sample-size-)
{{% /expand %}}

### Effective Sample Size 📊
{{% expand "Effective Sample Size" %}}
For a set of normalized weights $w_t$ (summing to 1), the *effective sample size* is $N_{\text{eff}} = 1 / \sum_t w_t^2$. It answers "my $T$ weighted samples are worth how many equally-weighted ones?"

**Two limits:** perfectly even weights give $N_{\text{eff}} = T$ (every sample counts); one dominating weight gives $N_{\text{eff}} \approx 1$. It is a **diagnostic of how well the proposal $q$ matches the target $p$** — not a direct measure of an estimate's accuracy.

**Appears in:** [Tutorial 3, Chapter 16: Monte Carlo](../intro2/16_monte_carlo/#effective-sample-size); the streaming version (weight degeneracy) in [Chapter 17: Particle Filtering](../intro2/17_particle_filtering/#resampling-and-degeneracy).

**See also:** [Importance Weight](#importance-weight-), [Importance Sampling](#importance-sampling-), [Weight Degeneracy](#weight-degeneracy-), [Resampling](#resampling-)
{{% /expand %}}

### Rejection Sampling 📊
{{% expand "Rejection Sampling" %}}
A way to sample from a target density $p$ you can *evaluate* but not directly draw from: put an easy **envelope** over $p$, throw points uniformly under the envelope, and **keep only those that fall under $p$**. The survivors are exact samples from $p$.

**Trade-off:** if the envelope is much larger than the area under $p$, most proposals are rejected — wasted work. That inefficiency is what [importance sampling](#importance-sampling-) avoids by *reweighting* instead of rejecting.

**Appears in:** [Tutorial 3, Chapter 16: Monte Carlo](../intro2/16_monte_carlo/#when-you-cant-sample-p-rejection-and-inverse-cdf)

**See also:** [Importance Sampling](#importance-sampling-), [Monte Carlo Simulation](#monte-carlo-simulation-)
{{% /expand %}}

### Proposal Distribution 📊
{{% expand "Proposal Distribution" %}}
In importance sampling and MCMC, the *proposal* $q$ is the distribution you actually draw from — usually one that is easy to sample — as a stand-in for a **target** that is hard. In importance sampling you correct for the swap with the [importance weight](#importance-weight-) $p/q$; in [Metropolis–Hastings](#metropolishastings-) the proposal generates a *candidate* next state that is then accepted or rejected.

**Choosing it well:** a proposal close to the target keeps weights even (importance sampling) or mixing fast (MCMC); a poor proposal wrecks both.

**Appears in:** [Tutorial 3, Chapter 16: Monte Carlo](../intro2/16_monte_carlo/#importance-sampling-sample-the-wrong-distribution) and [Chapter 18: Markov Chain Monte Carlo](../intro2/18_markov_chain_monte_carlo/#metropolishastings).

**See also:** [Importance Weight](#importance-weight-), [Metropolis–Hastings](#metropolishastings-), [Acceptance Ratio](#acceptance-ratio-)
{{% /expand %}}

### Particle Filter 📊
{{% expand "Particle Filter" %}}
A method for **streaming** inference about a hidden state that changes over time. It represents the posterior with a swarm of weighted samples (*particles*) and updates them each time new data arrive by looping **weight → resample → propagate** — *sequential importance sampling* with a resampling step. The guiding idea: *yesterday's posterior is today's prior.*

**As a process model:** a *small* particle filter — a handful of guesses updated left-to-right — predicts human limited memory, order effects, and run-to-run variability.

**Appears in:** [Tutorial 3, Chapter 17: Particle Filtering](../intro2/17_particle_filtering/)

**See also:** [Importance Sampling](#importance-sampling-), [Resampling](#resampling-), [Effective Sample Size](#effective-sample-size-)
{{% /expand %}}

### Resampling 📊
{{% expand "Resampling" %}}
In a [particle filter](#particle-filter-), *resampling* draws a new set of particles from the current ones *with probability proportional to their weights* (a Categorical draw of indices): heavy particles are cloned, light ones culled, and all weights reset to equal.

**Why it's needed:** without it, weights multiply over time until one particle carries everything — *weight degeneracy*, measured by a collapsing [effective sample size](#effective-sample-size-). Resampling concentrates the swarm where the action is and keeps the filter useful indefinitely.

**Appears in:** [Tutorial 3, Chapter 17: Particle Filtering](../intro2/17_particle_filtering/#sequential-importance-sampling)

**See also:** [Particle Filter](#particle-filter-), [Weight Degeneracy](#weight-degeneracy-), [Effective Sample Size](#effective-sample-size-)
{{% /expand %}}

### Markov Chain Monte Carlo (MCMC) 📊
{{% expand "Markov Chain Monte Carlo (MCMC)" %}}
A family of methods that sample from a target distribution $\pi$ (typically a hard-to-sample Bayesian posterior) by **designing a Markov chain whose [stationary distribution](#stationary-distribution-) is exactly $\pi$**, then running it and collecting the states it visits. It runs [Chapter 13](../intro2/13_markov_chains/)'s logic backwards: instead of being handed a chain and finding its $\pi$, you start from the $\pi$ you want and build the chain.

**Workhorses:** [Metropolis–Hastings](#metropolishastings-) (propose + accept/reject) and [Gibbs sampling](#gibbs-sampling-) (resample a coordinate from its conditional).

**Appears in:** [Tutorial 3, Chapter 18: Markov Chain Monte Carlo](../intro2/18_markov_chain_monte_carlo/) and [Chapter 19: Sampling the Mind](../intro2/19_sampling_the_mind/).

**See also:** [Metropolis–Hastings](#metropolishastings-), [Gibbs Sampling](#gibbs-sampling-), [Stationary Distribution](#stationary-distribution-), [Burn-in](#burn-in-), [Mixing](#mixing-)
{{% /expand %}}

### Metropolis–Hastings 📊
{{% expand "Metropolis-Hastings" %}}
The most general MCMC recipe. From the current state $x$: **propose** a candidate $x'$ from a [proposal distribution](#proposal-distribution-), then accept it with probability given by the [acceptance ratio](#acceptance-ratio-) $A = \min(1, P(x')/P(x))$ (for a symmetric proposal); otherwise stay at $x$.

**Why it works:** the rule forces *detailed balance* with respect to $P$, so $P$ is the chain's stationary distribution. Because only the **ratio** $P(x')/P(x)$ appears, the normalizer cancels — you can sample an *unnormalized* posterior.

**Appears in:** [Tutorial 3, Chapter 18: Markov Chain Monte Carlo](../intro2/18_markov_chain_monte_carlo/#metropolishastings) and [Chapter 19: Sampling the Mind](../intro2/19_sampling_the_mind/).

**See also:** [Markov Chain Monte Carlo (MCMC)](#markov-chain-monte-carlo-mcmc-), [Acceptance Ratio](#acceptance-ratio-), [Proposal Distribution](#proposal-distribution-), [Gibbs Sampling](#gibbs-sampling-)
{{% /expand %}}

### Acceptance Ratio 📊
{{% expand "Acceptance Ratio" %}}
In [Metropolis–Hastings](#metropolishastings-), the probability of moving to a proposed state $x'$ from the current $x$: $A = \min\left(1, \frac{P(x')}{P(x)}\right)$ for a symmetric proposal. **Uphill** moves ($P(x') > P(x)$) are always accepted; **downhill** moves are accepted in proportion to their relative height.

**Key feature:** only the *ratio* of target probabilities matters, so any normalizing constant cancels — the reason MCMC works on posteriors known only up to their evidence $p(\text{data})$.

**Appears in:** [Tutorial 3, Chapter 18: Markov Chain Monte Carlo](../intro2/18_markov_chain_monte_carlo/#metropolishastings)

**See also:** [Metropolis–Hastings](#metropolishastings-), [Proposal Distribution](#proposal-distribution-)
{{% /expand %}}

### Gibbs Sampling 📊
{{% expand "Gibbs Sampling" %}}
An MCMC method that updates **one coordinate at a time**, drawing each from its exact full conditional $P(x_i \mid x_{-i})$ (the distribution of $x_i$ given the current values of all other coordinates). It **never rejects** — sampling from the true conditional automatically satisfies detailed balance — but requires those conditionals to be available, which they are when the model is built from **[conjugate](#conjugate-prior-)** pieces.

**Appears in:** [Tutorial 3, Chapter 18: Markov Chain Monte Carlo](../intro2/18_markov_chain_monte_carlo/#gibbs-sampling) and [Chapter 19: Sampling the Mind](../intro2/19_sampling_the_mind/#step-1--gibbs-the-θᵢ-conjugate).

**See also:** [Markov Chain Monte Carlo (MCMC)](#markov-chain-monte-carlo-mcmc-), [Metropolis–Hastings](#metropolishastings-), [Conjugate Prior](#conjugate-prior-)
{{% /expand %}}

### Burn-in 📊
{{% expand "Burn-in" %}}
The initial portion of an MCMC run, *discarded* before collecting samples. Those early states reflect the chain's arbitrary starting point rather than the target distribution; once the chain has **mixed** (forgotten its start), the remaining states approximate the target and are kept.

**Appears in:** [Tutorial 3, Chapter 18: Markov Chain Monte Carlo](../intro2/18_markov_chain_monte_carlo/#mixing-burn-in-and-the-multimodal-trap)

**See also:** [Mixing](#mixing-), [Markov Chain Monte Carlo (MCMC)](#markov-chain-monte-carlo-mcmc-), [Ergodicity](#ergodicity-)
{{% /expand %}}

### Mixing 📊
{{% expand "Mixing" %}}
An MCMC chain has *mixed* when it has forgotten its starting point and is exploring the whole target distribution — the same forgetting-the-start property as [ergodicity](#ergodicity-). A well-mixed chain run from different starts gives the same answer.

**The trap:** on a **multimodal** target, a chain can have a perfectly healthy local acceptance rate yet stay stuck in one mode, never crossing the low-probability valleys between peaks. *Good local acceptance does not imply good global mixing* — which is why running multiple chains from different starts and checking they agree matters.

**Appears in:** [Tutorial 3, Chapter 18: Markov Chain Monte Carlo](../intro2/18_markov_chain_monte_carlo/#mixing-burn-in-and-the-multimodal-trap)

**See also:** [Burn-in](#burn-in-), [Ergodicity](#ergodicity-), [Markov Chain Monte Carlo (MCMC)](#markov-chain-monte-carlo-mcmc-)
{{% /expand %}}

### MCMC with People 📊
{{% expand "MCMC with People" %}}
A method (Sanborn & Griffiths, 2007) that treats a *person* as the accept step of a Metropolis sampler: show them two options, let them choose, repeat. If the person accepts in proportion to their own posterior, the chain of choices converges to that posterior — and with no data to fit, it converges to the **prior in their head**. Run on cartoon animals, it recovers people's mental category prototypes.

**Appears in:** [Tutorial 3, Chapter 19: Sampling the Mind](../intro2/19_sampling_the_mind/#when-a-person-is-the-accept-step)

**See also:** [Markov Chain Monte Carlo (MCMC)](#markov-chain-monte-carlo-mcmc-), [Metropolis–Hastings](#metropolishastings-), [Prior Distribution](#prior-distribution-)
{{% /expand %}}

### Statistical Decision Theory 📊
{{% expand "Statistical Decision Theory" %}}
The normative account of how to turn a belief into an action once you say what your mistakes cost. It has four pieces: the unknown state of the world $\theta$, an observation $x$, an action $a$ from a set $A$, and a **loss** $L(\theta,a)$; a decision rule $d(x)$ maps observations to actions, and the optimal rule minimizes expected loss.

**Appears in:** [Tutorial 3, Chapter 20: Statistical Decision Theory](../intro2/20_statistical_decision_theory/)

**See also:** [Loss Function](#loss-function-), [Risk](#risk-), [Bayes Estimator](#bayes-estimator-), [Minimax](#minimax-)
{{% /expand %}}

### Loss Function 📊
{{% expand "Loss Function" %}}
A function $L(\theta, a)$ giving the cost of taking action $a$ when the world is really $\theta$ — the mirror image of reward. The loss you choose determines which decision is optimal: **0–1 loss → posterior mode (MAP)**, **squared loss → posterior mean**, **absolute loss → posterior median**.

**Appears in:** [Tutorial 3, Chapter 20: Statistical Decision Theory](../intro2/20_statistical_decision_theory/)

**See also:** [Risk](#risk-), [Bayes Estimator](#bayes-estimator-), [MAP Estimate](#map-estimate-), [Reward](#reward-)
{{% /expand %}}

### Risk 📊
{{% expand "Risk" %}}
The expected loss of a decision rule, $R(\theta, d) = \mathbb{E}_x[L(\theta, d(x))]$ — a report card on a rule, averaged over the data. The **Bayes** criterion minimizes prior-/posterior-expected risk; the **minimax** criterion minimizes the worst-case risk over $\theta$.

**Appears in:** [Tutorial 3, Chapter 20: Statistical Decision Theory](../intro2/20_statistical_decision_theory/)

**See also:** [Loss Function](#loss-function-), [Minimax](#minimax-), [Bayes Estimator](#bayes-estimator-)
{{% /expand %}}

### Bayes Estimator 📊
{{% expand "Bayes Estimator" %}}
The action (often an estimate) that minimizes **posterior-expected loss**, $\arg\min_a \mathbb{E}_{\theta\mid x}[L(\theta,a)]$. Under 0–1, squared, and absolute loss it equals the posterior mode, mean, and median respectively.

**Appears in:** [Tutorial 3, Chapter 20: Statistical Decision Theory](../intro2/20_statistical_decision_theory/)

**See also:** [Loss Function](#loss-function-), [Risk](#risk-), [MAP Estimate](#map-estimate-), [Posterior Distribution](#posterior-distribution-)
{{% /expand %}}

### Minimax 📊
{{% expand "Minimax" %}}
A decision criterion that ignores the prior and minimizes the **worst-case** loss: $\arg\min_d \max_\theta R(\theta, d)$. The pessimist's rule — it buys insurance against the catastrophe at the cost of being worse in the typical case, where the Bayes rule wins.

**Appears in:** [Tutorial 3, Chapter 20: Statistical Decision Theory](../intro2/20_statistical_decision_theory/)

**See also:** [Risk](#risk-), [Bayes Estimator](#bayes-estimator-), [Statistical Decision Theory](#statistical-decision-theory-)
{{% /expand %}}

### MAP Estimate 📊
{{% expand "MAP Estimate" %}}
The **maximum a posteriori** estimate — the value that maximizes the posterior, i.e. the posterior **mode**. It is the Bayes estimator under **0–1 loss** (you pay 1 unless you are exactly right, so you bet on the densest point).

**Appears in:** [Tutorial 3, Chapter 20: Statistical Decision Theory](../intro2/20_statistical_decision_theory/)

**See also:** [Bayes Estimator](#bayes-estimator-), [Loss Function](#loss-function-), [Posterior Distribution](#posterior-distribution-)
{{% /expand %}}

### Probability Matching 📊
{{% expand "Probability Matching" %}}
Choosing options in proportion to their probability rather than always choosing the most likely one. Long seen as an "irrationality," it is exactly the policy of a sampler that values its time: **one and done** (Vul, Goodman, Griffiths & Tenenbaum, 2014) shows that when samples cost time, deciding from a *single* sample maximizes reward rate — and a one-sample decision matches the posterior.

**Appears in:** [Tutorial 3, Chapter 20: Statistical Decision Theory](../intro2/20_statistical_decision_theory/)

**See also:** [Monte Carlo Simulation](#monte-carlo-simulation-), [Bayes Estimator](#bayes-estimator-)
{{% /expand %}}

### Markov Decision Process 📊
{{% expand "Markov Decision Process" %}}
A model of sequential decision-making with five pieces: states $S$, actions $A$, a transition function $T(s'\mid s,a)$ (one transition matrix **per action**), a reward $R$, and a discount $\gamma$. An action selects *which* transition matrix governs the next state. A plain Markov chain is the special case with a single action.

**Appears in:** [Tutorial 3, Chapter 21: Markov Decision Processes](../intro2/21_markov_decision_processes/)

**See also:** [Policy](#policy-), [Reward](#reward-), [Discount Factor](#discount-factor-), [Value Function](#value-function-), [Markov Chain](#markov-chain-)
{{% /expand %}}

### Policy 📊
{{% expand "Policy" %}}
An agent's rule for acting, $\pi(a\mid s)$ — which action to take in each state. Because the world is Markov, a policy needs only the *current* state, which is what tames the combinatorial blow-up of planning a whole sequence of actions. The **optimal policy** $\pi^*$ achieves the highest value in every state and is deterministic.

**Appears in:** [Tutorial 3, Chapter 21: Markov Decision Processes](../intro2/21_markov_decision_processes/)

**See also:** [Markov Decision Process](#markov-decision-process-), [Value Function](#value-function-), [Value Iteration](#value-iteration-)
{{% /expand %}}

### Reward 📊
{{% expand "Reward" %}}
The payoff signal an agent seeks to maximize over time — the mirror image of loss. It may depend on the state, $R(s)$, or the state and action, $R(s,a)$. Agents maximize the **discounted return**, not the immediate reward, which is why far-sighted behavior can accept short-term losses.

**Appears in:** [Tutorial 3, Chapter 21: Markov Decision Processes](../intro2/21_markov_decision_processes/)

**See also:** [Return](#return-), [Discount Factor](#discount-factor-), [Loss Function](#loss-function-), [Reward Shaping](#reward-shaping-)
{{% /expand %}}

### Discount Factor 📊
{{% expand "Discount Factor" %}}
The number $\gamma \in [0,1)$ that weights future rewards relative to immediate ones, making the infinite-horizon return finite. It encodes how far ahead the agent looks: small $\gamma$ is impatient, large $\gamma$ is far-sighted. Sweeping $\gamma$ can flip the optimal policy at a sharp threshold.

**Appears in:** [Tutorial 3, Chapter 21: Markov Decision Processes](../intro2/21_markov_decision_processes/)

**See also:** [Return](#return-), [Value Function](#value-function-), [Reward](#reward-)
{{% /expand %}}

### Return 📊
{{% expand "Return" %}}
The discounted sum of all future rewards from time $t$, $G_t = \sum_{k\ge0}\gamma^k R_{t+k}$. The thing an agent actually maximizes; a state's **value** is the expected return from that state under a policy.

**Appears in:** [Tutorial 3, Chapter 21: Markov Decision Processes](../intro2/21_markov_decision_processes/)

**See also:** [Discount Factor](#discount-factor-), [Value Function](#value-function-), [Reward](#reward-)
{{% /expand %}}

### Value Function 📊
{{% expand "Value Function" %}}
The expected return from a state (the **state value** $v_\pi(s)$) or from a state–action pair (the **action value** or **Q-value** $q_\pi(s,a)$) under a policy $\pi$. The optimal values $v^*$, $q^*$ satisfy the Bellman equation $v^*(s)=\max_a q^*(s,a)$ and determine the optimal policy.

**Appears in:** [Tutorial 3, Chapter 21: Markov Decision Processes](../intro2/21_markov_decision_processes/)

**See also:** [Bellman Equation](#bellman-equation-), [Value Iteration](#value-iteration-), [Return](#return-), [Q-Learning](#q-learning-)
{{% /expand %}}

### Bellman Equation 📊
{{% expand "Bellman Equation" %}}
The recursion at the heart of dynamic programming: $v^*(s) = \max_a\big[R(s) + \gamma\sum_{s'}T(s'\mid s,a)\,v^*(s')\big]$ — the value of a state is the immediate reward plus the discounted value of the best next move. Recursive ($v^*$ on both sides), it is solved by iteration rather than by enumerating plans.

**Appears in:** [Tutorial 3, Chapter 21: Markov Decision Processes](../intro2/21_markov_decision_processes/)

**See also:** [Value Function](#value-function-), [Value Iteration](#value-iteration-), [Markov Decision Process](#markov-decision-process-)
{{% /expand %}}

### Value Iteration 📊
{{% expand "Value Iteration" %}}
A dynamic-programming algorithm that finds the optimal values by applying the Bellman update repeatedly from any starting guess: $v_{k+1}(s)=\max_a[R(s)+\gamma\sum_{s'}T(s'\mid s,a)v_k(s')]$. It converges because the Bellman operator is a $\gamma$-contraction — each sweep shrinks the error by a factor of $\gamma$. Requires the model $T, R$.

**Appears in:** [Tutorial 3, Chapter 21: Markov Decision Processes](../intro2/21_markov_decision_processes/)

**See also:** [Bellman Equation](#bellman-equation-), [Value Function](#value-function-), [Q-Learning](#q-learning-)
{{% /expand %}}

### Q-Learning 📊
{{% expand "Q-Learning" %}}
A **model-free** reinforcement-learning algorithm that estimates the optimal action values $Q(s,a)$ from raw experience — no transition model needed. After each step it nudges its estimate toward the **TD target** $r + \gamma\max_{a'}Q(s',a')$ by a fraction $\alpha$. It learns the same optimal policy value iteration would compute, using the single next state the world returned instead of the full expectation over $T$.

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Temporal-Difference Error](#temporal-difference-error-), [Learning Rate](#learning-rate-), [Epsilon-Greedy Exploration](#epsilon-greedy-exploration-), [Value Iteration](#value-iteration-)
{{% /expand %}}

### Temporal-Difference Error 📊
{{% expand "Temporal-Difference Error" %}}
The "surprise" term in a TD update: target − current estimate $= r + \gamma\max_{a'}Q(s',a') - Q(s,a)$. Learning is repeatedly reducing this error. It is also a model of the brain: midbrain **dopamine** neurons signal a reward-*prediction* error that matches the TD error (Schultz, Dayan & Montague, 1997).

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Q-Learning](#q-learning-), [Learning Rate](#learning-rate-), [Value Function](#value-function-)
{{% /expand %}}

### Learning Rate 📊
{{% expand "Learning Rate" %}}
The step size $\alpha \in (0,1]$ in a TD update, controlling how far each estimate moves toward the target. Large $\alpha$ learns fast but noisily; small $\alpha$ is slow but stable. Convergence guarantees require $\alpha$ to shrink appropriately over time.

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Q-Learning](#q-learning-), [Temporal-Difference Error](#temporal-difference-error-)
{{% /expand %}}

### Epsilon-Greedy Exploration 📊
{{% expand "Epsilon-Greedy Exploration" %}}
A simple exploration strategy: with probability $\varepsilon$ take a random action, otherwise take the current best (greedy) one. The random fraction guarantees the agent keeps trying every state–action pair, which Q-learning needs to converge. The tree-search analog is **UCB**.

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Q-Learning](#q-learning-), [Monte Carlo Tree Search](#monte-carlo-tree-search-)
{{% /expand %}}

### Reward Shaping 📊
{{% expand "Reward Shaping" %}}
Adding extra reward to guide learning. Done carelessly it backfires: a reward you can **farm in a loop** creates a positive cycle and the agent never finishes the task. **Potential-based shaping** (Ng, Harada & Russell, 1999), $F=\gamma\Phi(s')-\Phi(s)$, is the principled fix: the shaping along any path collapses to a constant fixed by its endpoints, so it can't create a farmable cycle and provably preserves the optimal policy.

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Reward](#reward-), [Reward Hacking](#reward-hacking-), [Policy](#policy-)
{{% /expand %}}

### Reward Hacking 📊
{{% expand "Reward Hacking" %}}
When an agent maximizes the reward you *specified* in a way that defeats what you *meant* — the positive-cycle trap at scale. In RLHF, agents farm a learned model of human approval without doing the task, exactly as a badly-shaped gridworld agent paces for praise instead of reaching the goal. A central problem in AI alignment.

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Reward Shaping](#reward-shaping-), [Reward](#reward-)
{{% /expand %}}

### Simulation-Based RL 📊
{{% expand "Simulation-Based RL" %}}
Learning a model from experience and then planning by **simulating** with it — the middle ground between value iteration (needs the whole model) and Q-learning (needs many real trials). Also called model-based RL. **Dyna** (learn $\hat T$ by counting, then plan) and **MCTS** (plan by tree-structured rollouts) are examples; AlphaZero and MuZero are the same idea at scale.

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Monte Carlo Tree Search](#monte-carlo-tree-search-), [Monte Carlo Simulation](#monte-carlo-simulation-), [Value Iteration](#value-iteration-), [Certainty Equivalence](#certainty-equivalence-)
{{% /expand %}}

### Certainty Equivalence 📊
{{% expand "Certainty Equivalence" %}}
Planning with a single **point estimate** of an unknown model as if it were exactly correct: fit the model (e.g. the maximum-likelihood transition matrix $\hat T$ from empirical transition counts), then optimize against $\hat T$ and ignore the uncertainty that remains. **Dyna** is the canonical example. It is simple and works well once enough data has made the estimate sharp, but because it throws the uncertainty away it never reasons about *exploring to reduce* that uncertainty — the principled alternative keeps a posterior over the model (see Bayes-Adaptive MDP).

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Bayes-Adaptive MDP](#bayes-adaptive-mdp-), [Simulation-Based RL](#simulation-based-rl-), [Dirichlet Distribution](#dirichlet-distribution-)
{{% /expand %}}

### Bayes-Adaptive MDP 📊
{{% expand "Bayes-Adaptive MDP" %}}
The reformulation of *learning* an unknown MDP as a *planning* problem. Fold the unknown model parameters $\theta$ (e.g. the transition matrix) into the state as a **hidden, static** component: the true state becomes $(s, \theta)$ with $s$ observed and $\theta$ never seen directly, and each observed transition sharpens a posterior (belief) over $\theta$. The augmented problem is a **partially observable MDP (POMDP)** — a special "parameter-uncertainty" one — in which optimal behavior automatically trades off **exploration** (acting to reduce uncertainty about $\theta$) against **exploitation**. Solving it exactly is intractable; **posterior sampling** (Thompson sampling) is the common tractable approximation, and **certainty equivalence** (Dyna) is the point-estimate shortcut that ignores the belief entirely.

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Certainty Equivalence](#certainty-equivalence-), [Simulation-Based RL](#simulation-based-rl-), [Dirichlet Distribution](#dirichlet-distribution-), [Partially Observable MDP (POMDP)](#partially-observable-mdp-pomdp-)
{{% /expand %}}

### Partially Observable MDP (POMDP) 📊
{{% expand "Partially Observable MDP (POMDP)" %}}
A Markov decision process in which the agent **cannot directly observe the state**. Instead of the state $s$, it receives an **observation** $o$ from an observation model $O(o \mid s)$ that depends on the hidden state; so it maintains a **belief** — a probability distribution over which state it is in — updates that belief by Bayes' rule after each action and observation, and chooses actions as a function of the belief rather than the unknown state. Optimal POMDP planning is much harder than MDP planning (the belief is a continuous state), and partial observability is the rule in realistic problems — noisy sensors, hidden diseases, unknown user intent. Learning an unknown MDP is itself a structured POMDP (see Bayes-Adaptive MDP). Covered in depth in later chapters.

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Bayes-Adaptive MDP](#bayes-adaptive-mdp-), [Markov Decision Process](#markov-decision-process-), [Posterior Distribution](#posterior-distribution-)
{{% /expand %}}

### Two-Step Task 📊
{{% expand "Two-Step Task" %}}
A two-stage decision task (Daw et al., 2011) designed to **dissociate model-free from model-based control** behaviorally. A first-stage choice leads *probabilistically* — a **common** (70%) or **rare** (30%) transition — to one of two second-stage states, where a slowly drifting reward is collected. The diagnostic is whether the agent **repeats** its first-stage choice (*stays*) as a function of the previous trial's *reward* and *transition type*: a **model-free** learner shows only a **main effect of reward** (rewarded → stay, regardless of transition), whereas a **model-based** learner shows a **reward × transition interaction** (a *rare* reward makes it *switch*, since the other first-stage option commonly reaches the now-valuable state). People show both, taken as evidence for parallel habitual and goal-directed systems.

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Simulation-Based RL](#simulation-based-rl-), [Q-Learning](#q-learning-), [Temporal-Difference Error](#temporal-difference-error-)
{{% /expand %}}

### Monte Carlo Tree Search 📊
{{% expand "Monte Carlo Tree Search" %}}
A planning algorithm (MCTS) that simulates forward only from the current state, building a search tree through four repeated phases — **select** (descend by UCB), **expand** (add a node), **simulate** (random rollout to estimate value), **backup** (send the return up the path). It is the engine inside AlphaZero (which replaces the random rollout with a learned value network) and MuZero (which learns the model it searches).

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Simulation-Based RL](#simulation-based-rl-), [Upper Confidence Bound (UCB)](#upper-confidence-bound-ucb-), [Monte Carlo Simulation](#monte-carlo-simulation-), [Epsilon-Greedy Exploration](#epsilon-greedy-exploration-)
{{% /expand %}}

### Upper Confidence Bound (UCB) 📊
{{% expand "Upper Confidence Bound (UCB)" %}}
The rule **Monte Carlo Tree Search** uses to pick which action to follow in its **select** phase. For each action $a$ at a node it scores $\frac{W_a}{N_a} + c\sqrt{\frac{\ln N_{\text{parent}}}{N_a}}$ and takes the highest: the first term is the action's **average return** so far (*exploit* — favor what has paid off), and the second is an **uncertainty bonus** (*explore*) that is large for actions tried few times ($N_a$ small) and shrinks as they are tried more. $N_a$ counts the visits to action $a$, $W_a$ sums their returns, $N_{\text{parent}} = \sum_{a'} N_{a'}$ is the node's total visits, and $c$ (the *exploration constant*, e.g. $1.4$) sets the balance. It is the tree-search cousin of **ε-greedy**, but it explores *where it is most uncertain* rather than at random — so an under-tried action can be selected even when its current average is lower.

**Appears in:** [Tutorial 3, Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Monte Carlo Tree Search](#monte-carlo-tree-search-), [Epsilon-Greedy Exploration](#epsilon-greedy-exploration-), [Simulation-Based RL](#simulation-based-rl-)
{{% /expand %}}

### Trajectory 📊
{{% expand "Trajectory" %}}
A single path the agent lives out: a sequence of states (and the rewards collected along the way) produced by starting in some state, following a policy to pick actions, and sampling each next state from the model. The Monte-Carlo value of a state is the average **discounted return** over many trajectories rolled out from it.

**Appears in:** [Tutorial 3, Chapter 21: Markov Decision Processes](../intro2/21_markov_decision_processes/)

**See also:** [Rollout](#rollout-), [Return](#return-), [Value Function](#value-function-), [Monte Carlo Simulation](#monte-carlo-simulation-)
{{% /expand %}}

### Rollout 📊
{{% expand "Rollout" %}}
Reinforcement learning's term for forward-simulating one **trajectory**: from a starting state, repeatedly pick an action (per the policy) and sample the next state from the model, stepping forward to a horizon or terminal state. "Roll out a trajectory" just means *generate one by simulation* — nothing more. In Monte Carlo Tree Search, *the rollout* often refers specifically to the simulation done with a random/default policy from a leaf node.

**Appears in:** [Tutorial 3, Chapter 21: Markov Decision Processes](../intro2/21_markov_decision_processes/) and [Chapter 22: Q-Learning](../intro2/22_q_learning/)

**See also:** [Trajectory](#trajectory-), [Monte Carlo Tree Search](#monte-carlo-tree-search-), [Simulation-Based RL](#simulation-based-rl-)
{{% /expand %}}

### Decision Rule 📊
{{% expand "Decision Rule" %}}
A strategy $d(x)$ that maps each possible **observation** $x$ to an action — the object statistical decision theory optimizes. A concrete example is a *threshold*: "eat the bento if it's $\le k$ days old, else compost." With no observation, a decision rule collapses to a single action; stretched across time, it generalizes to a **policy**.

**Appears in:** [Tutorial 3, Chapter 20: Statistical Decision Theory](../intro2/20_statistical_decision_theory/)

**See also:** [Statistical Decision Theory](#statistical-decision-theory-), [Observation](#observation-), [Policy](#policy-), [Bayes Estimator](#bayes-estimator-)
{{% /expand %}}

### Observation 📊
{{% expand "Observation" %}}
The data $x$ you get to see before acting (a sniff, a day-count, a measurement). Bayesian inference turns it into a posterior $p(\theta \mid x)$; decision theory then maps it to an action through a **decision rule** $d(x)$. Writing $x$ as a single observation is just for tidiness — conditioning on a whole batch $x_1, \dots, x_n$ changes nothing about the framework.

**Appears in:** [Tutorial 3, Chapter 20: Statistical Decision Theory](../intro2/20_statistical_decision_theory/)

**See also:** [Decision Rule](#decision-rule-), [Statistical Decision Theory](#statistical-decision-theory-), [Posterior Distribution](#posterior-distribution-)
{{% /expand %}}

### Inverse Reinforcement Learning 📊
{{% expand "Inverse Reinforcement Learning" %}}
Recovering the **reward or goal** behind observed behavior — running a planner backwards. Forward RL turns a goal into actions; inverse RL (IRL) watches actions and infers the goal, via Bayes' rule with a policy as the likelihood: $P(\text{goal}\mid\text{actions})\propto P(\text{actions}\mid\text{goal})\,P(\text{goal})$. Fundamentally **ill-posed** (many rewards fit), so a prior and a rationality assumption do the disambiguating.

**Appears in:** [Tutorial 3, Chapter 23: Inverse RL](../intro2/23_inverse_rl_goal_inference/)

**See also:** [Goal Inference](#goal-inference-), [Softmax Policy](#softmax-policy-), [Ill-Posed Problem](#ill-posed-problem-), [Theory of Mind](#theory-of-mind-)
{{% /expand %}}

### Goal Inference 📊
{{% expand "Goal Inference" %}}
The special case of inverse RL where the hidden cause is a discrete **goal**. Watch a few steps, score each candidate goal by the probability its softmax policy assigns the observed actions, and normalize to a posterior. The posterior **slides** as more behavior is seen — the basis of the Baker, Saxe & Tenenbaum (2009) "freeze the video, where's it headed?" experiments.

**Appears in:** [Tutorial 3, Chapter 23: Inverse RL](../intro2/23_inverse_rl_goal_inference/)

**See also:** [Inverse Reinforcement Learning](#inverse-reinforcement-learning-), [Theory of Mind](#theory-of-mind-), [Bayes' Theorem](#bayes-theorem-)
{{% /expand %}}

### Softmax Policy 📊
{{% expand "Softmax Policy" %}}
A **noisy-rational** policy that turns action values into action probabilities: $\pi(a\mid s)\propto e^{\beta Q(s,a)}$ (also called the Boltzmann policy). The best action is most likely but every action keeps some probability. As a likelihood for inverse RL, it is what lets a *detour* be informative.

**Appears in:** [Tutorial 3, Chapter 23: Inverse RL](../intro2/23_inverse_rl_goal_inference/)

**See also:** [Rationality (Inverse Temperature)](#rationality-inverse-temperature-), [Q-Learning](#q-learning-), [Inverse Reinforcement Learning](#inverse-reinforcement-learning-)
{{% /expand %}}

### Rationality (Inverse Temperature) 📊
{{% expand "Rationality (Inverse Temperature)" %}}
The parameter $\beta\ge 0$ in the softmax policy controlling **how rational** an agent is assumed to be. $\beta\to 0$ gives a random (coin-flipping) agent; $\beta\to\infty$ gives a greedy, pure-exploitation one. It is a **modeling assumption we choose**, not something inferred from the data — and it sets how much we read into behavior.

**Appears in:** [Tutorial 3, Chapter 23: Inverse RL](../intro2/23_inverse_rl_goal_inference/)

**See also:** [Softmax Policy](#softmax-policy-), [Inverse Reinforcement Learning](#inverse-reinforcement-learning-)
{{% /expand %}}

### Ill-Posed Problem 📊
{{% expand "Ill-Posed Problem" %}}
A problem whose solution is not uniquely determined by the data. Inverse RL is ill-posed: many rewards explain the same behavior (a flat reward makes every policy optimal; value-preserving reshaping leaves the policy unchanged). The **prior** and the **rationality assumption** are what pin down a single answer — not the behavior alone.

**Appears in:** [Tutorial 3, Chapter 23: Inverse RL](../intro2/23_inverse_rl_goal_inference/)

**See also:** [Inverse Reinforcement Learning](#inverse-reinforcement-learning-), [Reward Shaping](#reward-shaping-)
{{% /expand %}}

### Theory of Mind 📊
{{% expand "Theory of Mind" %}}
Attributing mental states — goals, beliefs, desires — to others to explain their behavior. The computational claim of this unit is that **Theory of Mind is inverse RL**: reading a mind is inverting a planner (Baker & Tenenbaum's inverse planning, reviewed by Jara-Ettinger 2019). Same Bayes-with-a-planner computation; only the name of the hidden cause changes.

**Appears in:** [Tutorial 3, Chapter 23: Inverse RL](../intro2/23_inverse_rl_goal_inference/), [Chapter 24: POMDPs](../intro2/24_pomdps_belief_inference/)

**See also:** [Inverse Reinforcement Learning](#inverse-reinforcement-learning-), [Bayesian Theory of Mind](#bayesian-theory-of-mind-), [ToMnet](#tomnet-)
{{% /expand %}}

### Maximum-Entropy IRL 📊
{{% expand "Maximum-Entropy IRL" %}}
The foundational scalable inverse-RL method (Ziebart 2008). It resolves IRL's ill-posedness by picking, among all reward-consistent trajectory distributions, the one of **maximum entropy** — the least-committal explanation — giving trajectory probability $P(\tau)\propto e^{\text{reward}(\tau)}$, the softmax-over-value form at trajectory scale.

**Appears in:** [Tutorial 3, Chapter 23: Inverse RL](../intro2/23_inverse_rl_goal_inference/)

**See also:** [Inverse Reinforcement Learning](#inverse-reinforcement-learning-), [GAIL](#gail-), [AIRL](#airl-)
{{% /expand %}}

### GAIL 📊
{{% expand "GAIL" %}}
**Generative Adversarial Imitation Learning** (Ho & Ermon 2016): casts imitation as a GAN — a discriminator separates expert from learner behavior, and the policy is trained to fool it. Scales imitation to high-dimensional control, but *skips* recovering a reward, so it yields no transferable objective.

**Appears in:** [Tutorial 3, Chapter 23: Inverse RL](../intro2/23_inverse_rl_goal_inference/)

**See also:** [Maximum-Entropy IRL](#maximum-entropy-irl-), [AIRL](#airl-)
{{% /expand %}}

### AIRL 📊
{{% expand "AIRL" %}}
**Adversarial Inverse RL** (Fu, Luo & Levine 2018): keeps GAIL's adversarial training but structures the discriminator so a **reward falls out**, disentangled from the dynamics — combining GAIL's scale with a *transferable* reward you can re-optimize under new dynamics.

**Appears in:** [Tutorial 3, Chapter 23: Inverse RL](../intro2/23_inverse_rl_goal_inference/)

**See also:** [GAIL](#gail-), [Maximum-Entropy IRL](#maximum-entropy-irl-)
{{% /expand %}}

### Belief State 📊
{{% expand "Belief State" %}}
In a POMDP, the agent never sees the world state, only noisy observations; its **belief** $b(s)=P(s\mid\text{history})$ is the posterior over the hidden state, updated by Bayes after each observation: $b'(s)\propto P(o\mid s)\,b(s)$. A belief is *just a probability*, and it is a **sufficient statistic** — it encodes everything the history says about the future.

**Appears in:** [Tutorial 3, Chapter 24: POMDPs](../intro2/24_pomdps_belief_inference/)

**See also:** [Partially Observable MDP (POMDP)](#partially-observable-mdp-pomdp-), [Belief MDP](#belief-mdp-), [Alpha-Vector](#alpha-vector-)
{{% /expand %}}

### Belief MDP 📊
{{% expand "Belief MDP" %}}
Because the belief is a sufficient statistic, a POMDP is equivalent to an ordinary MDP whose **state is the belief**: the belief simplex is the state space, the belief update is the transition, and the α-vectors give the (piecewise-linear, convex) value. Everything from MDPs transfers — it just runs on beliefs.

**Appears in:** [Tutorial 3, Chapter 24: POMDPs](../intro2/24_pomdps_belief_inference/)

**See also:** [Belief State](#belief-state-), [Markov Decision Process](#markov-decision-process-), [Alpha-Vector](#alpha-vector-)
{{% /expand %}}

### Alpha-Vector 📊
{{% expand "Alpha-Vector" %}}
The value of **committing** to one action, as a (linear) function of the belief — one line per action. Expected value is a weighted average, hence linear in the belief; the optimal value is the **upper envelope** of the action-lines, and its breakpoints are the decision thresholds (in the Tiger problem, open-right overtakes listen at belief $0.90$).

**Appears in:** [Tutorial 3, Chapter 24: POMDPs](../intro2/24_pomdps_belief_inference/)

**See also:** [Belief State](#belief-state-), [Tiger Problem](#tiger-problem-), [Belief MDP](#belief-mdp-)
{{% /expand %}}

### Tiger Problem 📊
{{% expand "Tiger Problem" %}}
The canonical POMDP (Kaelbling, Littman & Cassandra 1998): a tiger behind one of two doors, an 85%-accurate "listen," and rewards listen $-1$ / correct $+10$ / tiger $-100$. Repeated agreeing growls slide the belief $0.5\to 0.85\to 0.97$; opening beats listening once the belief crosses $0.90$. The cleanest illustration of belief updating and decision thresholds.

**Appears in:** [Tutorial 3, Chapter 24: POMDPs](../intro2/24_pomdps_belief_inference/)

**See also:** [Belief State](#belief-state-), [Alpha-Vector](#alpha-vector-), [Partially Observable MDP (POMDP)](#partially-observable-mdp-pomdp-)
{{% /expand %}}

### Bayesian Theory of Mind 📊
{{% expand "Bayesian Theory of Mind" %}}
Inverting a POMDP-planning agent to recover **both** what it wants (desire) and what it believes — which can be *false* (Baker, Jara-Ettinger, Saxe & Tenenbaum 2017). Explains detours that look irrational under known-state inference: the food-truck walker hedged because they *believed* the first truck might be closed.

**Appears in:** [Tutorial 3, Chapter 24: POMDPs](../intro2/24_pomdps_belief_inference/)

**See also:** [Theory of Mind](#theory-of-mind-), [Belief State](#belief-state-), [Inverse Reinforcement Learning](#inverse-reinforcement-learning-)
{{% /expand %}}

### Legibility 📊
{{% expand "Legibility" %}}
Acting so an observer can *read* your goal quickly — the flip of inverse planning (Dragan, Lee & Srinivasa 2013; Ho et al. 2016, "showing vs. doing"). A **legible** path resolves the observer's posterior early ($0.61$ vs $0.50$ on the first move) even when it is no longer than the **efficient** (predictable) one. Teaching is inverse planning run one level up.

**Appears in:** [Tutorial 3, Chapter 24: POMDPs](../intro2/24_pomdps_belief_inference/)

**See also:** [Goal Inference](#goal-inference-), [Cooperative Inverse RL](#cooperative-inverse-rl-)
{{% /expand %}}

### Cooperative Inverse RL 📊
{{% expand "Cooperative Inverse RL" %}}
**CIRL** (Hadfield-Menell et al. 2016): a human and robot in a shared world, both rewarded by the *human's* reward, which only the human knows. The robot infers it from behavior; the human, knowing this, should **teach**. Efficient expert demonstration is provably suboptimal, and CIRL reduces to a **POMDP whose hidden state is the human's reward** — alignment as a teaching game.

**Appears in:** [Tutorial 3, Chapter 24: POMDPs](../intro2/24_pomdps_belief_inference/)

**See also:** [Legibility](#legibility-), [Inverse Reinforcement Learning](#inverse-reinforcement-learning-), [RLHF](#rlhf-)
{{% /expand %}}

### RLHF 📊
{{% expand "RLHF" %}}
**Reinforcement Learning from Human Feedback**: collect pairwise human **preferences** between model outputs, fit a **reward model** to them (Bradley-Terry), then optimize the policy against that learned reward. The reward-modeling step is literally inverse RL — recover a hidden reward from human choices — which is why it aligns today's large language models.

**Appears in:** [Tutorial 3, Chapter 25: Modern RL](../intro2/25_modern_rl_world_models/)

**See also:** [DPO](#dpo-), [Reward Model](#reward-model-), [Bradley-Terry Model](#bradley-terry-model-), [Reward Hacking](#reward-hacking-)
{{% /expand %}}

### DPO 📊
{{% expand "DPO" %}}
**Direct Preference Optimization** (Rafailov et al. 2023): shows the optimal RLHF policy implies an *implicit* reward, so the policy can be optimized **directly** on preferences — folding the reward-model and policy-optimization steps into one. The underlying inference problem (recover a reward from preferences) is identical to RLHF's.

**Appears in:** [Tutorial 3, Chapter 25: Modern RL](../intro2/25_modern_rl_world_models/)

**See also:** [RLHF](#rlhf-), [Bradley-Terry Model](#bradley-terry-model-)
{{% /expand %}}

### Bradley-Terry Model 📊
{{% expand "Bradley-Terry Model" %}}
The standard choice model for pairwise preferences: $P(i\succ j)=\sigma(r_i-r_j)=\frac{e^{r_i}}{e^{r_i}+e^{r_j}}$ — a **pairwise softmax** over latent item rewards. Better items win more often, not always. Reward is identifiable only up to an additive constant, since preferences depend only on reward *differences*.

**Appears in:** [Tutorial 3, Chapter 25: Modern RL](../intro2/25_modern_rl_world_models/)

**See also:** [RLHF](#rlhf-), [Softmax Policy](#softmax-policy-), [Reward Model](#reward-model-)
{{% /expand %}}

### Reward Model 📊
{{% expand "Reward Model" %}}
A learned function that scores outputs by predicted human preference — the object RLHF fits from pairwise comparisons and then optimizes against. Because it only *approximates* human values, optimizing hard against it invites **reward hacking** (the policy finds high-scoring outputs that aren't actually good).

**Appears in:** [Tutorial 3, Chapter 25: Modern RL](../intro2/25_modern_rl_world_models/)

**See also:** [RLHF](#rlhf-), [Bradley-Terry Model](#bradley-terry-model-), [Reward Hacking](#reward-hacking-)
{{% /expand %}}

### Amortized Inference 📊
{{% expand "Amortized Inference" %}}
Paying the cost of inference *once*, up front, by training a network to map data straight to the answer — rather than running inference (enumeration, importance sampling) at query time. Fast but inherits its training distribution; the opposite tradeoff from exact Bayesian inference (interpretable, sample-efficient, but slow).

**Appears in:** [Tutorial 3, Chapter 25: Modern RL](../intro2/25_modern_rl_world_models/)

**See also:** [ToMnet](#tomnet-), [Importance Sampling](#importance-sampling-)
{{% /expand %}}

### ToMnet 📊
{{% expand "ToMnet" %}}
The "Theory-of-Mind network" (Rabinowitz et al. 2018): a neural net that watches many agents and **learns** to predict their behavior and (possibly false) beliefs in a single forward pass — the learned, scalable, but opaque cousin of explicit Bayesian Theory of Mind. Same inverse problem, traded from exact-but-slow to learned-but-opaque.

**Appears in:** [Tutorial 3, Chapter 25: Modern RL](../intro2/25_modern_rl_world_models/)

**See also:** [Theory of Mind](#theory-of-mind-), [Amortized Inference](#amortized-inference-), [Bayesian Theory of Mind](#bayesian-theory-of-mind-)
{{% /expand %}}

### World Model 📊
{{% expand "World Model" %}}
A learned model of an environment's dynamics, used to **plan by imagining** rollouts rather than acting in the costly real world — simulation-based RL with a *learned, compressed* model. A world model that tracks hidden state from partial observations is maintaining a **belief**, the POMDP machinery learned by a network.

**Appears in:** [Tutorial 3, Chapter 25: Modern RL](../intro2/25_modern_rl_world_models/)

**See also:** [MuZero](#muzero-), [Simulation-Based RL](#simulation-based-rl-), [Belief State](#belief-state-)
{{% /expand %}}

### MuZero 📊
{{% expand "MuZero" %}}
A world-model agent (Schrittwieser et al. 2020) that learns a **latent** model — a state space only rich enough to predict reward, value, and policy — and runs Monte-Carlo Tree Search inside it, mastering Go, chess, and Atari **without being told the rules**. The Chapter-22 simulation-based-RL idea with a learned, abstract model.

**Appears in:** [Tutorial 3, Chapter 25: Modern RL](../intro2/25_modern_rl_world_models/)

**See also:** [World Model](#world-model-), [Simulation-Based RL](#simulation-based-rl-), [Monte Carlo Tree Search](#monte-carlo-tree-search-)
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
