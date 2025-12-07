+++
title = "Conditioning and Inference"
weight = 5
+++

## From Simulation to Inference

So far, we've used GenJAX to **generate** outcomes — simulating what could happen.

Now we'll learn to **infer** — reasoning backwards from observations to causes.

This is the heart of probabilistic programming!

![Chibany thinking](images/chibanylayingdown.png)

{{% notice style="tip" title="📓 Interactive Notebook Available" %}}
**Prefer hands-on learning?** This chapter has a companion **[Jupyter notebook - Open in Colab](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/bayesian_learning.ipynb)** that walks through Bayesian inference interactively with working code, visualizations, and exercises. You can work through the notebook first, then return here for the detailed explanations, or use them side-by-side!
{{% /notice %}}

---

## Recall: Conditional Probability

From the probability tutorial, remember **conditional probability**:

> **"Given that I observed $B$, what's the probability of $A$?"**

**Written:** $P(A \mid B)$

**Meaning:** Restrict the outcome space to only outcomes in $B$, then calculate the probability of $A$ within that restricted space.

**Formula:** $P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{|A \cap B|}{|B|}$

{{% notice style="info" title="📘 Foundation Concept: Conditioning as Restriction" %}}
**Recall from Tutorial 1, Chapter 4** that conditional probability means **restricting the outcome space**:

$$P(A \mid B) = \frac{|A \cap B|}{|B|}$$

**The key idea:** Cross out outcomes where $B$ didn't happen, then calculate probabilities in what remains.

**Tutorial 1 example:** "At least one tonkatsu" given "first meal was tonkatsu"
- Original space: {HH, HT, TH, TT}
- Condition: First meal is T → Restrict to {TH, TT}
- Event: At least one T → In restricted space: {TH, TT}
- Probability: 2/2 = 1 (both remaining outcomes have tonkatsu!)

**What GenJAX does:**
- Tutorial 1: Manually cross out outcomes and count
- Tutorial 2: Code filters simulations or uses `ChoiceMap` to restrict

**The logic is identical** — conditioning = restricting possibilities to match observations!

[← Review conditional probability in Tutorial 1, Chapter 4](../../intro/04_conditional/)
{{% /notice %}}

---

## The Taxicab Problem: A Real Inference Challenge

Let's apply these ideas to a real problem from the probability tutorial.

**Scenario:** Chibany witnesses a hit-and-run at night. They say the taxi was blue. But:
- 85% of taxis are green, 15% are blue
- Chibany identifies colors correctly 80% of the time

**Question:** What's the probability it was actually a blue taxi?

### Why This Is Surprising

Most people's intuition says: "Chibany is 80% accurate, so probably 80% chance it's blue."

**But the answer is only about 41%!**

Why? Because **most taxis are green**. Even with 80% accuracy, there are more green taxis misidentified as blue than there are actual blue taxis correctly identified.

This is **base rate neglect** — ignoring how common something is in the population.

Let's see how GenJAX helps us solve this!

---

## The Generative Model

First, we express the taxicab scenario as a GenJAX generative function:

```python
import jax
import jax.numpy as jnp
from genjax import gen, flip

@gen
def taxicab_model(base_rate_blue=0.15, accuracy=0.80):
    """Generate the taxi color and what Chibany says.

    Args:
        base_rate_blue: Probability a taxi is blue (default 0.15)
        accuracy: Probability Chibany identifies correctly (default 0.80)

    Returns:
        True if taxi is blue, False if green
    """

    # True taxi color (blue = 1, green = 0)
    is_blue = flip(base_rate_blue) @ "is_blue"

    # What Chibany says depends on the true color
    # Use jnp.where for JAX compatibility
    says_blue_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    says_blue = flip(says_blue_prob) @ "says_blue"

    return is_blue
```

**What this encodes:**

1. **Prior:** Taxis are blue 15% of the time (base rate)
2. **Likelihood:** How observation ("says blue") depends on true color
   - If blue: says "blue" 80% of the time (correct)
   - If green: says "blue" 20% of the time (mistake)
3. **Complete model:** Joint distribution over true color and observation

{{% notice style="info" title="📘 Foundation Concept: Bayes' Theorem in Code" %}}
**Recall from Tutorial 1, Chapter 5** that Bayes' Theorem updates beliefs with evidence:

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

**In the taxicab problem:**
- **Hypothesis (H):** Taxi is blue
- **Evidence (E):** Chibany says "blue"
- **Question:** $P(\text{blue} \mid \text{says blue})$ = ?

**Tutorial 1 approach (by hand):**
1. Calculate $P(\text{says blue} \mid \text{blue}) \cdot P(\text{blue}) = 0.80 \times 0.15 = 0.12$
2. Calculate $P(\text{says blue} \mid \text{green}) \cdot P(\text{green}) = 0.20 \times 0.85 = 0.17$
3. Calculate $P(\text{says blue}) = 0.12 + 0.17 = 0.29$
4. Apply Bayes: $P(\text{blue} \mid \text{says blue}) = \frac{0.12}{0.29} \approx 0.41$

**Tutorial 2 approach (GenJAX):**
1. **Define the generative model** (prior + likelihood)
2. **Specify observation** (says blue)
3. **Let GenJAX compute** the posterior automatically!

**The structure is identical:**
- `is_blue = flip(0.15)` → Prior: $P(\text{blue})$
- `says_blue_prob = jnp.where(is_blue, 0.80, 0.20)` → Likelihood: $P(\text{says blue} \mid \text{blue})$
- GenJAX conditioning → Computes posterior: $P(\text{blue} \mid \text{says blue})$

**Key insight:** GenJAX does all the Bayes' Theorem algebra for you! You just write the generative story (prior + likelihood), and conditioning gives you the posterior.

[← Review Bayes' Theorem in Tutorial 1, Chapter 5](../../intro/05_bayes/)
{{% /notice %}}

---

## Three Approaches to Inference

GenJAX provides three ways to compute conditional probabilities:

### Approach 1: Filtering (Rejection Sampling)

Generate many traces, keep only those matching the observation.

**Pseudocode:**
```
1. Generate many traces
2. Keep only traces where observation is true
3. Among those, count how many satisfy the query
4. Calculate the ratio
```

This is **Monte Carlo conditional probability** — exactly what we did by hand with sets!

### Approach 2: Conditioning with `generate()`

GenJAX has built-in support for specifying observations. We provide a **choice map** with the observed values, and GenJAX generates traces consistent with those observations.

### Approach 3: Full Inference (Importance Sampling, MCMC)

More advanced methods (beyond this tutorial). These are more efficient when observations are rare.

**This chapter focuses on Approach 1 and 2** — the most intuitive methods.

{{% notice style="success" title="📐→💻 Math-to-Code Translation" %}}
**How Bayesian inference translates to GenJAX:**

| Math Concept | Mathematical Notation | GenJAX Code |
|--------------|----------------------|-------------|
| **Prior** | $P(H)$ | `flip(0.15) @ "is_blue"` |
| **Likelihood** | $P(E \mid H)$ | `jnp.where(is_blue, 0.80, 0.20)` |
| **Evidence** | $P(E)$ | GenJAX computes automatically |
| **Posterior** | $P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)}$ | Result of conditioning |
| **Observation** | $E$ = "says blue" | `ChoiceMap.d({"says_blue": 1})` |
| **Inference Query** | $P(\text{is\_blue} \mid \text{says\_blue})$ | `mean(posterior_samples)` |

**Three equivalent inference approaches:**

| Approach | Mathematical Idea | GenJAX Implementation |
|----------|------------------|----------------------|
| **1. Filtering** | Sample from joint, keep only matching $E$ | Filter traces where `says_blue == 1` |
| **2. generate()** | Direct sampling from $P(H \mid E)$ | `model.generate(key, observation, args)` |
| **3. importance()** | Weighted sampling | `target.importance(key, n_particles)` |

**Key insights:**
- **Generative model = Prior + Likelihood** — The @gen function encodes both
- **Conditioning = Computing posterior** — GenJAX does the Bayes' theorem math
- **All three methods compute the same thing** — They just differ in efficiency
- **Base rates matter!** — Prior P(H) heavily influences posterior P(H|E)
{{% /notice %}}

---

## Approach 1: Filtering (Rejection Sampling)

Let's solve the taxicab problem by generating many scenarios and filtering to the observation.

### Step 1: Generate Many Scenarios

```python
# Generate 100,000 scenarios
key = jax.random.key(42)
keys = jax.random.split(key, 100000)

def run_scenario_vec(k):
    trace = taxicab_model.simulate(k, (0.15, 0.80))
    choices = trace.get_choices()
    return jnp.array([choices['is_blue'], choices['says_blue']])

scenarios = jax.vmap(run_scenario_vec)(keys)
is_blue = scenarios[:, 0]
says_blue = scenarios[:, 1]
```

### Step 2: Filter to Observation

**Observation:** Chibany says "blue"

```python
# Keep only scenarios where Chibany says "blue"
observation_satisfied = says_blue == 1

n_says_blue = jnp.sum(observation_satisfied)
print(f"Scenarios where Chibany says blue: {n_says_blue} / {len(scenarios)}")
```

**Output (example):**
```
Scenarios where Chibany says blue: 29017 / 100000
```

**Why ~29%?**
- $P(\text{says Blue}) = P(\text{Blue}) \cdot P(\text{says Blue} \mid \text{Blue}) + P(\text{Green}) \cdot P(\text{says Blue} \mid \text{Green})$
- $= 0.15 \times 0.80 + 0.85 \times 0.20 = 0.12 + 0.17 = 0.29$

### Step 3: Count True Positives

Among scenarios where they say "blue", how many are actually blue?

```python
# Both says blue AND is blue
both_blue = observation_satisfied & (is_blue == 1)

n_actually_blue = jnp.sum(both_blue)
print(f"Scenarios where taxi IS blue: {n_actually_blue} / {n_says_blue}")
```

**Output (example):**
```
Scenarios where taxi IS blue: 12038 / 29017
```

### Step 4: Calculate Posterior

```python
prob_blue_given_says_blue = n_actually_blue / n_says_blue
print(f"\nP(Blue | says Blue) ≈ {prob_blue_given_says_blue:.3f}")
```

**Output:**
```
P(Blue | says Blue) ≈ 0.415
```

**Only 41.5%!** Even though Chibany is 80% accurate, there's less than 50% chance the taxi was actually blue!

{{% notice style="warning" title="The Base Rate Strikes!" %}}
**Why so low?**

Even though Chibany is 80% accurate, **most taxis are green** (85%). So even with his 20% error rate on green taxis, there are **more green taxis misidentified as blue** than there are actual blue taxis!

**The numbers:**
- Blue taxis correctly identified: $0.15 \times 0.80 = 0.12$ (12%)
- Green taxis incorrectly identified: $0.85 \times 0.20 = 0.17$ (17%)

**More false positives than true positives!**

This is why the posterior is only 41.5% ≈ 12/(12+17).
{{% /notice %}}

{{% notice style="success" title="The Filtering Pattern" %}}
**Conditional probability via filtering:**

1. **Generate** many traces
2. **Filter** to observations (keep only matching traces)
3. **Count** queries among filtered traces
4. **Divide** to get conditional probability

This is **rejection sampling** — the simplest form of inference!
{{% /notice %}}

---

## Approach 2: Using `generate()` with Observations

Now let's use GenJAX's built-in conditioning. This is usually more convenient!

### Creating a Choice Map and Generating Conditional Traces

A **choice map** is a dictionary specifying values for named random choices. We use it to condition the model on observations:

```python
from genjax import ChoiceMap

# Specify that Chibany says "blue"
observation = ChoiceMap.d({"says_blue": 1})

# Generate 10,000 traces conditional on observation
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_conditional(k):
    trace, weight = taxicab_model.generate(k, observation, (0.15, 0.80))
    return trace.get_retval()  # Returns is_blue

posterior_samples = jax.vmap(run_conditional)(keys)

# Calculate posterior probability
prob_blue_posterior = jnp.mean(posterior_samples)
print(f"P(Blue | says Blue) ≈ {prob_blue_posterior:.3f}")
```

**Output:**
```
P(Blue | says Blue) ≈ 0.414
```

**Same answer!** Both methods work — `generate()` is just more convenient.

{{% notice style="info" title="generate() vs simulate()" %}}
**`simulate(key, args)`:**
- Generates a trace with all choices random
- No observations specified
- Returns just the trace

**`generate(key, observations, args)`:**
- Generates a trace consistent with observations
- Specified choices take given values
- Unspecified choices are random
- Returns `(trace, weight)` where weight = log probability of observations

**When to use which:**
- **Forward simulation** (no observations): Use `simulate()`
- **Conditional sampling** (some observations): Use `generate()`
{{% /notice %}}

---

## Theoretical Verification

Let's verify our simulation against exact Bayes' theorem calculation:

```python
# Prior
P_blue = 0.15
P_green = 0.85

# Likelihood
P_says_blue_given_blue = 0.80
P_says_blue_given_green = 0.20

# Evidence (total probability of saying blue)
P_says_blue = (P_blue * P_says_blue_given_blue +
               P_green * P_says_blue_given_green)

# Posterior (Bayes' theorem)
P_blue_given_says_blue = (P_says_blue_given_blue * P_blue) / P_says_blue

print(f"=== Bayes' Theorem Calculation ===")
print(f"P(Blue) = {P_blue}")
print(f"P(says Blue | Blue) = {P_says_blue_given_blue}")
print(f"P(says Blue | Green) = {P_says_blue_given_green}")
print(f"P(says Blue) = {P_says_blue}")
print(f"\nP(Blue | says Blue) = {P_blue_given_says_blue:.3f}")
```

**Output:**
```
=== Bayes' Theorem Calculation ===
P(Blue) = 0.15
P(says Blue | Blue) = 0.8
P(says Blue | Green) = 0.2
P(says Blue) = 0.29

P(Blue | says Blue) = 0.414
```

**Perfect match!** GenJAX simulation ≈ 0.415, Bayes' theorem exact = 0.414

---

## Visualizing Prior vs Posterior

Let's visualize how evidence changes our beliefs:

```python
import matplotlib.pyplot as plt

# Prior: before observation
prior_blue = 0.15
prior_green = 0.85

# Posterior: after observation
posterior_blue = prob_blue_posterior  # From simulation
posterior_green = 1 - posterior_blue

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

categories = ['Green', 'Blue']
colors = ['#4ecdc4', '#6c5ce7']

# Prior
ax1.bar(categories, [prior_green, prior_blue], color=colors)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title('Prior: Before Chibany Speaks', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

for i, prob in enumerate([prior_green, prior_blue]):
    ax1.text(i, prob + 0.05, f'{prob:.1%}', ha='center', fontsize=14, fontweight='bold')

# Posterior
ax2.bar(categories, [posterior_green, posterior_blue], color=colors)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('Posterior: After Chibany Says "Blue"', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

for i, prob in enumerate([posterior_green, posterior_blue]):
    ax2.text(i, prob + 0.05, f'{prob:.1%}', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n📊 Belief Update:")
print(f"   Before: P(Blue) = {prior_blue:.1%}")
print(f"   After:  P(Blue | says Blue) = {posterior_blue:.1%}")
print(f"   Change: +{(posterior_blue - prior_blue):.1%}")
```

**Key insight:** Evidence increased our belief in blue from 15% to 41%, but **still not even 50%** because the base rate is so strong!

![Prior vs Posterior probability distributions](../../images/genjax/taxicab_prior_posterior.png)

---

## Exploring Base Rate Effects

Let's see how changing the base rate affects the answer.

### Scenario 1: Equal Taxis (50% blue, 50% green)

```python
observation = ChoiceMap.d({"says_blue": 1})

def run_equal_base(k):
    trace, weight = taxicab_model.generate(k, observation, (0.50, 0.80))
    return trace.get_retval()

keys = jax.random.split(key, 10000)
posterior_equal = jax.vmap(run_equal_base)(keys)
prob_equal = jnp.mean(posterior_equal)

print(f"If 50% blue: P(Blue | says Blue) = {prob_equal:.3f}")
```

**Output:**
```
If 50% blue: P(Blue | says Blue) = 0.800
```

**Now it's 80%!** When base rates are equal, accuracy dominates.

### Scenario 2: Mostly Blue (85% blue, 15% green)

```python
def run_mostly_blue(k):
    trace, weight = taxicab_model.generate(k, observation, (0.85, 0.80))
    return trace.get_retval()

posterior_mostly_blue = jax.vmap(run_mostly_blue)(keys)
prob_mostly_blue = jnp.mean(posterior_mostly_blue)

print(f"If 85% blue: P(Blue | says Blue) = {prob_mostly_blue:.3f}")
```

**Output:**
```
If 85% blue: P(Blue | says Blue) = 0.971
```

**Now it's 97%!** When most taxis are blue, seeing "blue" is strong evidence.

### Visualizing the Effect

```python
# Test different base rates
base_rates = jnp.linspace(0.01, 0.99, 50)
posteriors = []

for rate in base_rates:
    def run_with_rate(k):
        trace, weight = taxicab_model.generate(k, observation, (float(rate), 0.80))
        return trace.get_retval()

    keys = jax.random.split(key, 1000)
    post = jax.vmap(run_with_rate)(keys)
    posteriors.append(jnp.mean(post))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(base_rates, posteriors, linewidth=2, color='#6c5ce7')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
plt.axvline(x=0.15, color='red', linestyle='--', alpha=0.7, label='Original problem (15%)')
plt.scatter([0.15], [0.414], color='red', s=100, zorder=5)

plt.xlabel('Base Rate: P(Blue)', fontsize=12)
plt.ylabel('Posterior: P(Blue | says Blue)', fontsize=12)
plt.title('How Base Rates Affect Inference\n(Chibany 80% accurate)', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

**The graph shows:** Even with high accuracy (80%), the posterior depends heavily on the base rate!

![How base rates affect inference](../../images/genjax/taxicab_base_rate_effect.png)

{{% notice style="success" title="The Lesson" %}}
**Base rates matter enormously in real-world inference!**

Medical tests, fraud detection, witness testimony — all require considering:
1. How accurate is the test/witness? (likelihood)
2. How common is the condition/crime? (prior/base rate)

**Ignoring base rates leads to wrong conclusions.**

This is called **base rate neglect** — a common cognitive bias.
{{% /notice %}}

---

## Key Concepts

### Prior Distribution
**What we believe before seeing data.**

- Generated by `simulate()` (no observations)
- Represents our initial uncertainty

### Posterior Distribution
**What we believe after seeing data.**

- Generated by `generate()` with observations
- Represents updated beliefs incorporating evidence

### Bayes' Theorem in Action

GenJAX automatically handles the math:

$$P(\text{hypothesis} \mid \text{data}) = \frac{P(\text{data} \mid \text{hypothesis}) \cdot P(\text{hypothesis})}{P(\text{data})}$$

**You just:**
1. Define the generative model (encodes $P(\text{data} \mid \text{hypothesis})$ and $P(\text{hypothesis})$)
2. Specify observations (the data)
3. Generate conditional traces (GenJAX computes the posterior)

**No manual Bayes' rule calculation needed!**

{{% notice style="success" title="The Power of Generative Models" %}}
When you write a generative function, you're specifying:
- **Prior:** The distribution of random choices before observations
- **Likelihood:** How observations depend on hidden variables
- **Joint distribution:** The complete probabilistic model

GenJAX handles the inference automatically!
{{% /notice %}}

---

## Complete Example Code

Here's everything together for easy copying:

```python
import jax
import jax.numpy as jnp
from genjax import gen, flip, ChoiceMap
import matplotlib.pyplot as plt

@gen
def taxicab_model(base_rate_blue=0.15, accuracy=0.80):
    """Taxicab problem generative model."""
    is_blue = flip(base_rate_blue) @ "is_blue"

    # What Chibany says depends on true color
    says_blue_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    says_blue = flip(says_blue_prob) @ "says_blue"

    return is_blue

# Observation: Chibany says "blue"
observation = ChoiceMap.d({"says_blue": 1})

# Generate posterior samples
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_inference(k):
    trace, weight = taxicab_model.generate(k, observation, (0.15, 0.80))
    return trace.get_retval()

posterior_samples = jax.vmap(run_inference)(keys)
prob_blue = jnp.mean(posterior_samples)

print(f"=== TAXICAB INFERENCE ===")
print(f"Base rate: 15% blue")
print(f"Accuracy: 80%")
print(f"Observation: Says 'blue'")
print(f"\nP(Blue | says Blue) ≈ {prob_blue:.3f}")
```

---

## Interactive Exploration

{{% notice style="tip" title="📓 Interactive Notebook: Bayesian Learning" %}}
Want to explore Bayesian learning in depth with interactive examples? Check out the **[Bayesian Learning notebook - Open in Colab](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/bayesian_learning.ipynb)** which covers:

- The complete taxicab problem with visualizations
- Sequential Bayesian updating with multiple observations
- Interactive sliders to explore different base rates and accuracies
- How base rates affect posterior beliefs

This notebook lets you experiment with the concepts you just learned!
{{% /notice %}}

---

## Exercises

### Exercise 1: Higher Accuracy

What if Chibany were 95% accurate instead of 80%?

**Task:** Modify the code to use `accuracy=0.95` and calculate the posterior.

{{% expand "Solution" %}}
```python
def run_high_accuracy(k):
    trace, weight = taxicab_model.generate(k, observation, (0.15, 0.95))
    return trace.get_retval()

keys = jax.random.split(key, 10000)
posterior_high_acc = jax.vmap(run_high_accuracy)(keys)
prob_high_acc = jnp.mean(posterior_high_acc)

print(f"With 95% accuracy: P(Blue | says Blue) = {prob_high_acc:.3f}")
```

**Expected:** ≈ 0.77 (77%)

**Much higher!** Accuracy matters, but even at 95%, base rates still pull it below 100%.

**Theoretical:**
$$P = \frac{0.95 \times 0.15}{0.95 \times 0.15 + 0.05 \times 0.85} = \frac{0.1425}{0.1850} \approx 0.770$$
{{% /expand %}}

---

### Exercise 2: Opposite Observation

What if Chibany said "green" instead of "blue"?

**Task:** Calculate $P(\text{Blue} \mid \text{says Green})$

{{% expand "Solution" %}}
```python
# Observation: says "green"
observation_green = ChoiceMap.d({"says_blue": 0})

def run_says_green(k):
    trace, weight = taxicab_model.generate(k, observation_green, (0.15, 0.80))
    return trace.get_retval()

keys = jax.random.split(key, 10000)
posterior_green = jax.vmap(run_says_green)(keys)
prob_blue_given_green = jnp.mean(posterior_green)

print(f"P(Blue | says Green) = {prob_blue_given_green:.3f}")
```

**Expected:** ≈ 0.041 (4.1%)

**Very low!** If Chibany (80% accurate) says "green", it's very likely green.

**Theoretical:**
$$P = \frac{0.20 \times 0.15}{0.20 \times 0.15 + 0.80 \times 0.85} = \frac{0.03}{0.71} \approx 0.042$$
{{% /expand %}}

---

### Exercise 3: Two Witnesses

What if **two independent witnesses** both say "blue"?

**Task:** Extend the model to include two witnesses, both 80% accurate. Calculate the posterior.

{{% expand "Solution" %}}
```python
@gen
def taxicab_two_witnesses(base_rate_blue=0.15, accuracy=0.80):
    """Two independent witnesses."""
    is_blue = flip(base_rate_blue) @ "is_blue"

    # Witness 1
    witness1_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    witness1 = flip(witness1_prob) @ "witness1"

    # Witness 2 (independent)
    witness2_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    witness2 = flip(witness2_prob) @ "witness2"

    return is_blue

# Both say "blue"
observation_two = ChoiceMap.d({"witness1": 1, "witness2": 1})

def run_two_witnesses(k):
    trace, weight = taxicab_two_witnesses.generate(k, observation_two, (0.15, 0.80))
    return trace.get_retval()

keys = jax.random.split(key, 10000)
posterior_two = jax.vmap(run_two_witnesses)(keys)
prob_two = jnp.mean(posterior_two)

print(f"P(Blue | both say Blue) = {prob_two:.3f}")
```

**Expected:** ≈ 0.73 (73%)

**Much higher!** Two independent pieces of evidence are much stronger.

**Theoretical:**
$$P(\text{both say Blue} \mid \text{Blue}) = 0.80^2 = 0.64$$
$$P(\text{both say Blue} \mid \text{Green}) = 0.20^2 = 0.04$$
$$P = \frac{0.64 \times 0.15}{0.64 \times 0.15 + 0.04 \times 0.85} = \frac{0.096}{0.130} \approx 0.738$$

**Two witnesses push us above 50% despite the low base rate!**
{{% /expand %}}

---

## What You've Learned

In this chapter, you learned:

✅ **Conditional probability** — restriction to observations

✅ **Filtering approach** — rejection sampling for inference

✅ **`generate()` function** — conditioning with choice maps

✅ **Prior vs Posterior** — beliefs before and after data

✅ **Bayes' theorem in action** — automatic Bayesian update

✅ **Base rate effects** — why priors matter enormously

✅ **Real inference problems** — the taxicab scenario

**The key insight:** Probabilistic programming lets you **encode assumptions** (generative model) and **ask questions** (conditioning) without manual Bayes' rule calculations!

---

## Why This Matters

**Real-world applications:**

1. **Medical diagnosis:** Test accuracy + disease prevalence → probability of disease
2. **Fraud detection:** Transaction patterns + fraud base rate → probability of fraud
3. **Spam filtering:** Email features + spam base rate → probability of spam
4. **Criminal justice:** Witness accuracy + crime base rate → probability of guilt

**All follow the same pattern:**
- Define generative model (how data arises)
- Observe data
- Infer hidden causes

**GenJAX makes this systematic and scalable.**

---

## Next Steps

You now know:
- How to build generative models
- How to perform inference with observations
- How to interpret posterior probabilities
- Why base rates matter

**Next up:** Chapter 6 shows you how to build your own models from scratch!

---

|[← Previous: Understanding Traces](./03_traces.md) | [Next: Building Your Own Models →](./06_building_models.md)|
| :--- | ---: |
