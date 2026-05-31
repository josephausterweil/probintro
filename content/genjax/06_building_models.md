+++
date = "2026-05-31"
title = "Building Your Own Models"
weight = 6
+++

## From Following Recipes to Creating Your Own

You've learned to use GenJAX through examples. Now it's time to build your **own** probabilistic models!

This chapter shows you **how to think** about building generative models — turning real-world problems into code.

![Chibany ready to build](images/chibanyplain.png)

---

## The Model-Building Process

### Step 1: Understand the Problem

**Before writing any code, answer:**

1. **What am I trying to predict or understand?** (The question)
2. **What do I observe?** (The data/evidence)
3. **What's hidden?** (The unknown variables)
4. **How are they related?** (The causal structure)

**Example:** Spam detection
1. **Question:** Is this email spam?
2. **Observations:** Email content, sender, time
3. **Hidden:** True spam status
4. **Relationship:** Spam emails have certain word patterns

---

### Step 2: Sketch the Generative Story

**Write out the process that generates the data:**

"First, nature chooses..., then based on that, it generates..., which produces..."

**Example:** Coin flips
1. First, the coin has a (hidden) bias parameter
2. Based on that bias, each flip is heads or tails
3. We observe a sequence of flips

**This narrative becomes your code!**

---

### Step 3: Choose Distributions

**For each random choice, pick a distribution:**

| Type of Variable | Common Distributions |
|------------------|---------------------|
| Binary (yes/no) | `flip(p)` |
| Categorical (A/B/C) | `categorical(probs)` |
| Count (0, 1, 2, ...) | `poisson(rate)` |
| Continuous | `normal(mean, std)`, `uniform(low, high)` |

**Start simple!** Use `flip` for most binary choices.

---

### Step 4: Write the Code

**Pattern:**

<!-- validate: skip -->
```python
@gen
def my_model(parameters):
    # Hidden variables (causes)
    hidden = distribution(...) @ "hidden"

    # Observed variables (effects)
    # Usually depend on hidden variables
    if hidden:
        observed = distribution_A(...) @ "observed"
    else:
        observed = distribution_B(...) @ "observed"

    return hidden  # Or whatever you want to predict
```

**Key points:**
- Use `@gen` decorator
- Name all random choices with `@ "name"`
- Return what you want to infer
- Use `if` statements to model dependencies

---

### Step 5: Test and Validate

1. **Generate samples** — does the output look reasonable?
2. **Check extreme cases** — what if parameters are 0 or 1?
3. **Verify inference** — do posterior results make intuitive sense?

{{% notice style="success" title="📐→💻 Math-to-Code Translation" %}}
**How model-building concepts translate to GenJAX:**

| Math Concept | Mathematical Notation | GenJAX Pattern |
|--------------|----------------------|----------------|
| **Joint Distribution** | $P(X, Y)$ | Multiple `flip()` calls in @gen function |
| **Conditional Distribution** | $P(Y \mid X)$ | `if X: Y = flip(p1)` |
| **Independence** | $P(X, Y) = P(X) \cdot P(Y)$ | Separate random choices (no if statements) |
| **Dependence** | $P(Y \mid X) \neq P(Y)$ | Y's distribution uses X in if statement |
| **Hierarchical Model** | $\theta \sim \text{Prior}, X \mid \theta$ | Parameter as random variable: `theta = uniform() @ "theta"` |
| **Mixture Model** | $\sum_k P(Z=k) P(X \mid Z=k)$ | `if category == k: X = distribution_k()` |
| **Sequence Model** | $P(X_t \mid X_{t-1})$ | Loop with prev_state dependency |

**Common modeling patterns:**

| Pattern | Probability Structure | Code Structure |
|---------|---------------------|----------------|
| **Independent observations** | $P(X_1, \ldots, X_n) = \prod P(X_i)$ | `for i: X_i = flip()` |
| **Hierarchical** | $P(\theta) P(X \mid \theta)$ | `theta = uniform(); X = flip(theta)` |
| **Conditional** | $P(Y \mid X)$ depends on X | `if X: Y = flip(p1) else: Y = flip(p2)` |
| **Time series** | $P(X_t \mid X_{t-1})$ | `for t: X[t] = flip(f(X[t-1]))` |
| **Mixture** | $\sum_k \pi_k P(X \mid k)$ | `k = categorical(pi); if k==0: ... else: ...` |

**Key insights:**
- **@gen function = Joint distribution** — Defines P(all variables)
- **if statements = Conditional dependence** — Y depends on X
- **for loops = Repeated structure** — Multiple observations or time steps
- **Parameters as random variables = Hierarchical** — Uncertainty at multiple levels
- **Your generative story = The math** — If you can describe how data is generated, you can code it

**Example: Medical diagnosis**
```
Math: P(Disease, Fever, Cough) = P(Disease) × P(Fever|Disease) × P(Cough|Disease)
Code: has_disease = flip(0.01) @ "disease"
      fever_prob = jnp.where(has_disease, 0.9, 0.1)
      cough_prob = jnp.where(has_disease, 0.8, 0.2)
      fever = flip(fever_prob) @ "fever"
      cough = flip(cough_prob) @ "cough"
```
{{% /notice %}}

---

## Common Patterns

### Pattern 1: Independent Observations

**Scenario:** Multiple independent measurements

**Example:** Coin flips

```python
import jax
import jax.numpy as jnp
from genjax import gen, flip, uniform, categorical, ChoiceMap

# The number of flips is a Python constant captured by the @gen, NOT a model
# argument. JAX traces model arguments into abstract values, so a Python
# `for i in range(n)` loop can't use a count that arrives as a traced argument
# (it raises TracerIntegerConversionError). Fixing the count as a module-level
# constant keeps the teaching loop readable and runnable.
N_FLIPS = 10

@gen
def coin_flips(bias=0.5):
    """Generate N_FLIPS independent coin flips."""

    results = []
    for i in range(N_FLIPS):
        # Each flip is independent
        result = flip(bias) @ f"flip_{i}"
        results.append(result)

    return jnp.array(results).astype(int)
```

{{% notice style="note" title="Why is the count a constant, not an argument?" %}}
JAX traces a model's arguments into *abstract* values before running it, so a
Python loop like `for i in range(n)` can't use an `n` that comes in as a model
argument — it raises `TracerIntegerConversionError`. The fix is to make the
count a Python constant the `@gen` closes over (as `N_FLIPS` here), or to wrap
the model in a factory function that takes the count and returns a `@gen` (the
pattern Tutorial 3's DPMM chapter uses). Either way the loop itself is unchanged.
{{% /notice %}}

**Usage:**

<!-- validate: skip-output -->
```python
key = jax.random.key(42)
trace = coin_flips.simulate(key, (0.7,))
flips = trace.get_retval()
print(f"Flips: {flips}")
```

**Output (example):**
```
Flips: [0 1 1 0 1 1 1 1 1 1]
```

---

### Pattern 2: Hierarchical Structure

**Scenario:** Parameters have their own distributions

**Example:** Learning a coin's bias from flips

```python
@gen
def coin_with_unknown_bias():
    """Coin with unknown bias — infer it from N_FLIPS flips."""

    # Hidden: the coin's true bias (uniform between 0 and 1)
    bias = uniform(0.0, 1.0) @ "bias"

    # Observations: flip outcomes (N_FLIPS is the module constant from above)
    flips = []
    for i in range(N_FLIPS):
        result = flip(bias) @ f"flip_{i}"
        flips.append(result)

    return bias  # Want to infer this!
```

**Inference:**

<!-- validate: tol=0.1 -->
```python
# Observe 7 heads out of 10 flips
observations = ChoiceMap.d({
    "flip_0": 1, "flip_1": 1, "flip_2": 0,
    "flip_3": 1, "flip_4": 1, "flip_5": 0,
    "flip_6": 1, "flip_7": 1, "flip_8": 0,
    "flip_9": 1
})

# Infer bias
key = jax.random.key(42)
keys = jax.random.split(key, 1000)

def infer_bias(k):
    # generate(key, CONSTRAINTS, ARGS) — the model takes no args, so ()
    trace, weight = coin_with_unknown_bias.generate(k, observations, ())
    return trace.get_retval(), weight

results = jax.vmap(infer_bias)(keys)
posterior_bias = results[0]
weights = results[1]

# Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
mean_bias = jnp.sum(posterior_bias * normalized_weights)

print(f"Estimated bias: {mean_bias:.2f}")
# Should be around 0.70 (7 heads / 10 flips)
```

**Output:**
```
Estimated bias: 0.66
```

---

### Pattern 3: Conditional Dependencies

**Scenario:** Observations depend on hidden state

**Example:** Weather affects mood

```python
import jax.numpy as jnp

@gen
def mood_model():
    """Weather affects Chibany's mood."""

    # Hidden: today's weather
    is_sunny = flip(0.7) @ "is_sunny"  # 70% sunny days

    # Observable: Chibany's mood depends on weather
    # Sunny → happy 90% of the time, Rainy → happy only 30% of the time
    happy_prob = jnp.where(is_sunny, 0.9, 0.3)
    is_happy = flip(happy_prob) @ "is_happy"

    return is_sunny
```

**Question:** "Chibany is happy. What's the probability it's sunny?"

<!-- validate: tol=0.02 -->
```python
observation = ChoiceMap.d({"is_happy": 1})

def infer_weather(k):
    # generate(key, CONSTRAINTS, ARGS)
    trace, weight = mood_model.generate(k, observation, ())
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 10000)
results = jax.vmap(infer_weather)(keys)
posterior_sunny = results[0]
weights = results[1]

# Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
prob_sunny = jnp.sum(posterior_sunny * normalized_weights)

print(f"P(Sunny | Happy) ≈ {prob_sunny:.3f}")
```

**Output:**
```
P(Sunny | Happy) ≈ 0.873
```

{{% expand "Theoretical Answer" %}}
Using Bayes' theorem:

$$P(\text{Sunny} \mid \text{Happy}) = \frac{P(\text{Happy} \mid \text{Sunny}) \cdot P(\text{Sunny})}{P(\text{Happy})}$$

- $P(\text{Sunny}) = 0.7$
- $P(\text{Happy} \mid \text{Sunny}) = 0.9$
- $P(\text{Happy} \mid \text{Rainy}) = 0.3$
- $P(\text{Happy}) = 0.7 \times 0.9 + 0.3 \times 0.3 = 0.63 + 0.09 = 0.72$

$$P = \frac{0.9 \times 0.7}{0.72} = \frac{0.63}{0.72} \approx 0.875$$

**Expected:** ≈ 87.5%
{{% /expand %}}

---

### Pattern 4: Sequences and Time Series

**Scenario:** Events unfold over time

**Example:** Chibany's weekly meals

<!-- validate: skip-output -->
```python
# Like N_FLIPS above, the number of days is a Python constant, not a model arg.
DAYS = 7

@gen
def weekly_meals():
    """Model a week of meals with memory."""

    meals = []

    # First day is random
    prev_meal = flip(0.5) @ "day_0"
    meals.append(prev_meal)

    # Each subsequent day depends on the previous day. prev_meal is a traced
    # value, so we pick the probability with jnp.where (not a Python if):
    #   tonkatsu yesterday (1) → want variety → 0.3; hamburger (0) → craving → 0.8
    for day in range(1, DAYS):
        current_prob = jnp.where(prev_meal == 1, 0.3, 0.8)
        current_meal = flip(current_prob) @ f"day_{day}"
        meals.append(current_meal)
        prev_meal = current_meal

    return jnp.array(meals).astype(int)

# Simulate one week
meals = weekly_meals.simulate(jax.random.key(0), ()).get_retval()
print(f"Week of meals (1=tonkatsu, 0=hamburger): {meals}")
```

**Output (example):**
```
Week of meals (1=tonkatsu, 0=hamburger): [1 0 1 1 0 0 1]
```

**This models dependence through time!**

---

### Pattern 5: Mixture Models

**Scenario:** Data comes from multiple sources, but which source is not observed

**Example:** Two types of days (weekday vs weekend). Chibany doesn't know what day it is. Bentos on the weekend are much more likely to have tonkatsu.

```python
import jax.numpy as jnp

@gen
def mixed_days():
    """Different behavior on weekends vs weekdays."""

    # Hidden: is it a weekend?
    is_weekend = flip(2/7) @ "is_weekend"  # 2 out of 7 days

    # Weekend: high chance of tonkatsu (relaxed), Weekday: lower chance (busy)
    tonkatsu_prob = jnp.where(is_weekend, 0.9, 0.3)
    lunch = flip(tonkatsu_prob) @ "lunch"

    return is_weekend
```

**Infer:** "Given Chibany had tonkatsu, is it a weekend?"

---

## Building a Complete Model: Medical Diagnosis

Let's build a realistic example from scratch.

**Scenario:** Diagnosing a disease based on symptoms

**Setup:**
- Disease prevalence: 1% (rare)
- Symptom 1 (fever): 90% if diseased, 10% if healthy
- Symptom 2 (cough): 80% if diseased, 20% if healthy

**Question:** Patient has fever and cough. Probability of disease?

### Step 1: Understand the Problem

- **Question:** Does patient have disease?
- **Observations:** Fever and cough
- **Hidden:** True disease status
- **Relationships:** Symptoms more likely if diseased

### Step 2: Generative Story

1. First, patient either has disease (1%) or not (99%)
2. If diseased, fever is very likely (90%)
3. If diseased, cough is very likely (80%)
4. If healthy, both symptoms are rare (10%, 20%)

### Step 3: Write the Model

```python
import jax.numpy as jnp

@gen
def disease_model(prevalence=0.01, fever_if_disease=0.9, cough_if_disease=0.8,
                  fever_if_healthy=0.1, cough_if_healthy=0.2):
    """Medical diagnosis model."""

    # Hidden: disease status
    has_disease = flip(prevalence) @ "has_disease"

    # Symptoms depend on disease status
    fever_prob = jnp.where(has_disease, fever_if_disease, fever_if_healthy)
    cough_prob = jnp.where(has_disease, cough_if_disease, cough_if_healthy)
    fever = flip(fever_prob) @ "fever"
    cough = flip(cough_prob) @ "cough"

    return has_disease
```

### Step 4: Run Inference

<!-- validate: tol=0.05 -->
```python
# Patient has both symptoms
observation = ChoiceMap.d({"fever": 1, "cough": 1})

def infer_disease(k):
    # generate(key, CONSTRAINTS, ARGS)
    trace, weight = disease_model.generate(k, observation, ())
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 10000)
results = jax.vmap(infer_disease)(keys)
posterior = results[0]
weights = results[1]

# Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
prob_disease = jnp.sum(posterior * normalized_weights)

print(f"=== MEDICAL DIAGNOSIS ===")
print(f"Prevalence: 1%")
print(f"Symptoms: Fever + Cough")
print(f"P(Disease | Symptoms) ≈ {prob_disease:.3f}")
```

**Output:**
```
=== MEDICAL DIAGNOSIS ===
Prevalence: 1%
Symptoms: Fever + Cough
P(Disease | Symptoms) ≈ 0.269
```

**Expected:** ≈ 0.265 (26.5%)

**Interpretation:** Even with both symptoms, only 26.5% chance of disease because it's so rare!

{{% notice style="warning" title="Base Rate Neglect in Medicine!" %}}
**This is why false positives are a problem in medical testing.**

Even accurate tests produce many false positives for rare diseases because:
- True positives: $0.01 \times 0.9 \times 0.8 = 0.0072$ (0.72%)
- False positives: $0.99 \times 0.1 \times 0.2 = 0.0198$ (1.98%)

**More false positives than true positives!**

This is why doctors don't diagnose based on symptoms alone — they need confirmatory tests or consider patient history (updating the prior).
{{% /notice %}}

---

## Best Practices

### ✅ DO

#### 1. Name everything clearly

<!-- validate: skip -->
```python
# Good
is_diseased = flip(0.01) @ "is_diseased"

# Bad
x = flip(0.01) @ "x"
```

#### 2. Use meaningful parameters

<!-- validate: skip -->
```python
# Good
@gen
def model(disease_prevalence=0.01, test_accuracy=0.95):
    ...

# Bad
@gen
def model(p1=0.01, p2=0.95):
    ...
```

#### 3. Document your model

<!-- validate: skip -->
```python
@gen
def weather_mood(sunny_prior=0.7):
    """Model how weather affects mood.

    Args:
        sunny_prior: Base rate of sunny days (default 0.7)

    Returns:
        is_sunny: Whether it's sunny today
    """
```

#### 4. Start simple, add complexity

- Build the simplest model first
- Verify it works
- Add features incrementally

#### 5. Test edge cases

- What if parameters are 0? 1?
- What if all observations are the same?
- Does the posterior make intuitive sense?

---

### ❌ DON'T

#### 1. Don't forget to name random choices

<!-- validate: skip -->
```python
# Bad — can't condition on this!
x = flip(0.5)

# Good
x = flip(0.5) @ "x"
```

#### 2. Don't use the same name twice

<!-- validate: skip -->
```python
# Bad — name collision!
flip1 = flip(0.5) @ "flip"
flip2 = flip(0.5) @ "flip"  # ERROR!

# Good — unique names
flip1 = flip(0.5) @ "flip_1"
flip2 = flip(0.5) @ "flip_2"
```

#### 3. Don't overthink distributions

- `flip` covers most binary cases
- `normal` for continuous
- `categorical` for multiple choices
- You don't need exotic distributions to start!

#### 4. Don't skip validation

- Always generate samples first
- Check if outputs look reasonable
- Verify extreme parameter values

---

## Exercises

### Exercise 1: Email Spam Filter

Build a simple spam filter model.

**Scenario:**
- 30% of emails are spam
- Spam emails contain "FREE" 80% of the time
- Legitimate emails contain "FREE" 10% of the time

**Task:** Calculate $P(\text{Spam} \mid \text{contains "FREE"})$

{{% expand "Solution" %}}
<!-- validate: tol=0.05 -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, flip, ChoiceMap

@gen
def spam_filter(spam_rate=0.30):
    """Simple spam filter based on keyword."""

    # Hidden: is it spam?
    is_spam = flip(spam_rate) @ "is_spam"

    # Observation: contains "FREE"?
    contains_free_prob = jnp.where(is_spam, 0.80, 0.10)
    contains_free = flip(contains_free_prob) @ "contains_free"

    return is_spam

# Email contains "FREE"
observation = ChoiceMap.d({"contains_free": 1})

def infer_spam(k):
    # generate(key, CONSTRAINTS, ARGS)
    trace, weight = spam_filter.generate(k, observation, ())
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 10000)
results = jax.vmap(infer_spam)(keys)
posterior = results[0]
weights = results[1]

# Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
prob_spam = jnp.sum(posterior * normalized_weights)

print(f"P(Spam | contains 'FREE') ≈ {prob_spam:.3f}")
```

**Output:**
```
P(Spam | contains 'FREE') ≈ 0.777
```

**Expected:** ≈ 0.774 (77.4%)

**Theoretical:**
$$P = \frac{0.80 \times 0.30}{0.80 \times 0.30 + 0.10 \times 0.70} = \frac{0.24}{0.31} \approx 0.774$$
{{% /expand %}}

---

### Exercise 2: Learning from Multiple Observations

Extend the coin flip model to infer bias from multiple observations.

**Task:** Given a sequence of 20 flips (e.g., `[1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1]`), infer the coin's bias.

{{% expand "Solution" %}}
<!-- validate: tol=0.1 -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, flip, uniform, ChoiceMap

# Fixed count as a module constant (see the note at the top of the chapter).
N_OBSERVED = 20

@gen
def coin_model():
    """Infer coin bias from N_OBSERVED observed flips."""

    # Hidden: coin's true bias
    bias = uniform(0.0, 1.0) @ "bias"

    # Observations: flips
    for i in range(N_OBSERVED):
        result = flip(bias) @ f"flip_{i}"

    return bias

# Observed flips: 16 heads out of 20
observed_flips = [1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1]
observations = ChoiceMap.d({f"flip_{i}": observed_flips[i] for i in range(N_OBSERVED)})

def infer_bias(k):
    # generate(key, CONSTRAINTS, ARGS)
    trace, weight = coin_model.generate(k, observations, ())
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 1000)
results = jax.vmap(infer_bias)(keys)
posterior_bias = results[0]
weights = results[1]

# Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
mean_bias = jnp.sum(posterior_bias * normalized_weights)
# For standard deviation with weighted samples
variance = jnp.sum(normalized_weights * (posterior_bias - mean_bias)**2)
std_bias = jnp.sqrt(variance)

print(f"Estimated bias: {mean_bias:.2f} ± {std_bias:.2f}")
# Should be around 0.80 (16/20)
```

**Output:**
```
Estimated bias: 0.77 ± 0.09
```

**Expected:** Mean ≈ 0.80, with some uncertainty

**Plot the posterior:**
```python
import matplotlib.pyplot as plt

plt.hist(posterior_bias, bins=50, density=True, alpha=0.7, color='#4ecdc4')
plt.axvline(mean_bias, color='red', linestyle='--', label=f'Mean = {mean_bias:.2f}')
plt.xlabel('Coin Bias')
plt.ylabel('Posterior Density')
plt.title('Posterior Distribution of Coin Bias\n(16 heads in 20 flips)')
plt.legend()
plt.show()
```
{{% /expand %}}

---

### Exercise 3: Multi-Symptom Diagnosis

Extend the disease model to include 3 symptoms: fever, cough, fatigue.

**Parameters:**
- Disease: 2% prevalence
- If diseased: fever 90%, cough 80%, fatigue 95%
- If healthy: fever 10%, cough 20%, fatigue 30%

**Task:** Calculate posterior for:
1. Fever only
2. Fever + cough
3. All three symptoms

{{% expand "Solution" %}}
<!-- validate: tol=0.05 -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, flip, ChoiceMap

@gen
def disease_three_symptoms(prevalence=0.02):
    """Disease model with three symptoms."""

    has_disease = flip(prevalence) @ "has_disease"

    # Symptoms depend on disease status
    fever_prob = jnp.where(has_disease, 0.90, 0.10)
    cough_prob = jnp.where(has_disease, 0.80, 0.20)
    fatigue_prob = jnp.where(has_disease, 0.95, 0.30)
    fever = flip(fever_prob) @ "fever"
    cough = flip(cough_prob) @ "cough"
    fatigue = flip(fatigue_prob) @ "fatigue"

    return has_disease

key = jax.random.key(42)

# Scenario 1: Fever only
obs1 = ChoiceMap.d({"fever": 1})

# Scenario 2: Fever + cough
obs2 = ChoiceMap.d({"fever": 1, "cough": 1})

# Scenario 3: All three
obs3 = ChoiceMap.d({"fever": 1, "cough": 1, "fatigue": 1})

for i, obs in enumerate([obs1, obs2, obs3], 1):
    def infer(k, obs=obs):
        # generate(key, CONSTRAINTS, ARGS); obs bound per-iteration
        trace, weight = disease_three_symptoms.generate(k, obs, ())
        return trace.get_retval(), weight

    keys = jax.random.split(key, 10000)
    results = jax.vmap(infer)(keys)
    posterior = results[0]
    weights = results[1]

    # Use importance sampling
    normalized_weights = jnp.exp(weights - jnp.max(weights))
    normalized_weights = normalized_weights / jnp.sum(normalized_weights)
    prob = jnp.sum(posterior * normalized_weights)

    print(f"Scenario {i}: P(Disease) ≈ {prob:.3f}")
```

**Output:**
```
Scenario 1: P(Disease) ≈ 0.164
Scenario 2: P(Disease) ≈ 0.439
Scenario 3: P(Disease) ≈ 0.713
```

**Insight:** More evidence → higher posterior! (Fever only → fever + cough → all three symptoms.)
{{% /expand %}}

---

## What You've Learned

In this chapter, you learned:

✅ **The model-building process** — from problem to code
✅ **Common patterns** — independent, hierarchical, conditional, sequential, mixture
✅ **Best practices** — naming, documentation, testing
✅ **Complete examples** — medical diagnosis, spam filtering, coin flipping
✅ **How to think generatively** — "what generates the data?"

**The key insight:** Building models is about **encoding your assumptions** about how the world works, then letting GenJAX do the inference!

---

## Next Steps

### You're Ready to Build!

You now have all the tools to:
- Build generative models for your problems
- Perform Bayesian inference automatically
- Understand uncertainty in your predictions

**Where to go from here:**

### 1. Explore More Distributions

GenJAX supports many distributions beyond `flip`:

- `normal(mean, std)` — Continuous values (heights, weights, temperatures)
- `categorical(probs)` — Multiple discrete choices (A, B, C, D)
- `poisson(rate)` — Count data (number of events)
- `gamma`, `beta`, `exponential` — Specialized continuous distributions

**See the GenJAX documentation** for complete reference.

### 2. Learn Advanced Inference

This tutorial covered:
- Filtering/rejection sampling
- Conditioning with `generate()`

**Next level:**
- Importance sampling (more efficient for rare events)
- Markov Chain Monte Carlo (MCMC) for complex models
- Variational inference (approximate but fast)

**Check out:** GenJAX advanced tutorials

### 3. Real-World Applications

Apply what you learned to:
- **Science:** Modeling experiments, analyzing data
- **Medicine:** Diagnosis, treatment optimization
- **Engineering:** Fault detection, quality control
- **Social science:** Understanding human behavior
- **AI/ML:** Building better models with uncertainty

---

## The Journey

**You started with:** Sets, counting, basic probability

**Now you can:** Build probabilistic programs, perform Bayesian inference, reason under uncertainty

**That's a huge accomplishment!**

---

## Final Thoughts

Probabilistic programming is a **superpower**:

1. **Express uncertainty** — the world is uncertain, our models should reflect that
2. **Automate inference** — computers do the hard math
3. **Combine knowledge and data** — use both domain expertise (priors) and observations (data)
4. **Make better decisions** — understand risks and probabilities

**Keep building, keep learning, keep questioning!**

---

## Chapter Complete!

You've learned how to build your own probabilistic models from scratch. This is the final chapter of the GenJAX programming tutorial.

**What you accomplished in this tutorial:**
- Set up your GenJAX environment
- Learned essential Python for probabilistic programming
- Built generative models with the `@gen` decorator
- Understood traces and how GenJAX records execution
- Conditioned models on observations
- Performed inference to answer probabilistic questions
- Created complete models for real-world problems

**You're ready for the next step!**

---

## What's Next: Continuous Probability & Bayesian Learning

So far, you've worked with **discrete** random variables (coin flips, categories, yes/no outcomes). But many real-world quantities are **continuous** — heights, temperatures, waiting times.

In **Tutorial 3: Continuous Probability & Bayesian Learning**, you'll:

- Work with continuous distributions (normal, exponential, etc.)
- Learn about Bayesian updating with continuous parameters
- Build mixture models for clustering
- Explore the Dirichlet Process for infinite mixtures

**The probabilistic programming skills you've learned here will transfer directly!**

[**Continue to Tutorial 3: Continuous Probability →**](/probintro/intro2/)

---

|[← Previous: Inference in Action](./05_inference.md) | [Tutorial 3: Continuous Probability →](/probintro/intro2/)|
| :--- | ---: |
