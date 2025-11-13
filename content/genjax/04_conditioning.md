+++
title = "Conditioning and Observations"
weight = 5
+++

## From Simulation to Inference

So far, we've used GenJAX to **generate** outcomes — simulating what could happen.

Now we'll learn to **infer** — reasoning backwards from observations to causes.

This is the heart of probabilistic programming!

![Chibany thinking](images/chibanylayingdown.png)

---

## Recall: Conditional Probability

From the probability tutorial, remember **conditional probability**:

> **"Given that I observed $B$, what's the probability of $A$?"**

**Written:** $P(A \mid B)$

**Meaning:** Restrict the outcome space to only outcomes in $B$, then calculate the probability of $A$ within that restricted space.

**Formula:** $P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{|A \cap B|}{|B|}$

---

## Example from Set-Based Probability

Chibany's meals: $\Omega = \\{HH, HT, TH, TT\\}$

**Question:** "Given that dinner is Tonkatsu, what's the probability lunch was also Tonkatsu?"

**Set-based solution:**
1. Observe $D$ = "dinner is Tonkatsu" = $\\{HT, TT\\}$
2. Want $L$ = "lunch is Tonkatsu" = $\\{TH, TT\\}$
3. Intersection: $L \cap D = \\{TT\\}$
4. Conditional probability: $P(L \mid D) = \frac{|\\{TT\\}|}{|\\{HT, TT\\}|} = \frac{1}{2}$

**Key insight:** We **restricted the outcome space** from $\\{HH, HT, TH, TT\\}$ to just $\\{HT, TT\\}$ (outcomes where dinner = Tonkatsu).

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

## Conditional Probability in GenJAX

In GenJAX, we do the same thing — but with **code instead of sets**!

**Three approaches:**

### Approach 1: Filtering Simulations (Rejection Sampling)

Generate many traces, keep only those matching the observation.

**Pseudocode:**
```
1. Generate many traces
2. Keep only traces where observation is true
3. Among those, count how many satisfy the query
4. Calculate the ratio
```

This is **Monte Carlo conditional probability** — exactly what we did by hand with sets!

### Approach 2: Conditioning with `generate`

GenJAX has built-in support for specifying observations. We provide a **choice map** with the observed values, and GenJAX generates traces consistent with those observations.

### Approach 3: Full Inference (Importance Sampling, MCMC)

More advanced methods that we'll explore in Chapter 5. These are more efficient when observations are rare.

**This chapter focuses on Approach 1 and 2** — the most intuitive methods.

{{% notice style="success" title="📐→💻 Math-to-Code Translation" %}}
**How conditional probability translates to GenJAX:**

| Math Concept | Mathematical Notation | GenJAX Code |
|--------------|----------------------|-------------|
| **Conditional Probability** | $P(A \mid B)$ | `Target(model, (), observations)` |
| **Observation** | $B$ = "dinner is T" | `ChoiceMap.d({"dinner": 1})` |
| **Query** | $A$ = "lunch is T" | Check `trace["lunch"] == 1` |
| **Restriction** | Cross out outcomes where $B$ is false | Filter traces or use `Target` |

**The three approaches:**

| Approach | Math Equivalent | GenJAX Implementation |
|----------|----------------|----------------------|
| **1. Filtering** | Keep only outcomes in $B$, count $A$ | `traces[condition]` + count |
| **2. ChoiceMap** | Specify $B$ directly | `Target(model, (), observations)` |
| **3. Inference** | Weighted sampling from $P(A\mid B)$ | `target.importance(key, ...)` |

**Key insight:** All three compute the same conditional probability—they just differ in efficiency and how explicitly you specify the condition.
{{% /notice %}}

---

## Approach 1: Filtering Simulations

Let's answer: **"Given dinner is Tonkatsu, what's P(lunch is Tonkatsu)?"**

### Step 1: Generate Many Traces

```python
import jax
import jax.numpy as jnp
from genjax import gen, bernoulli

@gen
def chibany_day():
    lunch_is_tonkatsu = bernoulli(0.5) @ "lunch"
    dinner_is_tonkatsu = bernoulli(0.5) @ "dinner"
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)

# Generate 10,000 days
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_one_day(k):
    trace = chibany_day.simulate(k, ())
    return trace.get_retval()

days = jax.vmap(run_one_day)(keys)
```

### Step 2: Filter to Observation

**Observation:** Dinner is Tonkatsu (dinner = 1)

```python
# Filter: keep only days where dinner is Tonkatsu
dinner_is_tonkatsu = days[:, 1] == 1

# Count how many days match
n_matching = jnp.sum(dinner_is_tonkatsu)
print(f"Days where dinner is Tonkatsu: {n_matching} / {len(days)}")
```

**Output (example):**
```
Days where dinner is Tonkatsu: 4982 / 10000
```

**This is about 50%** — makes sense because dinner has 50% probability!

### Step 3: Query Among Filtered Traces

Among days where dinner is Tonkatsu, how many also have lunch as Tonkatsu?

```python
# Both meals are Tonkatsu
both_tonkatsu = (days[:, 0] == 1) & (days[:, 1] == 1)

# Count
n_both = jnp.sum(both_tonkatsu)

print(f"Days with both Tonkatsu: {n_both} / {n_matching}")
```

**Output (example):**
```
Days with both Tonkatsu: 2491 / 4982
```

### Step 4: Calculate Conditional Probability

```python
prob_lunch_given_dinner = n_both / n_matching
print(f"P(lunch=T | dinner=T) ≈ {prob_lunch_given_dinner:.3f}")
```

**Output:**
```
P(lunch=T | dinner=T) ≈ 0.500
```

**Perfect!** This matches the theoretical answer (0.5) because lunch and dinner are independent.

---

## Complete Example: Filtering

Here's the complete code:

```python
import jax
import jax.numpy as jnp
from genjax import gen, bernoulli

@gen
def chibany_day():
    lunch_is_tonkatsu = bernoulli(0.5) @ "lunch"
    dinner_is_tonkatsu = bernoulli(0.5) @ "dinner"
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)

# Generate simulations
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_one_day(k):
    trace = chibany_day.simulate(k, ())
    return trace.get_retval()

days = jax.vmap(run_one_day)(keys)

# Observation: dinner is Tonkatsu
observation_satisfied = days[:, 1] == 1

# Query: lunch is also Tonkatsu
query_satisfied = days[:, 0] == 1

# Both observation AND query
both_satisfied = observation_satisfied & query_satisfied

# Calculate conditional probability
n_observation = jnp.sum(observation_satisfied)
n_both = jnp.sum(both_satisfied)

prob_conditional = n_both / n_observation

print(f"=== Conditional Probability via Filtering ===")
print(f"Observation (dinner=T): {n_observation} traces")
print(f"Both (lunch=T AND dinner=T): {n_both} traces")
print(f"P(lunch=T | dinner=T) ≈ {prob_conditional:.3f}")
```

{{% notice style="success" title="The Pattern" %}}
**Conditional probability via filtering:**

1. **Generate** many traces
2. **Filter** to observations (keep only matching traces)
3. **Count** queries among filtered traces
4. **Divide** to get conditional probability

This is **rejection sampling** — the simplest form of inference!
{{% /notice %}}

---

## Approach 2: Conditioning with Choice Maps

GenJAX also lets you **specify observations** when generating traces.

### Creating a Choice Map

A **choice map** is a dictionary specifying values for named random choices:

```python
from genjax import ChoiceMap

# Specify that dinner must be Tonkatsu (1)
observations = ChoiceMap({
    "dinner": 1
})
```

### Generating with Observations

Use the `generate` function instead of `simulate`:

```python
key = jax.random.key(42)

# Generate a trace consistent with observations
trace, weight = chibany_day.generate(key, (), observations)

print(f"Lunch: {trace.get_choices()['lunch']}")
print(f"Dinner: {trace.get_choices()['dinner']}")
print(f"Weight: {weight}")
```

**Output (example):**
```
Lunch: 1
Dinner: 1  # Always 1 because we observed it!
Weight: -0.6931471805599453
```

**What's the weight?** It's the log probability of the observation. Here, $P(\text{dinner}=1) = 0.5$, so $\log(0.5) = -0.693...$

{{% notice style="info" title="generate() vs simulate()" %}}
**`simulate(key, args)`:**
- Generates a trace with all choices random
- No observations specified
- Returns just the trace

**`generate(key, args, observations)`:**
- Generates a trace consistent with observations
- Specified choices take given values
- Unspecified choices are random
- Returns `(trace, weight)` where weight = log probability of observations

**When to use which:**
- **Forward simulation** (no observations): Use `simulate()`
- **Conditional sampling** (some observations): Use `generate()`
{{% /notice %}}

---

## Generating Multiple Conditional Traces

Let's generate 1000 traces where dinner is Tonkatsu:

```python
from genjax import ChoiceMap

# Observation: dinner = Tonkatsu
observations = ChoiceMap({"dinner": 1})

# Generate many conditional traces
key = jax.random.key(42)
keys = jax.random.split(key, 1000)

def run_conditional(k):
    trace, weight = chibany_day.generate(k, (), observations)
    return trace.get_retval()

conditional_days = jax.vmap(run_conditional)(keys)

# Count lunch outcomes
lunch_tonkatsu = jnp.sum(conditional_days[:, 0] == 1)

print(f"Among {len(conditional_days)} days where dinner=Tonkatsu:")
print(f"  Lunch is Tonkatsu: {lunch_tonkatsu} ({lunch_tonkatsu/len(conditional_days):.1%})")
print(f"  Lunch is Hamburger: {len(conditional_days) - lunch_tonkatsu} ({(len(conditional_days) - lunch_tonkatsu)/len(conditional_days):.1%})")
```

**Output:**
```
Among 1000 days where dinner=Tonkatsu:
  Lunch is Tonkatsu: 501 (50.1%)
  Lunch is Hamburger: 499 (49.9%)
```

**Perfect!** Confirms $P(\text{lunch}=T \mid \text{dinner}=T) = 0.5$

---

## Connection to the Probability Tutorial

Let's revisit the exact example from Chapter 4 of the probability tutorial!

**Scenario:** Chibany observes that the student bringing his lunch said "He says it's from a place starting with T."

- If it's Tonkatsu, they'd definitely say "T" (P = 1.0)
- If it's Hamburger, they might still say "T" for "The Burger Place" (P = 0.3)

**Question:** What's the probability lunch is actually Tonkatsu?

### The Generative Model

```python
@gen
def lunch_with_clue():
    """Model lunch with a clue about the first letter."""

    # Prior: 50% chance of each meal
    is_tonkatsu = bernoulli(0.5) @ "is_tonkatsu"

    # Clue depends on the actual meal
    if is_tonkatsu:
        # If Tonkatsu, definitely says "T"
        says_t = bernoulli(1.0) @ "says_t"
    else:
        # If Hamburger, only 30% chance of saying "T"
        says_t = bernoulli(0.3) @ "says_t"

    return is_tonkatsu
```

### Prior (Before Hearing the Clue)

```python
# Generate without observations
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_prior(k):
    trace = lunch_with_clue.simulate(k, ())
    return trace.get_retval()

prior_samples = jax.vmap(run_prior)(keys)
prob_tonkatsu_prior = jnp.mean(prior_samples)

print(f"Prior: P(Tonkatsu) = {prob_tonkatsu_prior:.3f}")
```

**Output:**
```
Prior: P(Tonkatsu) = 0.500
```

**Makes sense!** Before hearing the clue, it's 50-50.

### Posterior (After Hearing "T")

```python
from genjax import ChoiceMap

# Observation: heard "T"
observations = ChoiceMap({"says_t": 1})

# Generate conditional on observation
def run_posterior(k):
    trace, weight = lunch_with_clue.generate(k, (), observations)
    return trace.get_retval()

keys = jax.random.split(key, 10000)
posterior_samples = jax.vmap(run_posterior)(keys)
prob_tonkatsu_posterior = jnp.mean(posterior_samples)

print(f"Posterior: P(Tonkatsu | heard 'T') = {prob_tonkatsu_posterior:.3f}")
```

**Output:**
```
Posterior: P(Tonkatsu | heard 'T') = 0.769
```

**Perfect!** This matches the theoretical answer from Bayes' theorem:

$$P(T \mid \text{heard "T"}) = \frac{P(\text{heard "T"} \mid T) \cdot P(T)}{P(\text{heard "T"})} = \frac{1.0 \times 0.5}{0.65} \approx 0.769$$

**The probability increased from 50% to 77%** after hearing the clue!

---

## Visualizing Prior vs Posterior

```python
import matplotlib.pyplot as plt

# Data
categories = ['Hamburger', 'Tonkatsu']
prior_probs = [1 - prob_tonkatsu_prior, prob_tonkatsu_prior]
posterior_probs = [1 - prob_tonkatsu_posterior, prob_tonkatsu_posterior]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Prior
ax1.bar(categories, prior_probs, color=['#ff6b6b', '#4ecdc4'])
ax1.set_ylabel('Probability')
ax1.set_title('Prior: Before Hearing Clue')
ax1.set_ylim(0, 1)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

for i, prob in enumerate(prior_probs):
    ax1.text(i, prob + 0.05, f'{prob:.1%}', ha='center', fontweight='bold')

# Posterior
ax2.bar(categories, posterior_probs, color=['#ff6b6b', '#4ecdc4'])
ax2.set_ylabel('Probability')
ax2.set_title('Posterior: After Hearing "T"')
ax2.set_ylim(0, 1)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

for i, prob in enumerate(posterior_probs):
    ax2.text(i, prob + 0.05, f'{prob:.1%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

**The visualization shows:** Evidence shifts our belief! We started at 50-50, but hearing "T" pushed us to 77% Tonkatsu.

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

## Exercises

### Exercise 1: Independent Variables

Verify that lunch and dinner are independent:

**Task:** Show that $P(\text{lunch}=T \mid \text{dinner}=T) = P(\text{lunch}=T)$

```python
# Generate unconditional samples (prior)
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_prior(k):
    trace = chibany_day.simulate(k, ())
    return trace.get_retval()

days = jax.vmap(run_prior)(keys)

# Prior: P(lunch=T)
prob_lunch_prior = jnp.mean(days[:, 0] == 1)

# Conditional: P(lunch=T | dinner=T)
dinner_t = days[:, 1] == 1
prob_lunch_given_dinner = jnp.sum((days[:, 0] == 1) & dinner_t) / jnp.sum(dinner_t)

print(f"P(lunch=T) = {prob_lunch_prior:.3f}")
print(f"P(lunch=T | dinner=T) = {prob_lunch_given_dinner:.3f}")
print(f"Independent: {abs(prob_lunch_prior - prob_lunch_given_dinner) < 0.05}")
```

{{% expand "Expected Output" %}}
```
P(lunch=T) = 0.500
P(lunch=T | dinner=T) = 0.500
Independent: True
```

**Conclusion:** Knowing dinner doesn't change lunch probability → independent!
{{% /expand %}}

---

### Exercise 2: Dependent Variables

Create a model where lunch and dinner are **not** independent:

**Scenario:** If lunch is Tonkatsu, Chibany wants variety for dinner (only 20% chance of Tonkatsu again). If lunch is Hamburger, he craves Tonkatsu for dinner (80% chance).

```python
@gen
def chibany_day_dependent():
    """Meals where dinner depends on lunch."""

    # Lunch is random (50-50)
    lunch_is_tonkatsu = bernoulli(0.5) @ "lunch"

    # Dinner depends on lunch!
    if lunch_is_tonkatsu:
        # Had Tonkatsu for lunch → wants variety
        dinner_is_tonkatsu = bernoulli(0.2) @ "dinner"
    else:
        # Had Hamburger for lunch → craves Tonkatsu
        dinner_is_tonkatsu = bernoulli(0.8) @ "dinner"

    return (lunch_is_tonkatsu, dinner_is_tonkatsu)
```

**Task:** Calculate $P(\text{lunch}=T \mid \text{dinner}=T)$ and compare to $P(\text{lunch}=T)$.

{{% expand "Solution" %}}
```python
# Prior: P(lunch=T)
keys = jax.random.split(key, 10000)

def run_prior(k):
    trace = chibany_day_dependent.simulate(k, ())
    return trace.get_retval()

days = jax.vmap(run_prior)(keys)
prob_lunch_prior = jnp.mean(days[:, 0] == 1)

# Conditional: P(lunch=T | dinner=T)
dinner_t = days[:, 1] == 1
prob_lunch_given_dinner = jnp.sum((days[:, 0] == 1) & dinner_t) / jnp.sum(dinner_t)

print(f"P(lunch=T) = {prob_lunch_prior:.3f}")
print(f"P(lunch=T | dinner=T) = {prob_lunch_given_dinner:.3f}")
print(f"Independent: {abs(prob_lunch_prior - prob_lunch_given_dinner) < 0.05}")
```

**Expected output:**
```
P(lunch=T) = 0.500
P(lunch=T | dinner=T) = 0.200
Independent: False
```

**Explanation:**
- Unconditionally, lunch is 50-50
- But if we know dinner=T, it's more likely lunch was H (because T→T is only 20%)!
- So $P(\text{lunch}=T \mid \text{dinner}=T) = 0.2 \neq 0.5 = P(\text{lunch}=T)$

**They're dependent!** Knowing dinner tells us about lunch.
{{% /expand %}}

---

### Exercise 3: Multiple Observations

Extend the lunch clue model to multiple observations:

**Scenario:**
1. Student says "It starts with T"
2. You smell the food and it smells fried (90% if Tonkatsu, 50% if Hamburger)

**Task:** Calculate $P(\text{Tonkatsu} \mid \text{says T AND smells fried})$

{{% expand "Solution" %}}
```python
@gen
def lunch_with_multiple_clues():
    """Model with two pieces of evidence."""

    is_tonkatsu = bernoulli(0.5) @ "is_tonkatsu"

    # Clue 1: What they say
    if is_tonkatsu:
        says_t = bernoulli(1.0) @ "says_t"
    else:
        says_t = bernoulli(0.3) @ "says_t"

    # Clue 2: Smell
    if is_tonkatsu:
        smells_fried = bernoulli(0.9) @ "smells_fried"
    else:
        smells_fried = bernoulli(0.5) @ "smells_fried"

    return is_tonkatsu

# Observations: both clues
from genjax import ChoiceMap
observations = ChoiceMap({
    "says_t": 1,
    "smells_fried": 1
})

# Generate posterior samples
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_posterior(k):
    trace, weight = lunch_with_multiple_clues.generate(k, (), observations)
    return trace.get_retval()

posterior = jax.vmap(run_posterior)(keys)
prob_tonkatsu = jnp.mean(posterior)

print(f"P(Tonkatsu | says T AND smells fried) = {prob_tonkatsu:.3f}")
```

**Expected:** Higher than 0.769 (from single clue) because we have more evidence!
{{% /expand %}}

---

## What You've Learned

In this chapter, you learned:

✅ **Conditional probability** — restriction to observations
✅ **Filtering approach** — rejection sampling for inference
✅ **`generate()` function** — conditioning with choice maps
✅ **Prior vs Posterior** — beliefs before and after data
✅ **Bayes' theorem in action** — automatic Bayesian update
✅ **Dependent vs Independent** — how observations provide information

**The key insight:** Probabilistic programming lets you **ask questions** instead of just **generate samples**!

---

## Next Steps

You now know how to:
- Generate samples (simulation)
- Condition on observations (inference)
- Calculate conditional probabilities

**Next up:** Chapter 5 applies these ideas to a real problem — the taxicab scenario from the probability tutorial!

You'll see how Bayes' theorem solves practical inference problems, and why base rates matter.

---

|[← Previous: Understanding Traces](./03_traces.md) | [Next: Inference in Action →](./05_inference.md)|
| :--- | ---: |
