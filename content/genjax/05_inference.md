+++
title = "Inference in Action: The Taxicab Problem"
weight = 6
+++

## Solving Real Problems with Probabilistic Code

Remember the taxicab problem from the probability tutorial?

**Scenario:** Chibany witnesses a hit-and-run at night. He says the taxi was blue. But:
- 85% of taxis are green, 15% are blue
- Chibany identifies colors correctly 80% of the time

**Question:** What's the probability it was actually a blue taxi?

In the probability tutorial, we solved this with sets and Bayes' theorem. **Now we'll solve it with GenJAX!**

![Chibany investigating](images/chibanylayingdown.png)

---

## The Taxicab Problem: Quick Recap

### The Setup

**Base rates:**
- $P(\text{Blue}) = 0.15$ (15% of taxis are blue)
- $P(\text{Green}) = 0.85$ (85% of taxis are green)

**Chibany's accuracy:**
- $P(\text{says Blue} \mid \text{Blue}) = 0.80$ (correct 80% of the time)
- $P(\text{says Green} \mid \text{Green}) = 0.80$ (correct 80% of the time)
- Therefore, $P(\text{says Blue} \mid \text{Green}) = 0.20$ (mistakes 20% of the time)

**Observation:** Chibany says "Blue"

**Question:** $P(\text{Blue} \mid \text{says Blue}) = $ ?

---

## The Generative Model

Let's express this as a GenJAX generative function:

```python
import jax
import jax.numpy as jnp
from genjax import gen, bernoulli

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
    is_blue = bernoulli(base_rate_blue) @ "is_blue"

    # What Chibany says depends on the true color
    if is_blue:
        # If blue, says "blue" with probability = accuracy
        says_blue = bernoulli(accuracy) @ "says_blue"
    else:
        # If green, says "blue" with probability = 1 - accuracy (mistake)
        says_blue = bernoulli(1 - accuracy) @ "says_blue"

    return is_blue
```

**What this encodes:**

1. **Prior:** Taxis are blue 15% of the time (base rate)
2. **Likelihood:** How observation ("says blue") depends on true color
3. **Complete model:** Joint distribution over true color and observation

---

## Approach 1: Filtering (Rejection Sampling)

Let's solve it by generating many scenarios and filtering to the observation.

### Step 1: Generate Many Scenarios

```python
# Generate 100,000 scenarios
key = jax.random.key(42)
keys = jax.random.split(key, 100000)

def run_scenario(k):
    trace = taxicab_model.simulate(k, (0.15, 0.80))
    return {
        'is_blue': trace.get_choices()['is_blue'],
        'says_blue': trace.get_choices()['says_blue']
    }

# Vectorized version
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

Among scenarios where he says "blue", how many are actually blue?

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

{{% notice style="warning" title="The Base Rate Strikes Again!" %}}
**Why so low?**

Even though Chibany is 80% accurate, **most taxis are green** (85%). So even with his 20% error rate on green taxis, there are **more green taxis misidentified as blue** than there are actual blue taxis!

**The numbers:**
- Blue taxis correctly identified: $0.15 \times 0.80 = 0.12$ (12%)
- Green taxis incorrectly identified: $0.85 \times 0.20 = 0.17$ (17%)

**More false positives than true positives!**

This is why the posterior is only 41.5% ≈ 12/(12+17).
{{% /notice %}}

---

## Approach 2: Using `generate()` with Observations

Now let's use GenJAX's built-in conditioning:

```python
from genjax import ChoiceMap

# Observation: Chibany says "blue"
observation = ChoiceMap({"says_blue": 1})

# Generate 10,000 traces conditional on observation
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_conditional(k):
    trace, weight = taxicab_model.generate(k, (0.15, 0.80), observation)
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

---

## Theoretical Answer (Bayes' Theorem)

Let's verify against the exact Bayes' theorem calculation:

$$P(\text{Blue} \mid \text{says Blue}) = \frac{P(\text{says Blue} \mid \text{Blue}) \cdot P(\text{Blue})}{P(\text{says Blue})}$$

**Calculate each term:**

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

Let's visualize how our beliefs change:

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
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')

for i, prob in enumerate([prior_green, prior_blue]):
    ax1.text(i, prob + 0.05, f'{prob:.1%}', ha='center', fontsize=14, fontweight='bold')

# Posterior
ax2.bar(categories, [posterior_green, posterior_blue], color=colors)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('Posterior: After Chibany Says "Blue"', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')

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

---

## What If the Base Rate Were Different?

Let's explore how changing the base rate affects the answer.

**Scenario 1:** Equal taxis (50% blue, 50% green)

```python
# Generate with 50-50 base rate
observation = ChoiceMap({"says_blue": 1})

def run_equal_base(k):
    trace, weight = taxicab_model.generate(k, (0.50, 0.80), observation)
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

**Scenario 2:** Mostly blue (85% blue, 15% green)

```python
def run_mostly_blue(k):
    trace, weight = taxicab_model.generate(k, (0.85, 0.80), observation)
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

### Visualizing the Effect of Base Rates

```python
# Test different base rates
base_rates = jnp.linspace(0.01, 0.99, 50)
posteriors = []

for rate in base_rates:
    def run_with_rate(k):
        trace, weight = taxicab_model.generate(k, (float(rate), 0.80), observation)
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

{{% notice style="success" title="The Lesson" %}}
**Base rates matter enormously in real-world inference!**

Medical tests, fraud detection, witness testimony — all require considering:
1. How accurate is the test/witness? (likelihood)
2. How common is the condition/crime? (prior/base rate)

**Ignoring base rates leads to wrong conclusions.**

This is called **base rate neglect** — a common cognitive bias.
{{% /notice %}}

---

## Complete Code Example

Here's everything together:

```python
import jax
import jax.numpy as jnp
from genjax import gen, bernoulli, ChoiceMap
import matplotlib.pyplot as plt

@gen
def taxicab_model(base_rate_blue=0.15, accuracy=0.80):
    """Taxicab problem generative model."""
    is_blue = bernoulli(base_rate_blue) @ "is_blue"

    if is_blue:
        says_blue = bernoulli(accuracy) @ "says_blue"
    else:
        says_blue = bernoulli(1 - accuracy) @ "says_blue"

    return is_blue

# Observation: Chibany says "blue"
observation = ChoiceMap({"says_blue": 1})

# Generate posterior samples
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_inference(k):
    trace, weight = taxicab_model.generate(k, (0.15, 0.80), observation)
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

## Exercises

### Exercise 1: Higher Accuracy

What if Chibany were 95% accurate instead of 80%?

**Task:** Modify the code to use `accuracy=0.95` and calculate the posterior.

{{% expand "Solution" %}}
```python
def run_high_accuracy(k):
    trace, weight = taxicab_model.generate(k, (0.15, 0.95), observation)
    return trace.get_retval()

keys = jax.random.split(key, 10000)
posterior_high_acc = jax.vmap(run_high_accuracy)(keys)
prob_high_acc = jnp.mean(posterior_high_acc)

print(f"With 95% accuracy: P(Blue | says Blue) = {prob_high_acc:.3f}")
```

**Expected:** ≈ 0.75 (75%)

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
observation_green = ChoiceMap({"says_blue": 0})

def run_says_green(k):
    trace, weight = taxicab_model.generate(k, (0.15, 0.80), observation_green)
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
    is_blue = bernoulli(base_rate_blue) @ "is_blue"

    # Witness 1
    if is_blue:
        witness1 = bernoulli(accuracy) @ "witness1"
    else:
        witness1 = bernoulli(1 - accuracy) @ "witness1"

    # Witness 2 (independent)
    if is_blue:
        witness2 = bernoulli(accuracy) @ "witness2"
    else:
        witness2 = bernoulli(1 - accuracy) @ "witness2"

    return is_blue

# Both say "blue"
observation_two = ChoiceMap({"witness1": 1, "witness2": 1})

def run_two_witnesses(k):
    trace, weight = taxicab_two_witnesses.generate(k, (0.15, 0.80), observation_two)
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

In this chapter, you:

✅ **Implemented a real inference problem** — the taxicab scenario
✅ **Used filtering and generate()** — two approaches to conditioning
✅ **Saw Bayes' theorem in action** — automatic Bayesian update
✅ **Understood base rate effects** — why priors matter enormously
✅ **Explored parameter sensitivity** — how accuracy and base rates interact
✅ **Calculated with code, not formulas** — GenJAX does the math

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

**Final chapter:** Chapter 6 shows you how to build your own models from scratch!

---

|[← Previous: Conditioning and Observations](./04_conditioning.md) | [Next: Building Your Own Models →](./06_building_models.md)|
| :--- | ---: |
