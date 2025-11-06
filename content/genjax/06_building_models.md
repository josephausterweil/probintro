+++
title = "Building Your Own Models"
weight = 7
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
| Binary (yes/no) | `bernoulli(p)` |
| Categorical (A/B/C) | `categorical(probs)` |
| Count (0, 1, 2, ...) | `poisson(rate)` |
| Continuous | `normal(mean, std)`, `uniform(low, high)` |

**Start simple!** Use `bernoulli` for most binary choices.

---

### Step 4: Write the Code

**Pattern:**

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

---

## Common Patterns

### Pattern 1: Independent Observations

**Scenario:** Multiple independent measurements

**Example:** Coin flips

```python
@gen
def coin_flips(n_flips, bias=0.5):
    """Generate n independent coin flips."""

    results = []
    for i in range(n_flips):
        # Each flip is independent
        flip = bernoulli(bias) @ f"flip_{i}"
        results.append(flip)

    return jnp.array(results)
```

**Usage:**

```python
key = jax.random.key(42)
trace = coin_flips.simulate(key, (10, 0.7))
flips = trace.get_retval()
print(f"Flips: {flips}")
```

---

### Pattern 2: Hierarchical Structure

**Scenario:** Parameters have their own distributions

**Example:** Learning a coin's bias from flips

```python
@gen
def coin_with_unknown_bias(n_flips):
    """Coin with unknown bias — infer it from flips."""

    # Hidden: the coin's true bias (uniform between 0 and 1)
    bias = uniform(0.0, 1.0) @ "bias"

    # Observations: flip outcomes
    flips = []
    for i in range(n_flips):
        flip = bernoulli(bias) @ f"flip_{i}"
        flips.append(flip)

    return bias  # Want to infer this!
```

**Inference:**

```python
from genjax import ChoiceMap

# Observe 7 heads out of 10 flips
observations = ChoiceMap({
    "flip_0": 1, "flip_1": 1, "flip_2": 0,
    "flip_3": 1, "flip_4": 1, "flip_5": 0,
    "flip_6": 1, "flip_7": 1, "flip_8": 0,
    "flip_9": 1
})

# Infer bias
keys = jax.random.split(key, 1000)

def infer_bias(k):
    trace, weight = coin_with_unknown_bias.generate(k, (10,), observations)
    return trace.get_retval()

posterior_bias = jax.vmap(infer_bias)(keys)
mean_bias = jnp.mean(posterior_bias)

print(f"Estimated bias: {mean_bias:.2f}")
# Should be around 0.70 (7 heads / 10 flips)
```

---

### Pattern 3: Conditional Dependencies

**Scenario:** Observations depend on hidden state

**Example:** Weather affects mood

```python
@gen
def mood_model():
    """Weather affects Chibany's mood."""

    # Hidden: today's weather
    is_sunny = bernoulli(0.7) @ "is_sunny"  # 70% sunny days

    # Observable: Chibany's mood depends on weather
    if is_sunny:
        # Sunny → happy 90% of the time
        is_happy = bernoulli(0.9) @ "is_happy"
    else:
        # Rainy → happy only 30% of the time
        is_happy = bernoulli(0.3) @ "is_happy"

    return is_sunny
```

**Question:** "Chibany is happy. What's the probability it's sunny?"

```python
observation = ChoiceMap({"is_happy": 1})

def infer_weather(k):
    trace, weight = mood_model.generate(k, (), observation)
    return trace.get_retval()

keys = jax.random.split(key, 10000)
posterior_sunny = jax.vmap(infer_weather)(keys)
prob_sunny = jnp.mean(posterior_sunny)

print(f"P(Sunny | Happy) ≈ {prob_sunny:.3f}")
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

```python
@gen
def weekly_meals(days=7):
    """Model a week of meals with memory."""

    meals = []

    # First day is random
    prev_meal = bernoulli(0.5) @ "day_0"
    meals.append(prev_meal)

    # Each subsequent day depends on previous day
    for day in range(1, days):
        if prev_meal == 1:  # Had tonkatsu yesterday
            # Want variety → lower probability
            current_meal = bernoulli(0.3) @ f"day_{day}"
        else:  # Had hamburger yesterday
            # Craving tonkatsu → higher probability
            current_meal = bernoulli(0.8) @ f"day_{day}"

        meals.append(current_meal)
        prev_meal = current_meal

    return jnp.array(meals)
```

**This models dependence through time!**

---

### Pattern 5: Mixture Models

**Scenario:** Data comes from multiple sources

**Example:** Two types of days (weekday vs weekend)

```python
@gen
def mixed_days():
    """Different behavior on weekends vs weekdays."""

    # Hidden: is it a weekend?
    is_weekend = bernoulli(2/7) @ "is_weekend"  # 2 out of 7 days

    if is_weekend:
        # Weekend: high chance of tonkatsu (relaxed)
        tonkatsu_prob = 0.9
    else:
        # Weekday: lower chance (busy)
        tonkatsu_prob = 0.3

    lunch = bernoulli(tonkatsu_prob) @ "lunch"

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
@gen
def disease_model(prevalence=0.01, fever_if_disease=0.9, cough_if_disease=0.8,
                  fever_if_healthy=0.1, cough_if_healthy=0.2):
    """Medical diagnosis model."""

    # Hidden: disease status
    has_disease = bernoulli(prevalence) @ "has_disease"

    # Symptoms depend on disease
    if has_disease:
        fever = bernoulli(fever_if_disease) @ "fever"
        cough = bernoulli(cough_if_disease) @ "cough"
    else:
        fever = bernoulli(fever_if_healthy) @ "fever"
        cough = bernoulli(cough_if_healthy) @ "cough"

    return has_disease
```

### Step 4: Run Inference

```python
# Patient has both symptoms
observation = ChoiceMap({"fever": 1, "cough": 1})

def infer_disease(k):
    trace, weight = disease_model.generate(k, (), observation)
    return trace.get_retval()

keys = jax.random.split(key, 10000)
posterior = jax.vmap(infer_disease)(keys)
prob_disease = jnp.mean(posterior)

print(f"=== MEDICAL DIAGNOSIS ===")
print(f"Prevalence: 1%")
print(f"Symptoms: Fever + Cough")
print(f"P(Disease | Symptoms) ≈ {prob_disease:.3f}")
```

**Expected output:** ≈ 0.265 (26.5%)

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

1. **Name everything clearly**
   ```python
   # Good
   is_diseased = bernoulli(0.01) @ "is_diseased"

   # Bad
   x = bernoulli(0.01) @ "x"
   ```

2. **Use meaningful parameters**
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

3. **Document your model**
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

4. **Start simple, add complexity**
   - Build the simplest model first
   - Verify it works
   - Add features incrementally

5. **Test edge cases**
   - What if parameters are 0? 1?
   - What if all observations are the same?
   - Does the posterior make intuitive sense?

---

### ❌ DON'T

1. **Don't forget to name random choices**
   ```python
   # Bad — can't condition on this!
   x = bernoulli(0.5)

   # Good
   x = bernoulli(0.5) @ "x"
   ```

2. **Don't use the same name twice**
   ```python
   # Bad — name collision!
   flip1 = bernoulli(0.5) @ "flip"
   flip2 = bernoulli(0.5) @ "flip"  # ERROR!

   # Good — unique names
   flip1 = bernoulli(0.5) @ "flip_1"
   flip2 = bernoulli(0.5) @ "flip_2"
   ```

3. **Don't overthink distributions**
   - `bernoulli` covers most binary cases
   - `normal` for continuous
   - `categorical` for multiple choices
   - You don't need exotic distributions to start!

4. **Don't skip validation**
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
```python
@gen
def spam_filter(spam_rate=0.30):
    """Simple spam filter based on keyword."""

    # Hidden: is it spam?
    is_spam = bernoulli(spam_rate) @ "is_spam"

    # Observation: contains "FREE"?
    if is_spam:
        contains_free = bernoulli(0.80) @ "contains_free"
    else:
        contains_free = bernoulli(0.10) @ "contains_free"

    return is_spam

# Email contains "FREE"
observation = ChoiceMap({"contains_free": 1})

def infer_spam(k):
    trace, weight = spam_filter.generate(k, (), observation)
    return trace.get_retval()

keys = jax.random.split(key, 10000)
posterior = jax.vmap(infer_spam)(keys)
prob_spam = jnp.mean(posterior)

print(f"P(Spam | contains 'FREE') ≈ {prob_spam:.3f}")
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
```python
@gen
def coin_model(n_flips):
    """Infer coin bias from observed flips."""

    # Hidden: coin's true bias
    bias = uniform(0.0, 1.0) @ "bias"

    # Observations: flips
    for i in range(n_flips):
        flip = bernoulli(bias) @ f"flip_{i}"

    return bias

# Observed flips: 16 heads out of 20
observed_flips = [1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1]
observations = ChoiceMap({f"flip_{i}": observed_flips[i] for i in range(20)})

def infer_bias(k):
    trace, weight = coin_model.generate(k, (20,), observations)
    return trace.get_retval()

keys = jax.random.split(key, 1000)
posterior_bias = jax.vmap(infer_bias)(keys)
mean_bias = jnp.mean(posterior_bias)
std_bias = jnp.std(posterior_bias)

print(f"Estimated bias: {mean_bias:.2f} ± {std_bias:.2f}")
# Should be around 0.80 (16/20)
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
```python
@gen
def disease_three_symptoms(prevalence=0.02):
    """Disease model with three symptoms."""

    has_disease = bernoulli(prevalence) @ "has_disease"

    if has_disease:
        fever = bernoulli(0.90) @ "fever"
        cough = bernoulli(0.80) @ "cough"
        fatigue = bernoulli(0.95) @ "fatigue"
    else:
        fever = bernoulli(0.10) @ "fever"
        cough = bernoulli(0.20) @ "cough"
        fatigue = bernoulli(0.30) @ "fatigue"

    return has_disease

# Scenario 1: Fever only
obs1 = ChoiceMap({"fever": 1})

# Scenario 2: Fever + cough
obs2 = ChoiceMap({"fever": 1, "cough": 1})

# Scenario 3: All three
obs3 = ChoiceMap({"fever": 1, "cough": 1, "fatigue": 1})

for i, obs in enumerate([obs1, obs2, obs3], 1):
    def infer(k):
        trace, weight = disease_three_symptoms.generate(k, (), obs)
        return trace.get_retval()

    keys = jax.random.split(key, 10000)
    posterior = jax.vmap(infer)(keys)
    prob = jnp.mean(posterior)

    print(f"Scenario {i}: P(Disease) ≈ {prob:.3f}")
```

**Expected output:**
```
Scenario 1: P(Disease) ≈ 0.155  # Fever only
Scenario 2: P(Disease) ≈ 0.419  # Fever + cough
Scenario 3: P(Disease) ≈ 0.774  # All three symptoms
```

**Insight:** More evidence → higher posterior!
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

GenJAX supports many distributions beyond `bernoulli`:

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

## Resources

**GenJAX Documentation:**
- Official docs: [gen.dev](https://gen.dev)
- More examples: [GenJAX GitHub](https://github.com/probcomp/genjax)

**Probability Theory:**
- This tutorial's probability chapters!
- "Probabilistic Programming & Bayesian Methods for Hackers" (free online)
- "Think Bayes" by Allen Downey (beginner-friendly)

**Community:**
- GenJAX Discourse: Ask questions, share models
- Probabilistic Programming Slack: Connect with practitioners

---

## You Did It!

**Congratulations on completing the GenJAX tutorial!**

You've gone from "what is GenJAX?" to building complete probabilistic models. That's no small feat!

**Now go build something amazing.** 🚀

---

|[← Previous: Inference in Action](./05_inference.md) | [Back to Introduction →](./index.md)|
| :--- | ---: |
