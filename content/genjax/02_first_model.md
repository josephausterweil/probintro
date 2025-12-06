+++
date = "2025-12-06"
title = "Your First GenJAX Model"
weight = 3
+++

## From Sets to Simulation

Remember Chibany's daily meals? We listed out the outcome space $\Omega = \\{HH, HT, TH, TT\\}$ and counted possibilities.

Now we'll teach a computer to **generate** those outcomes instead!

![Chibany laying down](images/chibanylayingdown.png)

---

## The Generative Process

Each day:
1. **Lunch arrives** — randomly H or T (equal probability)
2. **Dinner arrives** — randomly H or T (equal probability)
3. **Record the day** — the pair of meals

In GenJAX, we express this as a **generative function**.

---

## Your First Generative Function

Here's Chibany's meals in GenJAX:

```python
import jax
from genjax import gen, flip

@gen
def chibany_day():
    """Generate one day of Chibany's meals."""

    # Lunch: flip a coin (0=Hamburger, 1=Tonkatsu)
    lunch_is_tonkatsu = flip(0.5) @ "lunch"

    # Dinner: flip another coin
    dinner_is_tonkatsu = flip(0.5) @ "dinner"

    # Return the pair
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)
```

{{% notice style="warning" title="Important: Use flip(), not bernoulli()" %}}
GenJAX has two functions for Bernoulli distributions: `flip(p)` and `bernoulli(p)`. **Always use `flip(p)`** - it works correctly as your intuition expects!

The `bernoulli(p)` function expects the parameter to be the logit value of the coinflip, which is less intuitive for our purposes. The `flip(p)` function works as expected and is what the official GenJAX examples use.

**Example of the bug:**
- `bernoulli(0.9)` produces ~71% instead of 90% ❌
- `flip(0.9)` produces ~90% as expected ✅

This tutorial uses `flip()` throughout to ensure correct behavior.
{{% /notice %}}

{{% notice style="success" title="📐→💻 Math-to-Code Translation" %}}
**How mathematical concepts translate to GenJAX:**

| Math Concept | Mathematical Notation | GenJAX Code |
|--------------|----------------------|-------------|
| **Outcome Space** | $\Omega = \\{HH, HT, TH, TT\\}$ | `@gen def chibany_day(): ...` |
| **Random Variable** | $X \sim \text{Bernoulli}(0.5)$ | `flip(0.5) @ "lunch"` |
| **Probability** | $P(A) = \frac{\|A\|}{\|\Omega\|}$ | `jnp.mean(condition_satisfied)` |
| **Event** | $A = \\{HT, TH, TT\\}$ | `has_tonkatsu = (days[:, 0] == 1) \| (days[:, 1] == 1)` |

**Key insights:**
- **@gen function** = Generative process defining Ω
- **flip(p)** = Random variable with probability p (Bernoulli distribution)
- **@ "name"** = Label the random choice (for inference later)
- **Simulation + counting** = Computing probabilities
{{% /notice %}}

### Breaking It Down

**Line 1: `@gen`**
- Tells GenJAX: "This is a generative function"
- GenJAX will track all random choices

**Line 2-3: Function definition**
- `def chibany_day():` defines the function
- The docstring explains what it does

**Line 6: First random choice**
```python
lunch_is_tonkatsu = flip(0.5) @ "lunch"
```
- `flip(0.5)` — Flip a fair coin (50% chance of 1, 50% chance of 0)
- `@ "lunch"` — **Name** this random choice "lunch"
- Store the result in `lunch_is_tonkatsu`

**Line 9: Second random choice**
```python
dinner_is_tonkatsu = flip(0.5) @ "dinner"
```
- Another coin flip, named "dinner"

**Line 12: Return value**
```python
return (lunch_is_tonkatsu, dinner_is_tonkatsu)
```
- Returns a **tuple** (pair) of the two values
- This is like one outcome from $\Omega$!

---

## Running the Function

### Generating One Day

```python
# Create a random key (JAX requirement for randomness)
key = jax.random.key(42)

# Generate one day
trace = chibany_day.simulate(key, ())

# What happened?
meals = trace.get_retval()
print(f"Today's meals: {meals}")
```

**Output (example):**
```
Today's meals: (0, 1)
```

{{% notice style="warning" title="What You'll Actually See" %}}
When you run this code, you'll see output like:
```
Today's meals: (Array(0, dtype=int32), Array(1, dtype=int32))
```

**Don't panic!** This is because GenJAX returns JAX arrays, not plain Python numbers.

<details>
<summary><em>Why the difference? (Click to expand)</em></summary>

**What you see**: `Array(0, dtype=int32)` or `Array(1, dtype=int32)`

**What it means**:
- `Array(0, dtype=int32)` = 0 = Hamburger
- `Array(1, dtype=int32)` = 1 = Tonkatsu

**Why JAX does this**: JAX uses arrays for everything to enable fast computation on GPUs. These are JAX's way of representing numbers that can run efficiently on both CPUs and GPUs.

**To get simple numbers**, you can convert:
```python
meals_simple = (int(meals[0]), int(meals[1]))
print(f"Today's meals: {meals_simple}")
# Output: (0, 1)
```

**For this tutorial**: Just remember that `Array(0, dtype=int32)` is just a fancy way of saying 0, and `Array(1, dtype=int32)` means 1.

</details>
{{% /notice %}}

This means: Hamburger for lunch (0), Tonkatsu for dinner (1) — or in our notation: $HT$!

{{% notice style="info" title="What's a 'key'?" %}}
JAX uses **random keys** to control randomness. Think of it like a seed — the same key always gives the same "random" results, which helps with reproducibility.

**Don't worry about the details!** Just know:
- Create a key with `jax.random.key(some_number)`
- Split it for multiple uses with `jax.random.split(key, n)`
{{% /notice %}}

### Accessing the Random Choices

```python
# Get all the random choices made
choices = trace.get_choices()

print(f"Lunch was tonkatsu: {choices['lunch']}")
print(f"Dinner was tonkatsu: {choices['dinner']}")
```

**Output (for the trace above):**
```
Lunch was tonkatsu: 0
Dinner was tonkatsu: 1
```

{{% notice style="tip" title="Expected Output" %}}
You'll actually see:
```
Lunch was tonkatsu: 0
Dinner was tonkatsu: 1
```

**Good news**: When accessing individual choices with `choices['lunch']`, GenJAX gives you plain numbers (0 or 1), not the wrapped `Array(...)` format! This makes them easier to work with.
{{% /notice %}}

---

## Simulating Many Days

Now let's generate 10,000 days!

```python
# Generate 10,000 random keys
keys = jax.random.split(key, 10000)

# Run the generative function for each key
def run_one_day(k):
    trace = chibany_day.simulate(k, ())
    return trace.get_retval()

# Use JAX's vmap for parallel execution
days_tuples = jax.vmap(run_one_day)(keys)

# Convert from tuples to a 2D array for easier analysis
# Each row is one day, columns are [lunch, dinner]
days = jnp.stack(days_tuples, axis=1)
```

{{% notice style="tip" title="What's vmap?" %}}
`vmap` stands for "vectorized map" — it runs a function many times in parallel, which is **very fast**!

Think of it like: "Do this 10,000 times, but do them all at once instead of one-by-one"
{{% /notice %}}

### Counting Outcomes

Now we have 10,000 days. Let's count how many have at least one tonkatsu:

```python
import jax.numpy as jnp

# Check if either meal is tonkatsu (1)
has_tonkatsu = jnp.logical_or(days[:, 0], days[:, 1])

# Count how many days have tonkatsu
count_with_tonkatsu = jnp.sum(has_tonkatsu)

# Calculate probability
prob_tonkatsu = jnp.mean(has_tonkatsu)

print(f"Days with tonkatsu: {count_with_tonkatsu} out of 10000")
print(f"P(at least one tonkatsu) ≈ {prob_tonkatsu:.3f}")
```

**Output (example):**
```
Days with tonkatsu: 7489 out of 10000
P(at least one tonkatsu) ≈ 0.749
```

**Remember from the probability tutorial:** The exact answer is $3/4 = 0.75$!

With 10,000 simulations, we got very close: $0.749 \approx 0.75$

{{% notice style="info" title="📘 Foundation Concept: Simulation vs. Counting" %}}
**Recall from Tutorial 1, Chapter 3** that probability is counting:

$$P(A) = \frac{|A|}{|\Omega|} = \frac{\text{outcomes in event}}{\text{total outcomes}}$$

We calculated $P(\text{at least one tonkatsu}) = \frac{|\{HT, TH, TT\}|}{|\{HH, HT, TH, TT\}|} = \frac{3}{4} = 0.75$ by hand.

**Now with GenJAX, we simulate instead of enumerate:**

| Tutorial 1 (By Hand) | Tutorial 2 (GenJAX) |
|---------------------|---------------------|
| **List** all outcomes: {HH, HT, TH, TT} | **Generate** 10,000 samples |
| **Count** favorable: 3 out of 4 | **Count** favorable: ~7,500 out of 10,000 |
| **Divide**: 3/4 = 0.75 | **Divide**: 7,500/10,000 ≈ 0.75 |

**Why simulate?**
- Tutorial 1 approach breaks down with complex models (too many outcomes to list)
- Simulation scales: same code works whether Ω has 4 outcomes or 4 billion
- As simulations increase (10K → 100K → 1M), we get closer to exact answer

**The principle is identical** — count favorable outcomes and divide by total. But simulation lets us handle models that are impossible to enumerate by hand!

[← Review probability as counting in Tutorial 1, Chapter 3](../../intro/03_prob_count/)
{{% /notice %}}

---

## Visualizing the Results

Let's make a bar chart showing all four outcomes:

```python
import matplotlib.pyplot as plt

# Count each outcome
HH = jnp.sum((days[:, 0] == 0) & (days[:, 1] == 0))
HT = jnp.sum((days[:, 0] == 0) & (days[:, 1] == 1))
TH = jnp.sum((days[:, 0] == 1) & (days[:, 1] == 0))
TT = jnp.sum((days[:, 0] == 1) & (days[:, 1] == 1))

# Create bar chart
outcomes = ['HH', 'HT', 'TH', 'TT']
counts = [HH, HT, TH, TT]

plt.figure(figsize=(8, 5))
plt.bar(outcomes, counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7dc6f'])
plt.xlabel('Outcome')
plt.ylabel('Count (out of 10,000)')
plt.title("Chibany's Meals: 10,000 Simulated Days")
plt.axhline(y=2500, color='gray', linestyle='--', label='Expected (2500 each)')
plt.legend()
plt.show()
```

**What you'll see:** Four bars of roughly equal height (around 2500 each), matching our theoretical expectation of $1/4$ for each outcome!

![Outcome distribution from 10,000 simulations](../../images/genjax/first_model_outcome_distribution.png)

---

## Interactive Exploration

{{% notice style="tip" title="📓 Interactive Notebook" %}}
Try the **[interactive notebook - Open in Colab](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/02_first_genjax_model.ipynb)** with live sliders and visualizations! It includes all the code from this chapter plus interactive widgets to explore how changing parameters affects the results.
{{% /notice %}}

The companion notebook has **interactive widgets** so you can:

### Slider 1: Probability of Tonkatsu for Lunch
- Move the slider from 0.0 to 1.0
- See how the distribution changes!

### Slider 2: Probability of Tonkatsu for Dinner
- Make dinner independent from lunch
- Or make tonkatsu more/less likely at different meals

### Slider 3: Number of Simulations
- Try 100, 1,000, 10,000, or even 100,000 simulations
- See how the estimate gets more accurate with more simulations

**The chart updates automatically** as you move the sliders!

{{% notice style="success" title="Try This!" %}}
In the Colab notebook:
1. Set lunch probability to 0.8 (80% tonkatsu)
2. Set dinner probability to 0.2 (20% tonkatsu)
3. Run 10,000 simulations
4. What do you notice about the distribution?

**Answer:** Outcomes with tonkatsu for lunch (TH, TT) are much more common than those without (HH, HT)!
{{% /notice %}}

---

## Connection to Set-Based Probability

Let's connect this back to what you learned:

| Set-Based Concept | GenJAX Equivalent |
|-------------------|-------------------|
| Outcome space $\Omega$ | Running `simulate()` many times |
| One outcome $\omega$ | One call to `simulate()` |
| Event $A \subseteq \Omega$ | Filtering simulations |
| $\|A\|$ (count elements) | `jnp.sum(condition)` |
| $P(A) = \|A\|/\|\Omega\|$ | `jnp.mean(condition)` |

**Example:**

**Set-based:**
- Event: "At least one tonkatsu" = $\\{HT, TH, TT\\}$
- Probability: $|\\{HT, TH, TT\\}| / |\\{HH, HT, TH, TT\\}| = 3/4$

**GenJAX:**
```python
has_tonkatsu = (days[:, 0] == 1) | (days[:, 1] == 1)
prob = jnp.mean(has_tonkatsu)  # ≈ 0.75
```

**It's the same concept!** Just computed instead of counted by hand.

---

## Understanding Traces

When you run `chibany_day.simulate(key, ())`, GenJAX creates a **trace** that records:

1. **Arguments** — What inputs were provided (none in this case)
2. **Random choices** — All the random decisions made, with their names
3. **Return value** — The final result

```python
trace = chibany_day.simulate(key, ())

# Access different parts
print(f"Return value: {trace.get_retval()}")
print(f"Choices: {trace.get_choices()}")
print(f"Log probability: {trace.get_score()}")
```

**Output (example):**
```
Return value: (Array(False, dtype=bool), Array(False, dtype=bool))
Choices: {'lunch': Array(False, dtype=bool), 'dinner': Array(False, dtype=bool)}
Log probability: -1.3862943611198906
```

**What this means:**
- **Return value**: The pair of meals (both False = both Hamburger = HH)
- **Choices**: A dictionary of all named random choices and their values
- **Log probability**: The log-likelihood of this particular outcome ($\log(0.5 \times 0.5) = \log(0.25) \approx -1.386$)

{{% notice style="info" title="Why Track Everything?" %}}
Tracking all random choices is essential for **inference** — when we want to ask "given I observed this, what's probable?"

We'll see this in action in Chapter 4!
{{% /notice %}}

---

## Exercises

Try these in the Colab notebook:

### Exercise 1: Different Probabilities

Modify the code so:
- Lunch is 70% likely to be tonkatsu
- Dinner is 30% likely to be tonkatsu

**Hint:** Change the `flip(0.5)` values!

{{% expand "Solution" %}}
```python
@gen
def chibany_day_weighted():
    lunch_is_tonkatsu = flip(0.7) @ "lunch"
    dinner_is_tonkatsu = flip(0.3) @ "dinner"
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)
```
{{% /expand %}}

### Exercise 2: Counting Tonkatsu

Write code to count **how many tonkatsu** Chibany gets across all simulated days (not just which days have tonkatsu, but the total count).

**Hint:** Add up `days[:, 0] + days[:, 1]`

{{% expand "Solution" %}}
```python
total_tonkatsu = jnp.sum(days[:, 0]) + jnp.sum(days[:, 1])
avg_per_day = total_tonkatsu / len(days)

print(f"Total tonkatsu: {total_tonkatsu}")
print(f"Average per day: {avg_per_day:.2f}")
```

**With equal probabilities (0.5 each), you should get close to 1.0 tonkatsu per day on average!**
{{% /expand %}}

### Exercise 3: Three Meals?

Extend the model to include breakfast! Now Chibany gets three meals per day.

{{% expand "Solution" %}}
```python
@gen
def chibany_three_meals():
    breakfast_is_tonkatsu = flip(0.5) @ "breakfast"
    lunch_is_tonkatsu = flip(0.5) @ "lunch"
    dinner_is_tonkatsu = flip(0.5) @ "dinner"
    return (breakfast_is_tonkatsu, lunch_is_tonkatsu, dinner_is_tonkatsu)
```

**Now the outcome space has $2^3 = 8$ possible outcomes!**
{{% /expand %}}

---

## What You've Learned

In this chapter, you:

✅ Wrote your first generative function

✅ Simulated thousands of random outcomes

✅ Calculated probabilities through counting

✅ Visualized distributions

✅ Understood the connection between sets and simulation

✅ Learned about traces and random choices

**The key insight:** Generative functions let computers do what you did by hand with sets — but now you can handle millions of possibilities!

---

## Next Steps

Now that you can generate outcomes, the next question is:

**What if I observe something? How do I update my beliefs?**

That's **inference**, and it's where GenJAX really shines!

---

|[← Previous: Python Essentials](./01_python_basics.md) | [Next: Understanding Traces →](./03_traces.md)|
| :--- | ---: |
