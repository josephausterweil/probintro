+++
title = "Understanding Traces"
weight = 4
+++

## What Gets Recorded When Code Runs?

When you run a regular Python function, it does its work and returns a value. Then it's done — no record of what happened internally.

**GenJAX is different.** When you run a generative function, GenJAX creates a **trace** — a complete record of:
1. What random choices were made
2. What values they took
3. What the function returned
4. How probable this execution was

Think of it like a lab notebook that automatically records every detail of an experiment!

![Chibany investigating](images/chibanylayingdown.png)

---

## Why Traces Matter

**Short answer:** Traces enable **inference** — answering "what if I observed this?"

**Example scenario:**
- You run `chibany_day()` and it returns `(0, 1)` — Hamburger for lunch, Tonkatsu for dinner
- The trace records: "I chose 0 for lunch, 1 for dinner"
- Later, you can ask: "Given that dinner was Tonkatsu, what's the probability lunch was also Tonkatsu?"

**Traces let us reason backwards from observations to causes!**

We'll explore this fully in Chapter 4. For now, let's understand what traces contain.

---

## Anatomy of a Trace

Recall our generative function:

```python
@gen
def chibany_day():
    lunch_is_tonkatsu = flip(0.5) @ "lunch"
    dinner_is_tonkatsu = flip(0.5) @ "dinner"
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)
```

When we run it:

```python
key = jax.random.key(42)
trace = chibany_day.simulate(key, ())
```

GenJAX creates a trace object containing **three key components:**

### 1. The Return Value

**What the function returned:**

```python
meals = trace.get_retval()
print(meals)  # Output: (0, 1)
```

This is the final result — the observable outcome.

### 2. The Random Choices

**All the random decisions made, with their names:**

```python
choices = trace.get_choices()
print(choices)
# Output: {'lunch': 0, 'dinner': 1}
```

This is the **choice map** — a dictionary mapping addresses (names) to values.

{{% notice style="info" title="Why Names Matter" %}}
In `flip(0.5) @ "lunch"`, the `@ "lunch"` part gives this random choice a **name** (or address).

GenJAX uses these names to:
- Track which choice is which
- Let you specify observations (more in Chapter 4!)
- Enable inference algorithms

**Think of it like labeling test tubes in a chemistry lab.** You need to know which is which!
{{% /notice %}}

### 3. The Log Probability (Score)

**How probable was this execution?**

```python
score = trace.get_score()
print(score)  # Output: -1.3862943611198906
```

This is the **log probability** of this particular execution.

{{% notice style="note" title="Math Notation: Log Probability" %}}
For our example:
- Lunch = 0 has probability 0.5
- Dinner = 1 has probability 0.5
- Joint probability: $P(\text{lunch}=0, \text{dinner}=1) = 0.5 \times 0.5 = 0.25$

Log probability: $\log(0.25) = -1.386...$

**Why use logs?**
- Prevents numerical underflow (very small probabilities)
- Turns multiplication into addition (easier math!)
- Standard in probabilistic programming

**You don't need to work with log probabilities directly** — GenJAX handles this for you. Just know they measure "how likely was this outcome."
{{% /notice %}}

{{% notice style="success" title="📐→💻 Math-to-Code Translation" %}}
**How traces connect to probability theory:**

| Math Concept | Mathematical Notation | GenJAX Trace Component |
|--------------|----------------------|------------------------|
| **Outcome** | $\omega \in \Omega$ | One trace (one execution) |
| **Outcome Space** | $\Omega = \\{HH, HT, TH, TT\\}$ | All possible traces |
| **Random Variable** | $X(\omega)$ | A choice in the choice map |
| **Probability** | $P(\omega)$ | `jnp.exp(trace.get_score())` |
| **Log Probability** | $\log P(\omega)$ | `trace.get_score()` |
| **Joint Distribution** | $P(X_1, X_2)$ | Distribution over traces |

**Key insights:**
- **A trace IS an outcome** — It represents one complete way the random process unfolds
- **Choice map = Random variables** — Named random choices like `"lunch"` and `"dinner"`
- **get_retval() = Observable outcome** — What you can directly observe
- **get_score() = Log probability** — How likely this particular trace is
- **Multiple traces = Multiple outcomes** — Running `simulate()` repeatedly samples from Ω

**Example mapping:**
```
Math: ω = HT  (outcome from Ω)
Code: trace with choices = {'lunch': 0, 'dinner': 1}
They're the same thing, just different representations!
```
{{% /notice %}}

---

## The Complete Trace Diagram

Let's visualize what's in a trace:

```
┌─────────────────────────────────────────┐
│           TRACE OBJECT                  │
├─────────────────────────────────────────┤
│                                         │
│  1. Arguments: ()                       │
│     (what was passed to the function)   │
│                                         │
│  2. Random Choices (Choice Map):        │
│     {'lunch': 0, 'dinner': 1}           │
│     (all random decisions made)         │
│                                         │
│  3. Return Value:                       │
│     (0, 1)                              │
│     (what the function returned)        │
│                                         │
│  4. Log Probability (Score):            │
│     -1.386                              │
│     (how probable was this trace)       │
│                                         │
└─────────────────────────────────────────┘
```

**Every time you call `simulate()`, you get a new trace with (potentially) different random choices.**

---

## Accessing Trace Components

Here's a complete example showing all three ways to access trace information:

```python
import jax
from genjax import gen, bernoulli

@gen
def chibany_day():
    lunch_is_tonkatsu = bernoulli(0.5) @ "lunch"
    dinner_is_tonkatsu = bernoulli(0.5) @ "dinner"
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)

# Generate one trace
key = jax.random.key(42)
trace = chibany_day.simulate(key, ())

# Access different parts
print("=== TRACE CONTENTS ===")
print(f"Return value: {trace.get_retval()}")
print(f"Random choices: {trace.get_choices()}")
print(f"Log probability: {trace.get_score()}")

# Decode to outcome notation
outcome_map = {(0, 0): "HH", (0, 1): "HT", (1, 0): "TH", (1, 1): "TT"}
outcome = outcome_map[tuple(trace.get_retval())]
print(f"Outcome: {outcome}")
```

**Output (example):**
```
=== TRACE CONTENTS ===
Return value: (0, 1)
Random choices: {'lunch': 0, 'dinner': 1}
Log probability: -1.3862943611198906
Outcome: HT
```

---

## Multiple Traces, Multiple Histories

Each trace represents **one possible execution** of the generative function.

Run it 5 times, get 5 different traces:

```python
key = jax.random.key(42)

for i in range(5):
    # Split key for each run (JAX requirement)
    key, subkey = jax.random.split(key)

    trace = chibany_day.simulate(subkey, ())
    outcome = outcome_map[tuple(trace.get_retval())]
    choices = trace.get_choices()

    print(f"Day {i+1}: {outcome} — lunch={choices['lunch']}, dinner={choices['dinner']}")
```

**Output (example):**
```
Day 1: HT — lunch=0, dinner=1
Day 2: TH — lunch=1, dinner=0
Day 3: HH — lunch=0, dinner=0
Day 4: TT — lunch=1, dinner=1
Day 5: HT — lunch=0, dinner=1
```

Each trace is a **different history** — a different way the random process could have unfolded.

{{% notice style="tip" title="JAX Random Keys" %}}
Notice we use `jax.random.split(key)` to create new keys for each run?

**Why?** JAX uses explicit random keys for reproducibility. The same key always gives the same result.

**Pattern:**
```python
key, subkey = jax.random.split(key)  # Create new key
trace = model.simulate(subkey, ...)   # Use the subkey
```

This ensures different random outcomes each time while maintaining reproducibility.
{{% /notice %}}

---

## Traces vs Return Values

**Important distinction:**

| `simulate()` returns | `get_retval()` returns |
|---------------------|----------------------|
| **Trace object** | **The actual value** |
| Contains choices, score, return value | Just the return value |
| Used for inference | Used for the result |

**Example:**

```python
# This is a trace object
trace = chibany_day.simulate(key, ())

# This is the return value (a tuple)
meals = trace.get_retval()

# These are different!
print(type(trace))   # <class 'genjax.generative_functions.static.trace.StaticTrace'>
print(type(meals))   # <class 'tuple'>
```

**When to use which:**
- **Need just the outcome?** Use `trace.get_retval()`
- **Need to inspect random choices?** Use `trace.get_choices()`
- **Doing inference?** Use the full trace object

---

## Connection to Probability Theory

Let's connect traces back to set-based probability:

| Probability Concept | Trace Equivalent |
|---------------------|------------------|
| Outcome $\omega \in \Omega$ | One trace (one execution) |
| Outcome space $\Omega$ | All possible traces |
| $P(\omega)$ | `exp(trace.get_score())` |
| Random variable $X(\omega)$ | A choice in the choice map |
| Joint distribution | Distribution over traces |

**Key insight:** A trace IS an outcome! The trace represents one complete way the random process could unfold.

**Example:**
- **Set-based:** $\omega = HT$ (one outcome from $\Omega = \{HH, HT, TH, TT\}$)
- **Trace-based:** A trace where `choices = {'lunch': 0, 'dinner': 1}`

**They're the same thing!** Just different representations.

---

## Why This Matters for Inference

Consider this question:

> **"Given that Chibany got Tonkatsu for dinner, what's the probability they also got Tonkatsu for lunch?"**

**Set-based approach:**
1. Define event $D$ = "dinner is Tonkatsu" = $\{HT, TT\}$
2. Define event $L$ = "lunch is Tonkatsu" = $\{TH, TT\}$
3. Calculate $P(L \mid D) = \frac{|L \cap D|}{|D|} = \frac{1}{2}$

**Trace-based approach:**
1. Generate many traces
2. Filter traces where `choices['dinner'] == 1`
3. Among those, count how many have `choices['lunch'] == 1`
4. Calculate the ratio

**The trace structure makes this filtering possible!** Because GenJAX records all the random choices, we can look inside and check what happened.

We'll implement this in Chapter 4!

---

## Practical Example: Inspecting Traces

Let's generate 10 traces and inspect them:

```python
import jax
import jax.numpy as jnp
from genjax import gen, bernoulli

@gen
def chibany_day():
    lunch_is_tonkatsu = bernoulli(0.5) @ "lunch"
    dinner_is_tonkatsu = bernoulli(0.5) @ "dinner"
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)

# Generate 10 traces
key = jax.random.key(42)
outcome_map = {(0, 0): "HH", (0, 1): "HT", (1, 0): "TH", (1, 1): "TT"}

print("Day | Outcome | Lunch | Dinner | Log Prob")
print("----|---------|-------|--------|----------")

for i in range(10):
    key, subkey = jax.random.split(key)
    trace = chibany_day.simulate(subkey, ())

    outcome = outcome_map[tuple(trace.get_retval())]
    choices = trace.get_choices()
    score = trace.get_score()

    print(f" {i+1:2d} |   {outcome}    |   {choices['lunch']}   |   {choices['dinner']}    | {score:.2f}")
```

**Output (example):**
```
Day | Outcome | Lunch | Dinner | Log Prob
----|---------|-------|--------|----------
  1 |   HT    |   0   |   1    | -1.39
  2 |   TH    |   1   |   0    | -1.39
  3 |   HH    |   0   |   0    | -1.39
  4 |   TT    |   1   |   1    | -1.39
  5 |   HT    |   0   |   1    | -1.39
  6 |   HH    |   0   |   0    | -1.39
  7 |   TT    |   1   |   1    | -1.39
  8 |   HT    |   0   |   1    | -1.39
  9 |   TH    |   1   |   0    | -1.39
 10 |   HH    |   0   |   0    | -1.39
```

**Notice:** All log probabilities are the same (-1.39 ≈ log(0.25)) because all outcomes are equally probable!

---

## Exercises

### Exercise 1: Trace Exploration

Run this code and answer the questions:

```python
key = jax.random.key(123)
trace = chibany_day.simulate(key, ())

print(f"Return value: {trace.get_retval()}")
print(f"Choices: {trace.get_choices()}")
print(f"Score: {trace.get_score()}")
```

**Questions:**
1. What outcome did you get? (HH, HT, TH, or TT)
2. What's in the choice map?
3. Is the log probability the same as previous examples?

{{% expand "Solution" %}}
**Answers:**
1. The outcome depends on the random seed (123)
2. The choice map contains `{'lunch': 0 or 1, 'dinner': 0 or 1}`
3. Yes! All outcomes have equal probability (0.25), so log probability is always -1.386...

**Key insight:** Different random keys → different traces, but same probabilities (for this symmetric example)
{{% /expand %}}

---

### Exercise 2: Unequal Probabilities

Modify `chibany_day` to have unequal probabilities:

```python
@gen
def chibany_day_biased():
    lunch_is_tonkatsu = bernoulli(0.8) @ "lunch"  # 80% Tonkatsu
    dinner_is_tonkatsu = bernoulli(0.2) @ "dinner"  # 20% Tonkatsu
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)
```

Generate 5 traces and compare their log probabilities.

**Question:** Are all log probabilities the same? Why or why not?

{{% expand "Solution" %}}
```python
key = jax.random.key(42)

for i in range(5):
    key, subkey = jax.random.split(key)
    trace = chibany_day_biased.simulate(subkey, ())

    outcome = outcome_map[tuple(trace.get_retval())]
    score = trace.get_score()

    print(f"Day {i+1}: {outcome} — Log prob: {score:.3f}")
```

**Answer:** No! Log probabilities differ because outcomes have different probabilities:
- TT: $P = 0.8 \times 0.2 = 0.16$, $\log(0.16) = -1.83$
- TH: $P = 0.8 \times 0.8 = 0.64$, $\log(0.64) = -0.45$
- HT: $P = 0.2 \times 0.2 = 0.04$, $\log(0.04) = -3.22$
- HH: $P = 0.2 \times 0.8 = 0.16$, $\log(0.16) = -1.83$

**TH is most likely** (highest probability = least negative log probability)!
{{% /expand %}}

---

### Exercise 3: Conditional Counting

Generate 1000 traces from `chibany_day()` and answer:

**"Among days when dinner is Tonkatsu, what fraction also have Tonkatsu for lunch?"**

**Hint:** Filter traces where `choices['dinner'] == 1`, then count how many have `choices['lunch'] == 1`.

{{% expand "Solution" %}}
```python
import jax
import jax.numpy as jnp

key = jax.random.key(42)
keys = jax.random.split(key, 1000)

# Generate all traces
def run_one_day(k):
    trace = chibany_day.simulate(k, ())
    return trace.get_retval()

days = jax.vmap(run_one_day)(keys)

# Filter: dinner is Tonkatsu (dinner == 1)
dinner_is_tonkatsu = days[:, 1] == 1

# Among those, count lunch is Tonkatsu
both_tonkatsu = (days[:, 0] == 1) & (days[:, 1] == 1)

# Calculate conditional probability
n_dinner_tonkatsu = jnp.sum(dinner_is_tonkatsu)
n_both = jnp.sum(both_tonkatsu)

prob_lunch_given_dinner = n_both / n_dinner_tonkatsu

print(f"Days with dinner = Tonkatsu: {n_dinner_tonkatsu}")
print(f"Days with both = Tonkatsu: {n_both}")
print(f"P(lunch=T | dinner=T) ≈ {prob_lunch_given_dinner:.3f}")
```

**Expected result:** ≈ 0.5 (50%)

**Why?** Lunch and dinner are independent! Knowing dinner doesn't change lunch probability.

**This is conditional probability through filtering!** (More in Chapter 4)
{{% /expand %}}

---

## What You've Learned

In this chapter, you learned:

✅ **What traces are** — complete records of random execution

✅ **Three key components** — return value, choice map, log probability

✅ **Why names matter** — `@ "address"` enables tracking and inference

✅ **How to access trace parts** — `get_retval()`, `get_choices()`, `get_score()`

✅ **Traces as outcomes** — connection to probability theory

✅ **Preview of inference** — filtering traces to answer conditional questions

**The key insight:** Traces aren't just records — they're the bridge between generative code and probabilistic reasoning!

---

## Next Steps

Now that you understand traces, you're ready for the most powerful feature of GenJAX:

**Chapter 4: Conditioning and Observations** — How to ask "what if I observed this?" and update beliefs based on evidence!

This is where GenJAX really shines compared to regular simulation.

---

|[← Previous: Your First GenJAX Model](./02_first_model.md) | [Next: Conditioning and Observations →](./04_conditioning.md)|
| :--- | ---: |
