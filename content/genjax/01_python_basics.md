+++
title = "Python Essentials for GenJAX"
weight = 2
+++

## You Don't Need to Become a Programmer!

This chapter teaches you **just enough Python** to read and run GenJAX code. You won't become a software developer, but you'll be able to:
- Understand what the code is doing
- Modify values to experiment
- Run examples and see results

Think of it like learning enough Italian to order food in a restaurant — you don't need fluency, just practical knowledge!

---

## 1. Variables: Giving Names to Things

In Python, we give names to values so we can use them later.

```python
probability_hamburger = 0.5
probability_tonkatsu = 0.5
```

**What this means:** "Remember these numbers and call them by these names"

**Connection to probability:** Just like we wrote $P(H) = 0.5$ in math, we're storing that value.

**Try it:**
```python
x = 10
y = 20
result = x + y
print(result)  # Shows: 30
```

{{% notice style="info" title="The # Symbol" %}}
Anything after `#` is a **comment** — a note for humans, ignored by the computer.

```python
# This is a comment
x = 5  # Comments can go after code too
```
{{% /notice %}}

---

## 2. Functions: Recipes for Actions

A **function** is a named set of instructions. Think of it like a recipe.

```python
def greet_chibany():
    print("Hello, Chibany!")
    print("Time for tonkatsu!")

greet_chibany()  # "Calls" the function (runs the recipe)
```

**Output:**
```
Hello, Chibany!
Time for tonkatsu!
```

### Functions with Inputs (Parameters)

Functions can take **inputs** (called parameters):

```python
def greet_cat(name):
    print(f"Hello, {name}!")

greet_cat("Chibany")  # Output: Hello, Chibany!
greet_cat("Felix")    # Output: Hello, Felix!
```

**The f before the string** lets you put variables inside {} inside the text.

### Functions with Outputs (Return Values)

Functions can **return** values:

```python
def add_numbers(a, b):
    result = a + b
    return result

total = add_numbers(5, 3)  # total is now 8
```

**Connection to probability:** Remember how $f(\omega)$ is a function that takes an outcome and returns a number? Same idea!

---

## 3. Lists: Collections of Things

A **list** is like a shopping list — an ordered collection of items.

```python
meals = ["HH", "HT", "TH", "TT"]
```

**Connection to sets:** This is like $\Omega = \\{HH, HT, TH, TT\\}$ but with ordering!

### Accessing Items

```python
meals = ["HH", "HT", "TH", "TT"]

first_meal = meals[0]   # "HH" (Python counts from 0!)
second_meal = meals[1]  # "HT"
```

{{% notice style="warning" title="Python Counts from Zero!" %}}
The first item is `[0]`, the second is `[1]`, etc.

This trips up everyone at first. Just remember: Python is a bit quirky!
{{% /notice %}}

### How Many Items?

```python
meals = ["HH", "HT", "TH", "TT"]
count = len(meals)  # 4
```

**Connection:** This is like $|\Omega|$ (cardinality)!

---

## 4. Loops: Doing Things Repeatedly

A **for loop** repeats actions:

```python
for meal in ["HH", "HT", "TH", "TT"]:
    print(meal)
```

**Output:**
```
HH
HT
TH
TT
```

**How to read it:** "For each meal in this list, print the meal"

### Counting Loops

```python
for i in range(5):
    print(f"Day {i}")
```

**Output:**
```
Day 0
Day 1
Day 2
Day 3
Day 4
```

**Connection:** If we wanted to simulate 10,000 days, we'd use `range(10000)`!

---

## 5. Conditionals: Making Decisions

An **if statement** lets code make choices:

```python
meal = "TT"

if "T" in meal:
    print("Contains tonkatsu!")
else:
    print("No tonkatsu today :(")
```

**How to read it:** "If T is in the meal, do this. Otherwise, do that."

### Multiple Conditions

```python
tonkatsu_count = 2

if tonkatsu_count == 2:
    print("Two tonkatsus!")
elif tonkatsu_count == 1:
    print("One tonkatsu!")
else:
    print("No tonkatsu!")
```

**Note:**
- `==` means "equals" (comparison)
- `=` means "assign" (giving a value)

---

## 6. Decorators: Adding Special Powers

A **decorator** adds capabilities to a function. In GenJAX, we use `@gen`:

```python
@gen
def my_function():
    # ... code ...
```

**What `@gen` does:** Tells GenJAX "this is a generative function — please track all the random choices!"

**You don't need to fully understand decorators.** Just know:
- They go right before function definitions
- They modify how the function behaves
- In GenJAX, `@gen` is essential for probabilistic models

---

## 7. The @ Symbol in GenJAX (Addressing)

In GenJAX, we use `@` to **name random choices**:

```python
lunch = flip(0.5) @ "lunch"
```

**How to read it:** "Flip a coin with 50% chance of heads (true/1/tonkatsu), and call this choice 'lunch'"

**What is a Bernoulli random variable?** A Bernoulli random variable represents a single yes/no outcome, like a coin flip. It can be either 0 (failure/false/tails/tonkatsu) or 1 (success/true/heads/hamburger), with probability $p$ of being 1. In GenJAX, we use `flip(p)` to sample from a Bernoulli distribution—it's named after the coin flip metaphor!

**Connection to probability:** This is like saying "let $L$ be the random variable for lunch" where $L$ follows a Bernoulli distribution with $p=0.5$

---

## 8. Libraries and Imports

Libraries are collections of pre-written code we can use:

```python
import jax
import matplotlib.pyplot as plt
from genjax import gen, flip
```

**What this means:**
- `import jax` — Load the JAX library (for computation)
- `import matplotlib.pyplot as plt` — Load plotting tools, call them `plt`
- `from genjax import gen, flip` — From GenJAX, load these specific tools

**You don't need to memorize these.** Just run the import cells at the start of each notebook!

---

## 9. Calling Methods with Dot Notation

Sometimes we call functions "on" an object:

```python
trace = model.simulate(key, args)
choices = trace.get_choices()
```

**How to read it:** "Call the simulate function that belongs to model"

The `.` means "belonging to" or "part of".

---

## 10. Comments and Documentation

### Single-line Comments

```python
# This is a comment
x = 5  # This is also a comment
```

### Multi-line Comments (Docstrings)

```python
def my_function():
    """
    This is a docstring.
    It explains what the function does.
    """
    # ... code ...
```

**Why they matter:** They help you understand what code does!

---

## Quick Reference: Reading GenJAX Code

Here's a typical GenJAX function broken down:

```python
@gen                                    # Decorator: makes this a generative function
def chibany_meals():                    # Function name
    """Generate one day of meals."""   # Docstring: what it does

    # Random choice: lunch
    lunch = flip(0.5) @ "lunch"         # @ names the choice

    # Random choice: dinner
    dinner = flip(0.5) @ "dinner"       # Another named choice

    # Return both meals as a pair
    return (lunch, dinner)              # Return value
```

**To read GenJAX code:**
1. Find the `@gen` — it's a generative function
2. Read the docstring — what does it do?
3. Look for `@` symbols — those are the random choices
4. See what it returns — that's the outcome

---

## Practice: Can You Read This?

```python
@gen
def coin_flips(n):
    """Flip a fair coin n times."""
    heads_count = 0

    for i in range(n):
        coin = flip(0.5) @ f"flip_{i}"
        if coin == 1:
            heads_count = heads_count + 1

    return heads_count
```

{{% expand "What does this do?" %}}
**Line by line:**
1. `@gen` — This is a generative function
2. `def coin_flips(n):` — Takes a number `n` as input
3. `heads_count = 0` — Start counting at 0
4. `for i in range(n):` — Repeat n times
5. `coin = flip(0.5) @ f"flip_{i}"` — Flip a fair coin (Bernoulli with p=0.5), name it "flip_0", "flip_1", etc.
6. `if coin == 1:` — If it's heads (1)
7. `heads_count = heads_count + 1` — Add one to the count
8. `return heads_count` — Return how many heads we got

**What it does:** Flips a coin n times and counts the heads!

**Connection:** This is like the binomial distribution from probability theory.
{{% /expand %}}

---

## What You Don't Need to Learn

**Don't worry about:**
- ❌ Object-oriented programming
- ❌ Advanced data structures
- ❌ File I/O
- ❌ Error handling
- ❌ Most of Python's features!

**Focus on:**
- ✅ Reading code to understand what it does
- ✅ Running code cells in notebooks
- ✅ Changing parameter values to experiment
- ✅ Understanding the connection to probability

---

## Tips for Success

### 1. You Don't Need to Memorize

Keep this chapter open as a reference. When you see something in GenJAX code you don't recognize, come back here!

### 2. Run Code to Understand It

Don't just read — **run** the code! Seeing output makes everything clearer.

### 3. Experiment!

Try changing values:
- What happens if you change `0.5` to `0.8`?
- What if you change the number of simulations?
- Break things and see what errors you get!

### 4. Ask "What is This Doing?"

Not "How does this work?" but "What is this trying to accomplish?"

---

## Ready for GenJAX!

You now know enough Python to:
- ✅ Read GenJAX code
- ✅ Understand what generative functions do
- ✅ Run examples in Colab
- ✅ Modify values to experiment

**Next up:** Let's write your first generative function!

---

|[← Previous: Getting Started](./00_getting_started.md) | [Next: Your First GenJAX Model →](./02_first_model.md)|
| :--- | ---: |
