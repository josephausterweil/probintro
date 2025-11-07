+++
title = "Glossary"
weight = 6
+++

This glossary provides definitions for key terms used throughout the tutorial. Click on any term to expand its definition.

---

## Core Concepts

### set
{{% expand "set" %}}  A *set* is a collection of elements or members. Sets are defined by the elements they do or do not contain. The elements are listed with commas between them and "$\\{$" denotes the start of a set and "$\\}$" the end of a set. Note that the elements of a set are unique.

**Example:** $\\{H, T\\}$ is a set containing two elements: H and T.
{{% /expand %}}

### outcome space
{{% expand "outcome space" %}} The *outcome space* (denoted $\Omega$, the Greek letter omega) is the set of all possible outcomes for a random process. It forms the foundation for calculating probabilities.

**Example:** For Chibany's two daily meals, $\Omega = \\{HH, HT, TH, TT\\}$.
{{% /expand %}}

### event
{{% expand "event" %}} An *event* is a set that contains none, some, or all of the possible outcomes. In other words, an event is any subset of the outcome space $\Omega$.

**Example:** "At least one tonkatsu" is the event $\\{HT, TH, TT\\} \subseteq \Omega$.
{{% /expand %}}

### cardinality
{{% expand "cardinality" %}} The *cardinality* or *size* of a set is the number of elements it contains. If $A = \\{H, T\\}$, then the cardinality of $A$ is $|A|=2$.

**Notation:** $|A|$ means "the size of set $A$"
{{% /expand %}}

---

## Probability Concepts

### probability
{{% expand "probability" %}} The *probability* of an event $A$ relative to an outcome space $\Omega$ is the ratio of their sizes: $P(A) = \frac{|A|}{|\Omega|}$.

When outcomes are weighted (not equally likely), we sum the weights instead of counting.

**Interpretation:** "What fraction of possible outcomes are in event $A$?"
{{% /expand %}}

### conditional probability
{{% expand "conditional probability" %}}The *conditional probability* is the probability of an event conditioned on knowledge of another event. Conditioning on an event means that the possible outcomes in that event form the set of possibilities or outcome space. We then calculate probabilities as normal within that *restricted* outcome space.

Formally, this is written as $P(A \mid B) = \frac{|A \cap B|}{|B|}$, where everything to the left of the $\mid$ is what we're interested in knowing the probability of and everything to the right of the $\mid$ is what we know to be true.

**Alternative formula:** $P(A \mid B) = \frac{P(A,B)}{P(B)}$ (assuming $P(B) > 0$)
{{% /expand %}}

### the other definition of conditional probability
{{% expand "the other definition of conditional probability" %}} Using joint and marginal probabilities, conditional probability can be defined as the ratio of the joint probability to the marginal probability of the conditioned information:

$$P(A \mid B) = \frac{P(A,B)}{P(B)}$$

This is equivalent to the set-based definition but uses probability formulas instead of counting.
{{% /expand %}}

### marginal probability
{{% expand "marginal probability" %}} A *marginal probability* is the probability of a random variable that has been calculated by summing over the possible values of one or more other random variables.

**Formula:** $P(A) = \sum_{b} P(A, B=b)$

**Intuition:** "What's the probability of $A$ regardless of what $B$ is?"
{{% /expand %}}

### joint probability
{{% expand "joint probability" %}} The *joint probability* is the probability that multiple events all occur. This corresponds to the intersection of the events (outcomes that are in all the events).

**Notation:** $P(A, B)$ or $P(A \cap B)$

**Intuition:** "What's the probability that both $A$ and $B$ happen?"
{{% /expand %}}

---

## Relationships Between Events

### dependence
{{% expand "dependence" %}}  When knowing the outcome of one random variable or event influences the probability of another, those variables or events are called *dependent*. This is denoted as $A \not\perp B$.

When they do not influence each other, they are called *independent*. This is denoted as $A \perp B$.

**Formal definition of independence:** $P(A \mid B) = P(A)$, or equivalently, $P(A, B) = P(A) \times P(B)$
{{% /expand %}}

---

## Random Variables

### random variable
{{% expand "random variable" %}} A *random variable* is a function that maps from the set of possible outcomes to some set or space. The output or range of the function could be the set of outcomes again, a whole number based on the outcome (e.g., counting the number of Tonkatsu), or something more complex (e.g., the world's friendship matrix, an 8-billion by 8-billion binary matrix where $N_{1,100}=1$ if person 1 is friends with person 100).

Technically the output must be *measurable*. You shouldn't worry about that distinction unless your random variable's output gets really, really big (like continuous). We'll talk more about probabilities over continuous random variables later.

**Key insight:** It's called "random" because its value depends on which outcome occurs, but it's really just a function!
{{% /expand %}}

---

## Advanced Concepts

### Bayes theorem
{{% expand "Bayes Theorem" %}} *Bayes Theorem* (or Bayes' rule) is a formula for reversing the order that variables are conditioned — how to go from $P(A \mid B)$ to $P(B \mid A)$.

**Formula:** $P(H \mid D) = \frac{P(D \mid H) P(H)}{P(D)}$

**Components:**
- $P(H \mid D)$ = posterior (updated belief after seeing data)
- $P(D \mid H)$ = likelihood (how well data fits hypothesis)
- $P(H)$ = prior (belief before seeing data)
- $P(D)$ = evidence (total probability of data)

**Application:** Updating beliefs with new information
{{% /expand %}}

### generative process
{{% expand "generative process" %}} A *generative process* defines the probabilities for possible outcomes according to an algorithm with random choices. Think of it as a recipe for producing outcomes.

**Example:** "Flip two coins: first for lunch (H or T), second for dinner (H or T). Record the pair."

This connects to probabilistic programming, where we write code that generates outcomes.
{{% /expand %}}

### probabilistic computing
{{% expand "probabilistic computing" %}} *Probabilistic computing* refers to programming languages and systems for specifying probabilistic models and performing inference (calculating different probabilities according to the model) in an efficient manner.

**Examples:** GenJAX, PyMC, Stan, Turing.jl

**Key idea:** Instead of listing all outcomes by hand, write code that generates them, and let the computer do the counting!
{{% /expand %}}

---

## Additional Terms

### Monte Carlo simulation
{{% expand "Monte Carlo simulation" %}} A computational method for approximating probabilities by generating many random samples and counting outcomes. Named after the Monte Carlo casino.

**Process:**
1. Generate many random outcomes (e.g., 10,000 simulated days)
2. Count how many satisfy your event
3. Calculate the ratio

**When useful:** When outcome spaces are too large to enumerate by hand
{{% /expand %}}

### trace
{{% expand "trace" %}} In probabilistic programming, a *trace* records all random choices made during one execution of a generative function, along with their addresses (names) and the return value.

**Think of it as:** A complete record of "what happened" during one run of a probabilistic program

**Used in:** GenJAX and other probabilistic programming systems
{{% /expand %}}

### generative function
{{% expand "generative function" %}} In GenJAX and similar systems, a generative function is a Python function decorated with `@gen` that can make addressed random choices. It represents a probability distribution over its return values.

**Example:**
```python
@gen
def coin_flip():
    result = bernoulli(0.5) @ "flip"
    return result
```
{{% /expand %}}

### choice map
{{% expand "choice map" %}} A dictionary-like structure in GenJAX that maps addresses (names) to the values of random choices. Used for:
- Recording what random choices were made (from traces)
- Specifying observations for inference
- Constraining random choices

**Think of it as:** A way to name and track all the random decisions
{{% /expand %}}

---

|[← Previous: Bayes' Theorem](./05_bayes.md) | [Next: Acknowledgements →](./07_ack.md)|
| :--- | ---: |
