+++
date = "2026-05-31"
title = "Information Theory: Surprise, Uncertainty, and the Collider"
weight = 11
+++

## Measuring Surprise

We've reached the last chapter of the Bayes-net spine, and it ties everything together with a single idea: **information**. So far we've talked about whether one variable tells you *something* about another — dependence, independence, d-separation. Information theory lets us say *how much*, in a precise, additive unit: the **bit**.

Chibany has been keeping a journal of which bento they get each day. Some days they predict tonkatsu and they're right; some days they're caught off guard. Today they were sure it would be tonkatsu — and got a hamburger.

> **Chibany:** "Ugh, I was *so* sure. How surprised should I even have been?"
>
> **Jamal:** "Funny you ask — there's an exact number for that."

That number is where information theory begins.

---

## Surprise = $-\log P(x)$

The **surprise** (or *information content*) of an outcome $x$ is

$$\text{surprise}(x) = -\log_2 P(x),$$

measured in **bits** when the log is base 2. The idea: the *less* probable an outcome was, the *more* surprised you should be when it happens. Walk through it:

- You assigned $P = 0.99$ to tonkatsu, and tonkatsu came: surprise $= -\log_2(0.99) \approx 0.014$ bits. Almost none — you basically knew.
- You assigned $P = 0.01$ to hamburger, and hamburger came: surprise $= -\log_2(0.01) \approx 6.64$ bits. A lot — the world contradicted a confident belief.

Why the logarithm, of all functions? Because surprise should be **additive over independent events**: learning two independent things should surprise you by the sum of the separate surprises. Since independent probabilities *multiply* ($P(x, y) = P(x)P(y)$), the function that turns multiplication into addition is the log: $-\log P(x,y) = -\log P(x) - \log P(y)$. The base just sets the unit; base 2 gives bits, the amount of information in one fair coin flip. (There's a deep connection to the length of the optimal code for transmitting outcomes, but we won't need it here.)

---

## Entropy = Expected Surprise

A single outcome has a surprise. A whole *distribution* has an **average** surprise, weighted by how often each outcome occurs. That average is the **entropy**:

$$H(X) = \sum_x P(x)\,\bigl(-\log_2 P(x)\bigr) = -\sum_x P(x) \log_2 P(x) = \mathbb{E}\bigl[-\log_2 P(X)\bigr].$$

Entropy measures how *uncertain* you are about $X$ before you see it — how surprised you expect to be, on average. Three cases pin down the intuition:

| Distribution | Entropy | Reading |
|---|---:|---|
| Fair coin ($P = 0.5$) | $1.0$ bit | Maximum uncertainty for two outcomes |
| Biased coin ($P = 0.7$) | $0.881$ bits | Less uncertain — you can guess "heads" and often be right |
| Certain outcome ($P = 1$) | $0$ bits | No uncertainty, no surprise, no information |

A fair coin is the *most* uncertain a binary variable can be — one full bit. Bias it, and the entropy drops: the more predictable a variable, the less information its outcome carries. A deterministic outcome carries none at all.

{{% notice style="info" title="Notation note" %}}
$\mathbb{E}[\cdot]$ is the **expected value** (average) from [Chapter 1](../01_mystery_bentos/) — here, the average of the surprise $-\log_2 P(X)$ over all outcomes, weighted by their probabilities. So entropy is literally "expected surprise."
{{% /notice %}}

---

## Mutual Information

Entropy measures uncertainty about *one* variable. **Mutual information** measures how much learning one variable *reduces* your uncertainty about another:

$$I(X; Y) = H(X) - H(X \mid Y).$$

In words: start with your uncertainty about $X$ (that's $H(X)$); subtract the uncertainty that *remains* once you know $Y$ (that's $H(X \mid Y)$, the conditional entropy). What's left — the reduction — is how much $Y$ told you about $X$. If $Y$ says nothing, the uncertainty doesn't drop and $I(X; Y) = 0$. If $Y$ pins $X$ down completely, the remaining uncertainty is zero and $I(X; Y) = H(X)$.

A small, pleasing fact: mutual information is **symmetric**. $Y$ tells you exactly as much about $X$ as $X$ tells you about $Y$:

$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = I(Y; X).$$

Information is mutual — hence the name.

---

## Independence, in Information Units

This gives a clean, quantitative restatement of the central concept of [Chapter 9](../09_conditional_independence/). Two variables are independent **exactly when their mutual information is zero**:

$$X \perp Y \iff I(X; Y) = 0.$$

And the conditional version mirrors d-separation:

$$X \perp Y \mid Z \iff I(X; Y \mid Z) = 0,$$

where $I(X; Y \mid Z)$ is the mutual information computed after conditioning on $Z$. "Independent" and "carries zero bits about" are the same statement. Mutual information is just dependence with a number attached.

---

## The Collider, in Information-Theoretic Clothing

Now the payoff — and it ties the whole spine together. Recall the collider from [Chapter 9](../09_conditional_independence/): two independent causes of one shared effect. Here the two causes are rain ($R$) and a spilled **tea** ($T$), and the shared effect is the cafeteria's wet-floor **sign** ($S$) going out — a deterministic OR, exactly the explaining-away structure from before. Watch the mutual information between the two *causes*:

- **Before observing the sign:** $R$ and $T$ are independent, so $I(R; T) = 0$ bits. Knowing it rained tells you nothing about tea spills.
- **After observing the sign is out:** conditioning on the collider *opens* the path, and

$$I(R; T \mid S = 1) \approx 0.462 \text{ bits} > 0.$$

**Conditioning on the collider created mutual information out of nothing.** Two variables that shared zero bits now share almost half a bit — purely because we observed their common effect. This is *exactly* the explaining-away effect from Chapter 9, now with a number on it: the $0.462$ bits is the precise amount of dependence the collider manufactures. Explaining away isn't a vague "the variables become related" — it's a measurable quantity of information conjured by an observation.

{{% notice style="success" title="The spine, in one sentence" %}}
A Bayes net says which variables share information ([Ch 8](../08_bayes_nets/)); d-separation says when conditioning turns that sharing on or off ([Ch 9](../09_conditional_independence/)); the do-operator says when sharing reflects *causation* you can act on ([Ch 10](../10_causal_bayes_nets/)); and mutual information says *how many bits* are shared. The collider — independent causes, dependent once you condition on their effect — is the thread running through all four.
{{% /notice %}}

---

## A Note on KL Divergence

One more quantity you'll meet again. The **Kullback–Leibler divergence** measures how far one distribution $Q$ is from another $P$, in surprise units:

$$D_{\text{KL}}(P \parallel Q) = \sum_x P(x) \log_2 \frac{P(x)}{Q(x)}.$$

Read it as: "if the world really follows $P$ but you *believe* $Q$, how many extra bits of surprise does your wrong belief cost you, on average?" It's zero exactly when $Q = P$, and positive otherwise. KL divergence is the workhorse of modern machine learning — training a model often means *minimizing* the KL divergence between the data's true distribution and the model's — and it will return when this course reaches neural networks and large language models. For now, just file it away as "distance between distributions, measured in bits."

---

## GenJAX Implementation

Let's estimate entropy and the collider's mutual information by Monte Carlo — no formulas, just samples. We sample many traces from the rain/tea/sign network, then count.

<!-- validate: tol=0.03 -->
```python
import jax
import jax.numpy as jnp
import jax.random as jr
from genjax import gen, flip

@gen
def rain_tea_sign():
    rain = flip(0.3) @ "rain"
    tea = flip(0.1) @ "tea"
    # Deterministic OR: the wet-floor sign is out iff either cause fired.
    either = jnp.maximum(rain.astype(int), tea.astype(int))
    sign = flip(either.astype(float)) @ "sign"
    return rain, tea, sign

# Sample a big batch of full traces.
N = 200000
keys = jr.split(jr.key(0), N)
rain, tea, sign = jax.vmap(lambda k: rain_tea_sign.simulate(k, ()).get_retval())(keys)
rain, tea, sign = rain.astype(int), tea.astype(int), sign.astype(int)

def entropy_bits(samples):
    """Empirical entropy (bits) of a 0/1 sample array."""
    p1 = jnp.mean(samples)
    p1 = jnp.clip(p1, 1e-12, 1 - 1e-12)
    return float(-(p1 * jnp.log2(p1) + (1 - p1) * jnp.log2(1 - p1)))

def mi_bits(a, b, mask=None):
    """Empirical mutual information I(A;B) in bits, optionally within a subset."""
    if mask is not None:
        a, b = a[mask], b[mask]
    mi = 0.0
    n = a.shape[0]
    for av in (0, 1):
        for bv in (0, 1):
            p_ab = jnp.mean((a == av) & (b == bv))
            p_a = jnp.mean(a == av)
            p_b = jnp.mean(b == bv)
            if p_ab > 0 and p_a > 0 and p_b > 0:
                mi += float(p_ab * jnp.log2(p_ab / (p_a * p_b)))
    return mi

print(f"H(rain)             = {entropy_bits(rain):.3f} bits")
print(f"H(sign)             = {entropy_bits(sign):.3f} bits")
print(f"I(rain; tea)              = {mi_bits(rain, tea):.3f} bits   (independent)")
print(f"I(rain; tea | sign = 1)   = {mi_bits(rain, tea, sign == 1):.3f} bits   (collider opened)")
```

**Output:**
```
H(rain)             = 0.881 bits
H(sign)             = 0.950 bits
I(rain; tea)              = 0.000 bits   (independent)
I(rain; tea | sign = 1)   = 0.462 bits   (collider opened)
```

The empirical numbers confirm the theory exactly. `H(rain) = 0.881` bits — a $0.3$ coin, same entropy as the $0.7$ coin from the table (entropy is symmetric in $p$ and $1-p$). The sign is wet with probability $1 - 0.7 \times 0.9 = 0.37$, so `H(sign) = 0.950` bits — closer to a full bit because $0.37$ is nearer the maximally-uncertain $0.5$. And the two mutual-information lines are the whole spine in miniature: **$0.000$ bits** between the independent causes, jumping to **$0.462$ bits** the instant we condition on their shared effect. Explaining away, measured in bits.

{{% notice style="success" title="What you can do now — and the spine complete" %}}
You can quantify surprise, entropy, and mutual information; restate independence and d-separation in information units; and measure the collider's explaining-away effect as a concrete number of bits. With that, the Bayesian-networks spine is complete — you can draw a model as a graph, read its independencies, distinguish seeing from doing, and weigh it all in bits. From here the path leads to [hierarchical Bayes](../12_hierarchical_bayes/) (Bayes nets with priors stacked on priors) and, later in the course, to the information-theoretic heart of modern machine learning.
{{% /notice %}}

---

## Exercises

{{% notice style="info" title="Try it yourself" %}}
1. **Entropy by hand.** A weighted die lands on 6 with probability $0.5$ and on each of 1–5 with probability $0.1$. Is its entropy higher or lower than a fair die's ($\log_2 6 \approx 2.585$ bits)? Compute it.
2. **Surprise budget.** Chibany's friend claims to predict bentos with $P = 0.95$ accuracy. Over a 30-day month, what's the *expected* total surprise (in bits) if they're right that often? (Hint: $30 \times H(0.95)$.)
3. **MI and the chain.** In a chain $A \to B \to C$, estimate $I(A; C)$ and $I(A; C \mid B)$ by Monte Carlo (build a small `flip`-based model). Confirm that conditioning on the middle node $B$ drives the mutual information toward zero — the information-theoretic signature of a *blocked* path.
{{% /notice %}}

A companion notebook works through these interactively:

**📓 [Open in Colab: `11_information_theory.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/11_information_theory.ipynb)**

---

Special thanks to [JPPCA](https://jpcca.org/) for their generous support of this tutorial series.
