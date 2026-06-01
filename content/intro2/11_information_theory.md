+++
date = "2026-06-01"
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

The **surprise** (or *information content*; [glossary](../../glossary/#surprise-information-content-)) of an outcome $x$ is

$$\text{surprise}(x) = -\log_2 P(x),$$

measured in **bits** when the log is base 2. The idea: the *less* probable an outcome was, the *more* surprised you should be when it happens. Walk through it:

- You assigned $P = 0.99$ to tonkatsu, and tonkatsu came: surprise $= -\log_2(0.99) \approx 0.014$ bits. Almost none — you basically knew.
- You assigned $P = 0.01$ to hamburger, and hamburger came: surprise $= -\log_2(0.01) \approx 6.64$ bits. A lot — the world contradicted a confident belief.

Why the logarithm, of all functions? Because surprise should be **additive over independent events**: learning two independent things should surprise you by the sum of the separate surprises. Since independent probabilities *multiply* ($P(x, y) = P(x)P(y)$), the function that turns multiplication into addition is the log: $-\log P(x,y) = -\log P(x) - \log P(y)$. The base just sets the unit; base 2 gives bits, the amount of information in one fair coin flip. (There's a deep connection to the length of the optimal code for transmitting outcomes, but we won't need it here.)

---

## Entropy = Expected Surprise

A single outcome has a surprise. A whole *distribution* has an **average** surprise, weighted by how often each outcome occurs. That average is the **entropy** ([glossary](../../glossary/#entropy-)):

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

## Joint and Conditional Entropy

Entropy measures uncertainty about *one* variable. With two variables we get two more quantities — and one clean law connecting them.

**Joint entropy** ([glossary](../../glossary/#joint-entropy-)) is just the entropy of the pair $(X, Y)$, treated as a single combined outcome:

$$H(X, Y) = -\sum_{x, y} P(x, y) \log_2 P(x, y) = \mathbb{E}\bigl[-\log_2 P(X, Y)\bigr].$$

It's the total uncertainty in *both* variables at once — how surprised you expect to be by the full pair.

**Conditional entropy** ([glossary](../../glossary/#conditional-entropy-)) is the uncertainty that *remains* in $Y$ once you already know $X$. For a *specific* value $X = x$, the leftover uncertainty is just the entropy of the conditional distribution $P(Y \mid x)$; the conditional entropy averages that over all values of $X$:

$$H(Y \mid X) = \sum_x P(x)\,\underbrace{\Bigl[-\sum_y P(y \mid x) \log_2 P(y \mid x)\Bigr]}_{H(Y \mid X = x)} = -\sum_{x, y} P(x, y) \log_2 P(y \mid x).$$

Read $H(Y \mid X)$ as: "on average, how surprised will the value of $Y$ still leave me, given that I've seen $X$?" If $X$ pins $Y$ down completely, $H(Y \mid X) = 0$ — no surprise left. If $X$ says nothing about $Y$, then $P(y \mid x) = P(y)$ and $H(Y \mid X) = H(Y)$ — knowing $X$ didn't help at all.

### The chain rule for entropy

These three quantities obey a beautifully simple law — **the total uncertainty in a pair equals the uncertainty in the first variable plus the leftover uncertainty in the second**:

$$\boxed{\,H(X, Y) = H(X) + H(Y \mid X)\,}$$

It's the information-theoretic echo of the probability chain rule $P(x, y) = P(x)\,P(y \mid x)$. Here is the one-line derivation — it's just that chain rule, run through a logarithm:

$$
\begin{aligned}
H(X, Y) &= -\sum_{x, y} P(x, y) \log_2 P(x, y) \\
        &= -\sum_{x, y} P(x, y) \log_2 \bigl[P(x)\,P(y \mid x)\bigr] && \text{(probability chain rule)}\\
        &= -\sum_{x, y} P(x, y) \log_2 P(x) \;-\; \sum_{x, y} P(x, y) \log_2 P(y \mid x) && \text{(}\log\text{ of a product = sum of logs)}\\
        &= -\sum_{x} P(x) \log_2 P(x) \;+\; H(Y \mid X) && \text{(sum out } y \text{ in the first term: } \textstyle\sum_y P(x,y)=P(x)\text{)}\\
        &= H(X) + H(Y \mid X).
\end{aligned}
$$

The whole trick was *the log of the probability chain rule splits a joint surprise into "surprise about $X$" plus "leftover surprise about $Y$ given $X$."* Surprises add the way probabilities multiply — which is exactly why we used a logarithm in the first place.

This immediately gives a tidy formula for conditional entropy by rearranging: $H(Y \mid X) = H(X, Y) - H(X)$. "What's left to learn about $Y$ after $X$" = "the total" minus "what $X$ already covered."

---

## Mutual Information

Now the payoff quantity. **Mutual information** ([glossary](../../glossary/#mutual-information-)) measures how much learning one variable *reduces* your uncertainty about another:

$$I(X; Y) = H(X) - H(X \mid Y).$$

In words: start with your uncertainty about $X$ (that's $H(X)$); subtract the uncertainty that *remains* once you know $Y$ (that's the conditional entropy $H(X \mid Y)$ we just defined). What's left — the reduction — is how much $Y$ told you about $X$. If $Y$ says nothing, the uncertainty doesn't drop and $I(X; Y) = 0$; if $Y$ pins $X$ down completely, the remaining uncertainty is zero and $I(X; Y) = H(X)$.

### Mutual information from the chain rule — and why it's symmetric

Substitute the chain rule $H(X \mid Y) = H(X, Y) - H(Y)$ into the definition and watch a symmetric formula fall out:

$$
\begin{aligned}
I(X; Y) &= H(X) - H(X \mid Y) \\
        &= H(X) - \bigl[H(X, Y) - H(Y)\bigr] && \text{(chain rule, the version } H(X\mid Y)=H(X,Y)-H(Y)\text{)}\\
        &= H(X) + H(Y) - H(X, Y).
\end{aligned}
$$

That last line, $\;I(X; Y) = H(X) + H(Y) - H(X, Y)$, is **completely symmetric in $X$ and $Y$** — swapping them changes nothing. So mutual information is **symmetric**: $Y$ tells you exactly as much about $X$ as $X$ tells you about $Y$.

$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = I(Y; X).$$

Information is mutual — hence the name. (There's a vivid picture here: think of $H(X)$ and $H(Y)$ as two overlapping circles. $H(X,Y)$ is the area of their union, and $I(X;Y)$ is the **overlap** — the bits the two variables share. The formula $I = H(X) + H(Y) - H(X,Y)$ is just inclusion–exclusion for those areas.)

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

## Cross-Entropy and KL Divergence

Every quantity so far measured uncertainty under the *true* distribution. The last two measure what happens when you use the **wrong** distribution — and they are the bridge from this chapter to modern machine learning.

Suppose the world really follows $P$, but you *believe* it follows $Q$ — your model is $Q$, reality is $P$. Surprise was $-\log_2 P(x)$ when your beliefs were right; now you assign probability $Q(x)$ to each outcome, so *your* surprise at outcome $x$ is $-\log_2 Q(x)$. Averaging **your** surprise over the **real** outcomes gives the **cross-entropy** ([glossary](../../glossary/#cross-entropy-)):

$$H(P, Q) = -\sum_x P(x) \log_2 Q(x) = \mathbb{E}_{X \sim P}\bigl[-\log_2 Q(X)\bigr].$$

It is the average surprise you actually feel when reality is $P$ but you predicted with $Q$. Notice $H(P, P) = H(P)$: if your model is exactly right, cross-entropy collapses back to plain entropy.

The **Kullback–Leibler divergence** ([glossary](../../glossary/#kl-divergence-)) is the *excess* — how many **extra** bits of surprise your wrong model costs you, beyond the irreducible $H(P)$:

$$D_{\text{KL}}(P \parallel Q) = \sum_x P(x) \log_2 \frac{P(x)}{Q(x)}.$$

### Cross-entropy = entropy + KL

These two are linked by a one-line identity that says exactly that — *your total surprise is the unavoidable part plus the penalty for being wrong*:

$$\boxed{\,H(P, Q) = H(P) + D_{\text{KL}}(P \parallel Q)\,}$$

The derivation is again just splitting a logarithm, this time using $\log \frac{P}{Q} = \log P - \log Q$ in reverse:

$$
\begin{aligned}
H(P) + D_{\text{KL}}(P \parallel Q)
  &= \Bigl[-\sum_x P(x)\log_2 P(x)\Bigr] + \sum_x P(x)\log_2 \frac{P(x)}{Q(x)} \\
  &= -\sum_x P(x)\log_2 P(x) + \sum_x P(x)\bigl[\log_2 P(x) - \log_2 Q(x)\bigr] \\
  &= -\sum_x P(x)\log_2 Q(x) && \text{(the } \log_2 P(x) \text{ terms cancel)}\\
  &= H(P, Q).
\end{aligned}
$$

Because $H(P)$ doesn't depend on your model $Q$, **minimizing the cross-entropy and minimizing the KL divergence are the same optimization** — they differ only by the constant $H(P)$. That single fact is why "cross-entropy loss" is the default objective for training classifiers and language models: you can't change reality's entropy $H(P)$, so driving down cross-entropy *just is* driving your model $Q$ toward the truth $P$, bit for bit.

{{% notice style="info" title="A useful fact: KL is never negative" %}}
$D_{\text{KL}}(P \parallel Q) \ge 0$ always, with equality only when $Q = P$ (this is **Gibbs' inequality**). So $H(P, Q) \ge H(P)$: predicting with the wrong distribution can *only* increase your average surprise, never decrease it. The truth is the cheapest model to be surprised by — which is the whole reason learning works. (KL is *not* symmetric, though: $D_{\text{KL}}(P \parallel Q) \ne D_{\text{KL}}(Q \parallel P)$ in general, so it's a "divergence," not a true distance.)
{{% /notice %}}

These quantities return in force when this course reaches neural networks and large language models — a classifier's training loss *is* a cross-entropy, and much of probabilistic machine learning is one long campaign to make $Q$ resemble $P$.

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
You can quantify surprise and entropy; decompose the uncertainty in a pair via **joint** and **conditional** entropy and the **chain rule** $H(X,Y) = H(X) + H(Y \mid X)$; express **mutual information** three equivalent ways and see *why* it's symmetric; restate independence and d-separation in information units; measure the collider's explaining-away effect as a concrete number of bits; and connect **cross-entropy** and **KL divergence** to the loss functions of machine learning via $H(P,Q) = H(P) + D_{\text{KL}}(P \parallel Q)$. With that, the Bayesian-networks spine is complete — you can draw a model as a graph, read its independencies, distinguish seeing from doing, and weigh it all in bits. From here the path leads to [hierarchical Bayes](../12_hierarchical_bayes/) (Bayes nets with priors stacked on priors) and, later in the course, to the information-theoretic heart of modern machine learning.
{{% /notice %}}

---

## Exercises

{{% notice style="info" title="Try it yourself" %}}
1. **Entropy by hand.** A weighted die lands on 6 with probability $0.5$ and on each of 1–5 with probability $0.1$. Is its entropy higher or lower than a fair die's ($\log_2 6 \approx 2.585$ bits)? Compute it.
2. **Surprise budget.** Chibany's friend claims to predict bentos with $P = 0.95$ accuracy. Over a 30-day month, what's the *expected* total surprise (in bits) if they're right that often? (Hint: $30 \times H(0.95)$.)
3. **MI and the chain.** In a chain $A \to B \to C$, estimate $I(A; C)$ and $I(A; C \mid B)$ by Monte Carlo (build a small `flip`-based model). Confirm that conditioning on the middle node $B$ drives the mutual information toward zero — the information-theoretic signature of a *blocked* path.
4. **Check the chain rule.** For the rain/tea pair (independent, $P(R)=0.3$, $P(T)=0.1$), compute $H(R)$, $H(T)$, and $H(R, T)$ by hand. Verify $H(R, T) = H(R) + H(T \mid R)$, and confirm that here $H(T \mid R) = H(T)$ (because $R$ and $T$ are independent, knowing $R$ tells you nothing about $T$).
5. **Cross-entropy cost.** You believe a coin is fair ($Q = \text{Bernoulli}(0.5)$) but it's actually biased ($P = \text{Bernoulli}(0.3)$). Compute the cross-entropy $H(P, Q)$ and the entropy $H(P)$. How many *extra* bits of surprise does your wrong belief cost you per flip? (That excess is $D_{\text{KL}}(P \parallel Q)$.)
{{% /notice %}}

A companion notebook works through these interactively:

**📓 [Open in Colab: `11_information_theory.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/11_information_theory.ipynb)**

---

Special thanks to [JPPCA](https://jpcca.org/) for their generous support of this tutorial series.
