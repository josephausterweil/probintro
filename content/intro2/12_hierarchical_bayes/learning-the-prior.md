+++
date = "2026-06-03"
title = "Learning the Prior"
weight = 3
+++

## Where does the population prior come from?

So far we *fixed* $(a, b) = (6, 4)$ by hand. But the whole promise of this chapter was to **learn** the prior.
The hierarchical model already contains the answer: $(a, b)$ is itself a latent variable with its own
distribution, so we can **infer it from the students' data** — the same way we've inferred every other unknown
in this tutorial.

We put a broad, weakly-informative **hyperprior** on $(a, b)$ — a *prior on the population prior*, just "some
plausible range of population shapes, nothing committed" (below, a uniform box over $0.5 \le a, b \le 20$;
widen it and the estimate barely moves until the bounds get extreme) — observe all the students' counts, and
weight candidate $(a, b)$ values by how well they explain the data. This is plain **importance sampling** — the exact tool from
[Chapter 5](../../05_mixture_models/) and the GenJAX tutorial, now aimed one level up at the hyperparameters.

To score a candidate $(a, b)$ we need the probability it assigns to a student's count $k_i$ — but $(a, b)$
only tells us the *distribution* of that student's rate $\theta_i$, not its value. So we **average over all
possible $\theta_i$**: this is the same Beta-Binomial conjugacy that let us update
$\text{Beta}(a,b) \to \text{Beta}(a+k, b+n-k)$ in §2, used the other direction. The average has a clean closed
form (the **Beta-Binomial** marginal):

$$p(k_i \mid n_i, a, b) = \binom{n_i}{k_i}  \frac{B(a + k_i,  b + n_i - k_i)}{B(a, b)},$$

where $\binom{n_i}{k_i}$ is the binomial coefficient ("$n_i$ choose $k_i$") and $B(\cdot,\cdot)$ is the Beta
function — the normalizer of the Beta distribution, whose log is `betaln` in JAX. We don't need to memorize it;
we just sum its log across students to score a population:

<!-- validate: tol=1.5 -->
```python
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import betaln, gammaln

k = jnp.array([70, 28, 6, 3, 2, 0])
n = jnp.array([100, 40, 10, 5, 2, 1])

def log_binom_coeff(n, k):
    # gammaln = log of the Gamma function (a continuous factorial); this is log of "n choose k".
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def population_loglik(a, b):
    """log p(all students' counts | a, b), theta integrated out (Beta-Binomial)."""
    per_student = (log_binom_coeff(n, k)
                   + betaln(a + k, b + n - k)
                   - betaln(a, b))
    return per_student.sum()

# Importance sampling over (a, b): draw many candidate populations from a broad
# hyperprior, weight each by how well it explains the data, report the weighted mean.
key = jr.PRNGKey(0)
ka, kb = jr.split(key)
N = 20000
a_samples = jr.uniform(ka, (N,), minval=0.5, maxval=20.0)   # broad hyperprior on a
b_samples = jr.uniform(kb, (N,), minval=0.5, maxval=20.0)   # broad hyperprior on b

log_w = jax.vmap(population_loglik)(a_samples, b_samples)
w = jnp.exp(log_w - log_w.max())
w = w / w.sum()

a_post = jnp.sum(w * a_samples)
b_post = jnp.sum(w * b_samples)
print(f"inferred a ~= {float(a_post):.2f}")
print(f"inferred b ~= {float(b_post):.2f}")
print(f"implied population tonkatsu rate ~= {float(a_post / (a_post + b_post)):.3f}")
```

**Output:**
```
inferred a ~= 14.57
inferred b ~= 8.12
implied population tonkatsu rate ~= 0.642
```

The data alone pinned the population rate at about **0.64** — in the same ballpark as the 0.60 we had
hand-picked, but now *learned* from the six students rather than assumed (it lands a bit higher because the
heavy bringers, Alyssa and Ben at 0.70, carry most of the evidence). We never told the model the population
mean; it inferred it, and that inferred prior is what then shrinks each student's estimate.

{{% notice style="warning" title="Honest note: importance sampling is noisy here" %}}
As in the [Chapter 5](../../05_mixture_models/) mixture inference, importance sampling over a broad hyperprior is
a *blunt* tool — most sampled $(a, b)$ explain the data poorly, so only a few carry real weight, and the
estimate wobbles from run to run. That's expected, and it's the honest face of the method. Sharper inference
(MCMC, variational methods) is a topic for later; the point here is conceptual: **"learn the prior" is just
inference, one level up.**
{{% /notice %}}

"The prior has its own prior" is not an infinite regress — it bottoms out at a weakly-informative hyperprior
you're willing to commit to, and the data does the rest. That is the whole trick of hierarchical Bayes.

---

## What the model learns about *variability* — and why Farid depends on it

The inference above learned a single number for the population — its mean rate, about $0.64$. But back in the concentration discussion we saw that the mean is only half the story; the **concentration** $a + b$ decides *how much students differ*, and that is what controls how hard each student is shrunk. So let's separate the two explicitly and let the model learn **both**.

### Reparameterizing: mean and concentration

It is awkward to put a hyperprior directly on $(a, b)$, because the pair entangles "what's the average rate?" with "how alike are students?". Following the standard move in hierarchical models (e.g. Kemp, Perfors, & Tenenbaum, 2007), we **reparameterize** into those two independent questions:

$$\mu = \frac{a}{a+b} \quad (\text{the population } \textbf{mean}), \qquad \lambda = a + b \quad (\text{the } \textbf{concentration}),$$

and invert with $a = \mu\lambda,  b = (1-\mu)\lambda$. Now we can put a *separate* hyperprior on each — and the crucial one is on $\lambda$.

{{% notice style="info" title="Same distribution, two sets of dials: $(a, b)$ vs. $(\mu, \lambda)$" %}}
This is **not a new distribution** — it is the *exact same* Beta distribution, described with two different sets of dials. Nothing about the math changes; we are only relabeling.

| | Standard parameterization | Reparameterization |
|---|---|---|
| **Parameters** | $a,\ b$ (the two "soft counts") | $\mu = \frac{a}{a+b}$, $\ \lambda = a + b$ |
| **What each dial does** | $a$ and $b$ each pull *both* the mean and the spread | $\mu$ sets the **mean** alone; $\lambda$ sets the **concentration** alone |
| **Convert** | — | $a = \mu\lambda,\quad b = (1-\mu)\lambda$ |
| **Example** | $\text{Beta}(6, 4)$ | $\mu = 0.6,\ \lambda = 10$ |

$\text{Beta}(6,4)$ and "$\mu = 0.6,\ \lambda = 10$" are **the same distribution written two ways** — plug $\mu\lambda = 6$ and $(1-\mu)\lambda = 4$ back in to check. We switch to $(\mu, \lambda)$ for one reason: it lets us reason about — and put independent priors on — "what's the average rate?" and "how alike are students?" *separately*, which is precisely the distinction this section is about. Everywhere a model wants $a$ and $b$ (as in `beta(a, b)`), we still pass $a = \mu\lambda$ and $b = (1-\mu)\lambda$.
{{% /notice %}} To let the model discover a U-shaped population (each student near-deterministic, $\lambda < 1$) **or** a tight one (students all alike, $\lambda \gg 1$), the hyperprior on $\lambda$ must **span both regimes** — orders of magnitude above and below $1$. A **log-uniform** prior does exactly that:

$$\mu \sim \text{Uniform}(0, 1), \qquad \log \lambda \sim \text{Uniform}(\log 0.1,  \log 100).$$

{{% notice style="warning" title="Why the hyperprior on $\lambda$ must reach below 1" %}}
If your hyperprior can't produce $\lambda < 1$, your model *cannot* represent a population of near-deterministic students — it is structurally blind to the U-shape, no matter what the data say. A naive box like "$a, b \in [0.5, 20]$" forces $\lambda = a + b \ge 1$ and quietly rules out heterogeneity. Spanning $\lambda$ across orders of magnitude (here via log-uniform) is what lets the **data** choose the regime.
{{% /notice %}}

### Two classrooms, the same Emi and Farid

Here is the payoff, and it answers a question you might have had all along: *does shrinking Farid (0/1) up toward the average always make sense?* **No — it depends on the company Farid keeps.** Consider two different classrooms, each with the **same two data-light students** — Emi (2/2) and Farid (0/1) — but different *heavy* bringers. (Throughout, $k_i/n_i$ counts the **bentos student $i$ has brought Chibany** — how often they bring *tonkatsu* — not what the student eats themselves.)

- **Classroom A — mixed bringers.** Alyssa 70/100, Ben 28/40, Carmen 6/10, Diego 3/5. The well-observed students bring tonkatsu at middling rates ($0.6$–$0.7$): nobody reliably brings just one kind. This data is exactly what a **concentrated** population looks like.
- **Classroom B — creatures of habit.** Alyssa 97/100, Ben 2/40, Carmen 19/20, Diego 0/20. The well-observed students *almost always bring the same thing* — Alyssa and Carmen nearly always bring tonkatsu, Ben and Diego nearly always bring hamburger. This data is what a **U-shaped** population looks like.

We feed each classroom to the same inference and let it learn $\mu$ and $\lambda$, then shrink Emi and Farid with whatever population it found:

<!-- validate: tol=0.06 -->
```python
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import betaln, gammaln

def log_beta_binom(k, n, a, b):
    # log p(k | n, a, b): the Beta-Binomial marginal (theta integrated out).
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) \
        + betaln(a + k, b + n - k) - betaln(a, b)

def infer_population(k, n, seed=0, n_samples=60000):
    """Infer (mu, lambda) by importance sampling, a = mu*lam, b = (1-mu)*lam.
    Hyperpriors: mu ~ Uniform(0,1); lambda log-uniform over [0.1, 100] so it can
    land BELOW 1 (students differ / U-shaped) or ABOVE 1 (students are alike)."""
    km, kl = jr.split(jr.key(seed))
    mu = jr.uniform(km, (n_samples,), minval=0.01, maxval=0.99)
    lam = jnp.exp(jr.uniform(kl, (n_samples,), minval=jnp.log(0.1), maxval=jnp.log(100.0)))
    a, b = mu * lam, (1 - mu) * lam
    log_w = jax.vmap(lambda a, b: log_beta_binom(k, n, a, b).sum())(a, b)
    w = jnp.exp(log_w - log_w.max()); w = w / w.sum()
    return float(jnp.sum(w * mu)), float(jnp.sum(w * lam))

# Both classrooms share the SAME two data-light students: Emi 2/2, Farid 0/1.
classrooms = {
    "A (mixed bringers)":    (jnp.array([70, 28, 6, 3, 2, 0]),  jnp.array([100, 40, 10, 5, 2, 1])),
    "B (creatures of habit)":(jnp.array([97, 2, 19, 0, 2, 0]),  jnp.array([100, 40, 20, 20, 2, 1])),
}

for label, (k, n) in classrooms.items():
    mu, lam = infer_population(k, n)
    a, b = mu * lam, (1 - mu) * lam
    print(f"Classroom {label}:  inferred mean mu={mu:.2f}, concentration lambda={lam:.1f}")
    for name, ki, ni in [("Emi", 2, 2), ("Farid", 0, 1)]:
        print(f"    {name} {ki}/{ni}: raw {ki/ni:.2f} -> shrunk {(a + ki) / (a + b + ni):.2f}")
```

**Output:**
```
Classroom A (mixed bringers):  inferred mean mu=0.66, concentration lambda=41.3
    Emi 2/2: raw 1.00 -> shrunk 0.68
    Farid 0/1: raw 0.00 -> shrunk 0.65
Classroom B (creatures of habit):  inferred mean mu=0.47, concentration lambda=0.6
    Emi 2/2: raw 1.00 -> shrunk 0.89
    Farid 0/1: raw 0.00 -> shrunk 0.17
```

The two classrooms learn opposite concentrations — $\lambda \approx 41$ (students alike) for A, $\lambda \approx 0.6$ (students differ, U-shaped) for B — **purely from their heavy bringers**, and that flips the verdict on the *identical* data-light students:

| Student (same data) | Classroom A ($\lambda \approx 41$) | Classroom B ($\lambda \approx 0.6$) |
|---|---:|---:|
| **Emi** (2/2) | $1.00 \to 0.68$ (pulled to the mean) | $1.00 \to 0.89$ (believed near-1) |
| **Farid** (0/1) | $0.00 \to 0.65$ (pulled to the mean) | $0.00 \to 0.17$ (believed near-0) |

In classroom A, everyone else is moderate, so a single hamburger brought by Farid is almost surely a fluke — shrink it hard toward the group. In classroom B, everyone else is an extremist, so Farid's single hamburger is taken nearly at face value — *this is probably a student who always brings hamburger*. **Same observation about Farid, opposite conclusion, because the population told the model how much to trust one data point.** That — learning how much to trust thin data from the structure of everyone else's — is the deepest thing hierarchical Bayes does.

---
