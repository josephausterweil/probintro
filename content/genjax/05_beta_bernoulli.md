+++
date = "2025-12-24"
title = "Beta-Bernoulli: Your First Inference"
weight = 6
+++

## From Simulation to Inference

So far, we've been **generating** data from models. But the real power of probabilistic programming is **inference**: given observed data, what can we learn about the hidden variables?

This is like detective work: we see the evidence (coin flips), and we want to deduce the cause (the coin's bias).

---

## The Beta-Bernoulli Model

Imagine a coin with unknown bias $p$. We don't know if it's fair!

**The generative story:**
1. Nature picks a bias $p$ from a Beta(α, β) distribution
2. The coin is flipped, landing heads with probability $p$
3. We observe the outcome

```python
from genjax import gen, beta, flip

@gen
def beta_bernoulli(α, β):
    p = beta(α, β) @ "p"  # Unknown bias
    v = flip(p) @ "v"     # Observed flip
    return v
```

The `@ "p"` and `@ "v"` are **addresses**—named random choices we can later constrain or query.

---

## The Inference Question

> "Given that we observed heads, what's our belief about $p$?"

This is **conditioning**: we fix `v = True` and ask what values of `p` are consistent with this observation.

Mathematically, we want:
$$P(p \mid v = \text{True})$$

This is the **posterior distribution** over the coin's bias.

---

## Inference in GenJAX

GenJAX provides inference algorithms that estimate posteriors. Here's the pattern:

```python
from genjax import Target, ChoiceMap
from genjax.inference.smc import ImportanceK

# Define what we want to infer
posterior_target = Target(
    beta_bernoulli,          # the model
    (2.0, 2.0),              # prior parameters
    ChoiceMap.d({"v": True}), # observation
)

# Run importance sampling with 50 particles
alg = ImportanceK(posterior_target, k_particles=50)
```

---

## Try It Yourself

Open the companion notebook to:

1. **Simulate** from the model and examine traces
2. **Run inference** to estimate $p$ given observations
3. **Experiment** with different priors (α, β values)
4. **Observe** how more data sharpens our beliefs

{{% notice style="info" title="Companion Notebook" %}}
📓 [`beta_bernoulli.ipynb`](/notebooks/beta_bernoulli.ipynb) — Full working code with exercises
{{% /notice %}}

---

## Key Insights

| Concept | What It Means |
|---------|---------------|
| **Prior** | Beta(α, β) — our belief before seeing data |
| **Likelihood** | Bernoulli(p) — how data is generated |
| **Posterior** | Updated belief after seeing data |
| **Inference** | Computing the posterior from prior + data |

---

## What's Next?

With Beta-Bernoulli mastered, you're ready for:
- **Multiple observations** — what if we see 10 flips?
- **More complex models** — Gaussian mixtures, hierarchical models
- **Different inference algorithms** — MCMC, variational inference

The core pattern stays the same: define a model, specify observations, run inference.
