+++
date = "2026-06-24"
title = "No Free Lunch & Summary"
weight = 4
+++

## No Free Lunch: why the prior is unavoidable

Every section so far leaned on a hypothesis space $\mathcal{H}$ that we *chose* — seven sensible number-rules,
"multiples of $k$" for the numbers, intervals for the continuous case. We even saw in §4 that the choice of
$\mathcal{H}$ *is* a prior: anything left out has probability zero. It's natural to feel uneasy about that. Isn't
choosing $\mathcal{H}$ cheating? Shouldn't a truly unbiased learner consider **every** possible rule and let the
data sort it out? This last section shows why the answer is a hard **no** — and why the prior isn't a wart on the
method but the very thing that makes learning possible.

### The mirror-world argument

Here is the cleanest version of the idea (due to [Wolpert, 1996](#references)). Strip the problem to the bone: you see two
bits, $0$ then $1$, and must predict the third bit $x_3$. With no assumptions, the world could continue either
way — $0, 1, \mathbf{0}$ or $0, 1, \mathbf{1}$ — and *a priori* nothing favors one over the other.

Whatever your prediction rule outputs, picture its **mirror world**: the world identical in the data you saw
($0, 1$) but with the opposite continuation. If your rule predicts $x_3 = 0$, it is right in the world
$0,1,\mathbf 0$ and wrong in its mirror $0,1,\mathbf 1$. Those two worlds are equally possible, so across the
pair your rule scores exactly one hit out of two. And **every** world has a mirror — so averaged over *all
possible worlds*, every rule, no matter how clever, scores exactly $1/2$. No learning algorithm beats any
other when you average over all the ways the world could be. That is the **No Free Lunch theorem**: with no
prior assumptions, the data you've seen tells you *nothing* about what you haven't.

### Watching learning collapse

That sounds abstract, so let's make it happen. To keep the full hypothesis space small enough to enumerate by
brute force, shrink the universe to just the numbers **1 through 6**. All chapter long we used a small,
sensible $\mathcal{H}$ of named rules. What if instead we go "unbiased" and throw in **every possible rule** —
all $2^6 - 1 = 63$ non-empty subsets of $\{1,2,3,4,5,6\}$, with a uniform prior? We can build that and re-run
the prediction:

```python
import itertools
import jax
import jax.numpy as jnp

# The FULL hypothesis space: every non-empty subset of the numbers 1..6, as 0/1 rows.
# itertools.product([0,1], repeat=6) is Python's built-in "all combinations" generator:
# with two choices (0 or 1) in each of 6 slots, it walks through all 2^6 = 64 patterns.
# The list comprehension's `if sum(bits) > 0` keeps only patterns with at least one 1,
# dropping the all-zeros "empty rule" -- leaving 2^6 - 1 = 63 rules.
rows = [list(bits) for bits in itertools.product([0, 1], repeat=6) if sum(bits) > 0]
H_full = jnp.array(rows, dtype=jnp.float32)
prior_full = jnp.ones(H_full.shape[0]) / H_full.shape[0]   # uniform over all 63 rules
print("number of hypotheses:", H_full.shape[0])

def predictive(observed, H, prior, strong):
    """Posterior-weighted vote over an arbitrary hypothesis matrix H (same machinery as §5/§6)."""
    def log_like(rule):
        in_rule = rule[observed]
        size = rule.sum()
        per_strong = jnp.where(in_rule > 0, -jnp.log(size), -jnp.inf)
        per_weak   = jnp.where(in_rule > 0, 0.0,            -jnp.inf)
        per = per_strong if strong else per_weak
        return per.sum()
    log_post = jnp.log(prior) + jax.vmap(log_like)(H)
    log_post = log_post - jnp.max(log_post)
    post = jnp.exp(log_post)
    post = post / post.sum()
    # predict each number y, one column at a time (as in §5; number n is column n-1)
    for n in range(1, 7):
        p = jnp.sum(post * H[:, n - 1])
        print(f"  P({n}) = {round(float(p), 3)}")

observed = jnp.array([1])    # we saw the number 2 (column index 1)
print("WEAK sampling, full 63-rule space:")
predictive(observed, H_full, prior_full, strong=False)
```

**Output:**
```
number of hypotheses: 63
WEAK sampling, full 63-rule space:
  P(1) = 0.5
  P(2) = 1.0
  P(3) = 0.5
  P(4) = 0.5
  P(5) = 0.5
  P(6) = 0.5
```

There it is: **generalization has collapsed.** Having seen the number 2, the model predicts every *other*
number has the property with probability exactly $0.5$ — a coin flip. It learned nothing transferable. The
reason is pure No Free Lunch: among all 63 rules, any unobserved number sits in exactly half of the ones still
consistent with "2 has it," so its vote is split right down the middle. The number 2 itself reads 1.0 (every
surviving rule contains it, by construction), but the data says nothing about any other number.

Compare this to §5, where a *small, structured* $\mathcal{H}$ of seven named rules gave a real, graded
gradient off the very same kind of single observation. The **only** difference is which hypotheses we were
willing to consider. The structured $\mathcal{H}$ wasn't cheating — it was the **inductive bias** (the built-in
assumptions a learner brings before seeing any data) that made generalization possible at all.

{{% notice style="warning" title="The lesson, stated plainly" %}}
A learner that assumes nothing learns nothing. Generalization *requires* a prior — a commitment, before seeing
data, about which patterns are even worth considering. The size principle, the choice of hypothesis space, the
exponential prior on interval size: these aren't optional embellishments on top of "pure" Bayesian learning.
They **are** the learning. This is why the chapter spent so long on $\mathcal{H}$ and $p(h)$: they are where a
learner's knowledge of the world actually lives.
{{% /notice %}}

This same move — replace a structured $\mathcal{H}$ with *all* subsets and watch generalization flatten — is
what the **"break your model"** part of this chapter's assignment asks you to do (in your own chosen domain),
and then to explain in terms of No Free Lunch.

---

## Summary, and where this goes next

{{% notice style="success" title="Key takeaways" %}}
- **A hypothesis is a set.** Bayesian generalization treats each candidate concept as a *set* of items, and
  asks which sets the data supports — the one shift that turns Bayes' rule into a model of generalization.
- **Generalization is a posterior-weighted vote:** $p(y \in C \mid X) = \sum_{h} \mathbf{1}[y \in h] \cdot
  p(h \mid X)$ — predict a novel item by the total posterior belief on the hypotheses that contain it.
- **The size principle** (from strong sampling, likelihood $(1/|h|)^n$) makes *smaller, tighter* hypotheses win
  among those consistent with the data, exponentially fast in the number of examples — turning a few examples
  into a confident rule.
- **One framework, many domains:** the same equation handled golden stickers, the number game, and
  continuous intervals — only $\mathcal{H}$ changed. For continuous concepts, Shepard's exponential law of
  generalization even **emerges from the model** rather than being assumed.
- **The prior is unavoidable.** No Free Lunch: a learner that considers every hypothesis learns nothing. The
  hypothesis space and its prior are the inductive bias that makes generalization possible.
- **In code:** for a finite $\mathcal{H}$, you don't need sampling — enumerate every hypothesis, score it by
  prior × likelihood in log space, normalize, and vote. That's *exact* for a finite list, and arbitrarily
  accurate on a fine grid for continuous concepts — and simple either way.
{{% /notice %}}

### A loose end, and the next chapter

We kept choosing $\mathcal{H}$ and $p(h)$ by hand. No Free Lunch says we *must* commit to some prior — but it
doesn't say the commitment has to be hand-picked forever. Could a learner **learn its prior** from experience —
watching many related concepts and inferring what kinds of rules tend to hold? That is the idea of
**hierarchical Bayes**, where the prior itself has a prior, and it's where this thread continues later in the
tutorial.

A nearer step: every hypothesis space in this chapter was a flat *list* of rules. But real models often have
*structure* — variables that depend on each other in specific ways. Writing that structure down as a graph, and
reading independence and causation off it, is the subject of the next chapters on **Bayesian networks**.

{{% notice style="note" title="🧠 Do large models generalize like this?" %}}
The Bayesian generalization story of this chapter — a prior over hypotheses, the **size principle**, similarity — is now a live testbed for frontier models: do today's large (and multimodal) models generalize the way people do? Strikingly often, yes. Large language models reproduce human **similarity judgments across six perceptual modalities** (color, pitch, taste, …), recovering structures like the color wheel mostly *from language alone* (Marjieh, Sucholutsky, van Rijn, Jacoby & Griffiths 2024), and a meta-learned network can be pushed to human-like **systematic, compositional** generalization (Lake & Baroni 2023). But matching human *outputs* is not the same as sharing the human *mechanism* — the same caution this book raises about machine Theory of Mind in [Chapter 25](../25_modern_rl_world_models/).
{{% /notice %}}

### Practice

{{% notice style="info" title="Try it yourself" %}}
1. **A second example.** In the §5/§6 number game, you saw what happens when you observe 10, then 10, 20, and
   30. Predict by hand (then check in code) what the strong-sampling posterior over the seven rules becomes if
   instead the observations are $\{2, 4, 8\}$. Which rule should win, and why does the size principle pick it?
   (Hint: which named rules contain all three, and which of those is the smallest?)
2. **Tighter or looser?** In the 1-D interval learner, change the exponential-prior rate from $\lambda = 0.5$
   to $\lambda = 2.0$. Before running it: will generalization get *tighter* or *broader*? Run it and check
   against the relation "mean interval length $= 1/\lambda$."
3. **Weak vs. strong, again.** Re-run the number game with the examples $\{2, 4, 6, 8\}$ and the two hypotheses
   "even numbers" and "powers of 2." Which way does strong sampling lean, and does weak sampling have any
   opinion at all? (Hint: compute each hypothesis's size first.)
4. **Break it yourself.** Following the No Free Lunch section, build the full 63-rule space but keep **strong**
   sampling instead of weak. Does generalization still collapse to a flat line? Explain what the size principle
   is (and isn't) able to do once *every* subset is on the table. *(Hint: with strong sampling, which single
   consistent rule is the *smallest*, and what does the size principle do to it?)*
{{% /notice %}}

---

## References

- Attneave, F. (1950). Dimensions of similarity. *American Journal of Psychology*, 63(4), 516–556.
  [https://doi.org/10.2307/1418869](https://doi.org/10.2307/1418869) — an early report of exponential-like
  similarity/generalization falloff, predating Shepard's normative account.
- Lake, B. M., & Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural
  network. *Nature*, 623(7985), 115–121.
  [https://doi.org/10.1038/s41586-023-06668-3](https://doi.org/10.1038/s41586-023-06668-3) — a meta-learned
  network reaching human-like compositional generalization.
- Marjieh, R., Sucholutsky, I., van Rijn, P., Jacoby, N., & Griffiths, T. L. (2024). Large language models
  predict human sensory judgments across six modalities. *Scientific Reports*, 14, 21445.
  [https://doi.org/10.1038/s41598-024-72071-1](https://doi.org/10.1038/s41598-024-72071-1) — LLMs reproduce
  human similarity structure across perceptual modalities, largely from language.
- Shepard, R. N. (1987). Toward a universal law of generalization for psychological science. *Science*,
  237(4820), 1317–1323. [https://doi.org/10.1126/science.3629243](https://doi.org/10.1126/science.3629243) —
  states the universal exponential law *and* gives the first normative (Bayesian) argument for why it holds.
- Tenenbaum, J. B., & Griffiths, T. L. (2001). Generalization, similarity, and Bayesian inference.
  *Behavioral and Brain Sciences*, 24(4), 629–640.
  [https://doi.org/10.1017/S0140525X01000061](https://doi.org/10.1017/S0140525X01000061) — the
  hypothesis-space / consequential-set framework this chapter develops, unifying Shepard's law with similarity-
  and rule-based generalization.
- Tenenbaum, J. B. (1999). *A Bayesian framework for concept learning* (PhD thesis, MIT).
  [https://dspace.mit.edu/handle/1721.1/16714](https://dspace.mit.edu/handle/1721.1/16714) — the number game
  and the rectangle game, with the strong-sampling size principle.
- Wolpert, D. H. (1996). The lack of a priori distinctions between learning algorithms. *Neural Computation*,
  8(7), 1341–1390. [https://doi.org/10.1162/neco.1996.8.7.1341](https://doi.org/10.1162/neco.1996.8.7.1341) —
  the "No Free Lunch" result behind §9.

---

Special thanks to [JPCCA](https://jpcca.org/) for their generous support of this tutorial series.
