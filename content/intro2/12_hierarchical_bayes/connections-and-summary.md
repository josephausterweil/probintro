+++
date = "2026-06-01"
title = "Connections & Summary"
weight = 4
+++

## The connection to No Free Lunch

Step back to where [Chapter 7](../../07_generalization/no-free-lunch-and-summary/) left us. The **No Free Lunch
(NFL)** theorem proved that a learner **must** bring a prior — inductive bias is not optional, because a learner
that entertains every hypothesis equally can't generalize at all. That sounds like a life sentence: someone has
to *hand* the learner its bias.

Hierarchical Bayes is the escape hatch. The prior is still required — NFL is not repealed — but the learner can
**acquire** it from data *about related problems* instead of being born with it. Each student is a small
learning problem; the population level is where the learner discovers "students tend to be around 60%
tonkatsu," and that discovered bias is exactly what lets it make a sane guess for a brand-new student it has
barely any data on. **The hierarchy is where inductive bias comes from when you don't want to hand-pick it.**

{{% notice style="info" title="Overhypotheses" %}}
This idea has a name: an **overhypothesis** — a second-level hypothesis *about what the first-level hypotheses
tend to look like*. The term is **Nelson Goodman's** (*Fact, Fiction, and Forecast*, 1955), coined as part of
his "new riddle of induction"; **Kemp, Perfors & Tenenbaum (2007)** later gave it a precise hierarchical-Bayes
formalization — the one we've been using. A child learning that "objects of the same kind tend to share a
shape" (the *shape bias*) has acquired an overhypothesis: it doesn't tell you any particular object's shape,
but it tells you *what kind of rule* to expect, so the next object can be learned from a single example. That's
the same move as inferring $(a, b)$: learning the *shape of the prior* from many problems so each new problem
needs almost no data. (Name-drop only — we won't derive it.)
{{% /notice %}}

---

## Connections to the rest of the tutorial

Three quick threads back into material you've seen (or will), not re-derived:

- **The DPMM is a hierarchical model.** The Dirichlet Process Mixture from [Chapter 6](../../06_dpmm/) has a
  hyperparameter — the DP *concentration* $\alpha$ — sitting above the cluster structure, governing how many
  clusters tend to appear. That's the same top-level "prior over the prior" shape you just built, applied to
  *how many groups exist* rather than *each group's rate*.
- **This diagram is a Bayes net.** The $(a, b) \to \theta_i \to k_i$ picture in §4 is a directed graphical
  model — a **Bayes net** with a repeated *plate* of students (a plate is just shorthand for "repeat this
  sub-graph once per student"). The forthcoming Bayes-net chapters of this tutorial will make that language
  precise; everything here is consistent with it.
- **Shrinkage is the Chapter 4 compromise, scaled up.** The posterior mean $(a + k)/(a + b + n)$ is the exact
  analogue of Chapter 4's precision-weighted blend of prior and data — one parameter then, a whole population
  of them now, tied together by a shared, learned prior.

---

## Summary

{{% notice style="success" title="Key takeaways" %}}
- **Two extremes both fail.** *No pooling* (estimate each unit alone) gives absurd, overconfident estimates
  from little data; *complete pooling* (one shared estimate) erases real differences. **Partial pooling** is
  the principled middle.
- **Shrinkage.** A hierarchical model pulls each estimate toward the shared population — **hardest for units
  with little data, barely at all for units with lots.** Estimates "borrow strength" from one another
  automatically.
- **The Beta-Binomial.** $\text{Beta}(a, b)$ is a prior over a rate ($a, b$ = soft counts of prior
  successes/failures, mean $a/(a+b)$); observing $k$ of $n$ updates it to $\text{Beta}(a + k, b + n - k)$, with
  posterior mean $(a + k)/(a + b + n)$ — shrinkage, in one formula.
- **Learning the prior is just inference, one level up.** Put a hyperprior on $(a, b)$, observe the units,
  weight candidate populations by likelihood (importance sampling, unchanged from Chapter 5). The prior has its
  own prior — coherent, not infinite regress.
- **This is the answer to No Free Lunch.** NFL says a learner needs inductive bias; the hierarchy is where a
  learner **acquires** that bias from related problems instead of being handed it.
{{% /notice %}}

### Practice

{{% notice style="info" title="Try it yourself" %}}
1. **Predict the shift.** Before running anything: a new student, **Greta**, brings 4 bentos, all tonkatsu
   (4/4). Under the population prior $\text{Beta}(6, 4)$, what is her partial-pooling estimate $(a+k)/(a+b+n)$?
   Is her shrinkage more or less than Emi's (2/2)? Check in code.
2. **Stronger or weaker prior.** Re-run the shrinkage cell with $\text{Beta}(60, 40)$ instead of
   $\text{Beta}(6, 4)$ — same mean (0.6) but ten times the strength. Before running: will the data-light
   students be pulled *more* or *less* toward 0.6? Explain using $a + b$ as a prior sample size.
3. **Complete pooling as a limit.** Show (by trying a few values) that as $a + b \to \infty$ with the mean
   fixed, every student's estimate collapses to the population mean — i.e. an infinitely strong prior *is*
   complete pooling. What does $a + b \to 0$ give you instead?
4. **Inferred vs. assumed.** Re-run the §5 importance-sampling cell a few times with different `PRNGKey`
   seeds. How much does the inferred population rate wobble? Increase `N` from 20000 to 200000 — does it
   steady? Relate this to the "importance sampling is noisy" note.
{{% /notice %}}

---

## References

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).
  *Bayesian Data Analysis* (3rd ed.). CRC Press — the standard treatment of hierarchical models, partial
  pooling, and shrinkage (Chapter 5).
- Goodman, N. (1955). *Fact, Fiction, and Forecast*. Harvard University Press — coins the term
  **overhypothesis** as part of the "new riddle of induction."
- Kemp, C., Perfors, A., & Tenenbaum, J. B. (2007). Learning overhypotheses with hierarchical Bayesian models.
  *Developmental Science*, 10(3), 307–321.
  [https://doi.org/10.1111/j.1467-7687.2007.00585.x](https://doi.org/10.1111/j.1467-7687.2007.00585.x) — the
  hierarchical-Bayes *formalization* of Goodman's overhypotheses; hierarchies as where inductive bias (the shape
  bias, object-vs-substance) is *acquired*.

---

Special thanks to [JPCCA](https://jpcca.org/) for their generous support of this tutorial series.
