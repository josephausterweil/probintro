+++
date = "2026-06-01"
title = "Hierarchical Bayes"
weight = 12
toc = true
+++



## Two extremes that both feel wrong

Chibany has been keeping a journal: for each student who brings them a bento, they record whether it was
**tonkatsu** or **hamburger**. After a while the journal looks like this — each student with a tonkatsu count
$k_i$ out of their total bento count $n_i$:

| Student | Tonkatsu $k_i$ | Total $n_i$ | Raw fraction $k_i / n_i$ |
|---|---:|---:|---:|
| Alyssa | 70 | 100 | 0.70 |
| Ben | 28 | 40 | 0.70 |
| Carmen | 6 | 10 | 0.60 |
| Diego | 3 | 5 | 0.60 |
| Emi | 2 | 2 | **1.00** |
| Farid | 0 | 1 | **0.00** |

Chibany wants, for each student, a believable estimate of $\theta_i$ — that student's underlying probability of
bringing tonkatsu. Two obvious strategies both fail:

<div style="display:flex; flex-wrap:wrap; gap:1.5rem; margin:1rem 0;">
<div style="flex:1; min-width:260px;">

**No pooling — estimate each student alone.** Just use the raw fraction $k_i / n_i$. For Alyssa (70/100) that's
fine. But **Emi brought 2 bentos, both tonkatsu**, so this says $\theta_{\text{Emi}} = 1.00$ — Emi *always*
brings tonkatsu, with certainty, on the strength of two data points. **Farid (0/1)** is even worse the other
way: one bento, a hamburger, and we declare him a 0%-tonkatsu person who will *never* bring tonkatsu. Nobody
believes either of these.

</div>
<div style="flex:1; min-width:260px;">

**Complete pooling — one shared rate for everyone.** Lump all the bentos together: $109$ tonkatsu out of $158$,
so $\theta = 109/158 \approx 0.69$ for *everyone* (really $0.690$, dominated by the heavy bringers Alyssa and
Ben). This fixes the Emi/Farid absurdity, but now it **throws away the real differences** between students —
and we have good reason to think students differ.

</div>
</div>

Neither extreme is right. The fix is the principled middle, **partial pooling**: estimate each $\theta_i$
*using that student's own data, pulled toward what the other students do.* A student with lots of data stays
near their own fraction; a student with almost no data leans heavily on the group. This is exactly the
**"prior vs. data compromise"** you met as precision-weighting in [Chapter 4](../../04_bayesian_learning/) — only
now the *prior* is the population of other students, and it is **learned**, not assumed.

{{% notice style="warning" title="The pathology, concretely" %}}
The danger of no-pooling is loudest for **data-light** students: one or two bentos give raw fractions of 0.00
or 1.00 — maximally confident estimates from minimal evidence. Watch what partial pooling does to Emi and
Farid specifically; they are the whole point.
{{% /notice %}}

Here is the no-pooling pathology in code — raw fractions with wildly different amounts of data behind them:

```python
import jax.numpy as jnp

# (student, tonkatsu count k_i, total bentos n_i)
names = ["Alyssa", "Ben", "Carmen", "Diego", "Emi", "Farid"]
k = jnp.array([70, 28, 6, 3, 2, 0])     # tonkatsu counts
n = jnp.array([100, 40, 10, 5, 2, 1])   # total bentos

raw_fraction = k / n
for name, kf, nf, r in zip(names, k, n, raw_fraction):
    print(f"  {name:7s} {int(kf):>2d}/{int(nf):<3d} -> raw estimate {float(r):.2f}")
```

**Output:**
```
  Alyssa  70/100 -> raw estimate 0.70
  Ben     28/40  -> raw estimate 0.70
  Carmen   6/10  -> raw estimate 0.60
  Diego    3/5   -> raw estimate 0.60
  Emi      2/2   -> raw estimate 1.00
  Farid    0/1   -> raw estimate 0.00
```

Emi at 1.00 and Farid at 0.00 are the tell: no-pooling lets one or two bentos masquerade as certainty — in
*either* direction.

---

