+++
date = "2026-06-24"
title = "Modern RL: Preferences, World Models, and Machine Minds"
weight = 25
+++

## The Same Inversion, at Frontier Scale

The last two chapters built one machine and pointed it at smaller and smaller hidden things: a goal, a belief, a reward, another mind. All of it was hand-sized — three grid cells, two doors. This chapter is the same machine at the scale of the systems making headlines: aligning large language models, planning by *imagining*, and the genuinely unsettled question of whether those models have minds to read at all.

> **Alyssa:** "So the toy inversions weren't toys. They were the *mechanism* — and now we just run it bigger."
>
> **Jamal:** "Bigger, and with a twist. Up to now we watched whole demonstrations. The frontier mostly learns from something cheaper: a human glancing at two answers and saying *that one's better.*"

That twist — learning a reward from **preferences** instead of demonstrations — is exactly how today's models are aligned, and it is inverse RL wearing a new coat. We'll build it, then survey two more frontier ideas the same machine illuminates (amortized mind-reading and world models), and end on the honest state of the LLM Theory-of-Mind debate.

---

## RLHF and DPO Are Inverse RL

You cannot write down a reward function for "be helpful and harmless." So modern alignment does not try. **RLHF** — reinforcement learning from human feedback — instead (1) shows people **pairs** of model outputs and records **which they prefer**, (2) fits a **reward model** to those preferences, then (3) optimizes the policy against that learned reward. Step (2) is the whole of inverse RL: *recover a hidden reward from observed human choices.* **DPO** (direct preference optimization) folds (2) and (3) together — it shows the optimal RLHF policy implies an *implicit* reward, so you can optimize the policy directly on preferences — but the inference problem underneath is identical.

The choice model is **Bradley–Terry**, the same logistic form we have met as the softmax: a human prefers item $i$ over $j$ with probability

$$P(i \succ j) \;=\; \sigma\!\big(r_i - r_j\big), \qquad \sigma(x)=\frac{1}{1+e^{-x}}.$$

This is just a **pairwise softmax** — $P(i \succ j) = \frac{e^{r_i}}{e^{r_i}+e^{r_j}} = \sigma(r_i - r_j)$ — the two-item version of [Chapter 23](../23_inverse_rl_goal_inference/)'s policy. Better items win *more often*, not always — noisy-rational raters, exactly like the noisy-rational agents of Chapter 23. And fitting the latent rewards $r$ from observed preferences is **inverse RL by the same Bayes' rule as goal inference**:

$$P(\text{reward} \mid \text{preferences}) \;\propto\; P(\text{preferences} \mid \text{reward})\;P(\text{reward}),$$

where the likelihood is the Bradley–Terry choice model. We recover a *reward* instead of a *goal*, but the machine is identical. We can do the literal reward-modeling step in GenJAX: latent rewards with a prior, each observed comparison a `flip` whose probability is $\sigma(r_i - r_j)$, then condition on the humans' choices and read off the posterior-mean reward. (We recover that posterior by **importance sampling** — draw many reward vectors from the prior, weight each by how well it explains the observed preferences, and take the weighted average; a standard tool when the posterior has no closed form.)

{{% expand "Show the Bradley-Terry reward model + recovery (sampling-importance-resampling)" %}}
<!-- validate: skip-output -->
```python
import jax, jax.numpy as jnp
from jax import random, vmap
from genjax import gen, normal, flip, ChoiceMap

K = 3                                              # three candidate answers
ITEMS = ["A", "B", "C"]
TRUE_R = jnp.array([2.0, 0.5, -1.0])               # hidden quality (A > B > C)
PAIRS = jnp.array([[0, 1], [0, 2], [1, 2]])        # the three head-to-head comparisons

@gen
def pref_model(pairs):                             # Bradley-Terry: P(i > j) = sigmoid(r_i - r_j)
    r = jnp.array([normal(0.0, 2.0) @ f"r_{k}" for k in range(K)])     # latent reward, normal prior
    for n in range(pairs.shape[0]):
        i, j = pairs[n, 0], pairs[n, 1]
        flip(jax.nn.sigmoid(r[i] - r[j])) @ f"pref_{n}"                 # True => i preferred over j
    return r

def make_dataset(n_each=30, key=random.PRNGKey(0)):                     # noisy human prefs from TRUE_R
    pairs = jnp.repeat(PAIRS, n_each, axis=0)
    pi = jax.nn.sigmoid(TRUE_R[pairs[:, 0]] - TRUE_R[pairs[:, 1]])
    return pairs, random.bernoulli(key, pi)

def recover_reward(pairs, prefs, key=random.PRNGKey(1), n_particles=40000):
    cm = ChoiceMap.d({f"pref_{n}": bool(prefs[n]) for n in range(prefs.shape[0])})
    def one(k):
        tr, w = pref_model.importance(k, cm, (pairs,))                  # condition on the preferences
        ch = tr.get_choices()
        return jnp.array([ch[f"r_{i}"] for i in range(K)]), w
    rs, ws = vmap(one)(random.split(key, n_particles))                 # sampling-importance-resampling
    return (jax.nn.softmax(ws)[:, None] * rs).sum(0)                    # posterior-mean reward

def center(r):                                     # remove the unidentifiable additive shift
    return r - jnp.mean(r)
```
{{% /expand %}}

```python
pairs, prefs = make_dataset()
r_hat = recover_reward(pairs, prefs)
print(f"{prefs.shape[0]} human preferences from a hidden quality A=2.0, B=0.5, C=-1.0 (A>B>C).")
print("recovered reward from the preferences alone (mean-centered):")
print("  item   true   recovered")
for k in range(K):
    print(f"   {ITEMS[k]}    {float(center(TRUE_R)[k]):+5.2f}    {float(center(r_hat)[k]):+5.2f}")
order = [ITEMS[i] for i in jnp.argsort(-r_hat)]
print(f"recovered ranking: {' > '.join(order)}")
```

**Output:**
```
90 human preferences from a hidden quality A=2.0, B=0.5, C=-1.0 (A>B>C).
recovered reward from the preferences alone (mean-centered):
  item   true   recovered
   A    +1.50    +1.36
   B    +0.00    -0.00
   C    -1.50    -1.36
recovered ranking: A > B > C
```

From nothing but $90$ "this one's better" judgments, we recovered the hidden quality of all three answers — correct ranking, roughly correct spacing. Note the recovered numbers ($+1.36, 0, -1.36$) are a little *compressed* toward zero versus the truth ($+1.5, 0, -1.5$): finite noisy data and the normal prior pull the estimate gently inward. And the reward is identifiable **only up to an additive constant**: every preference depends on the *difference* $r_i - r_j$, so adding the same constant to all rewards leaves every difference — and thus every preference — unchanged. That is why we mean-centered both columns before comparing them. That residual ambiguity is the same ill-posedness from Chapter 23, alive and well at the heart of how every aligned model is trained. Scale this up — millions of preferences, a transformer for the reward model — and you have RLHF.

![A timeline of inverse-RL methods leading to the frontier: MaxEnt IRL, then GAIL, then AIRL, then RLHF and DPO, each annotated with its one-line contribution; RLHF/DPO are highlighted as preference-based inverse RL.](../../images/intro2/irl-methods-timeline.png)

{{% notice style="note" title="Reward hacking is the positive cycle at scale" %}}
[Chapter 22](../22_q_learning/) showed an agent **farm** a badly-shaped reward — pacing a praise-giving path forever instead of finishing. A *learned* reward model has the same failure mode, now at frontier scale: the policy discovers inputs that score high on the reward model without being genuinely good (sycophancy, length-gaming, confident nonsense). The reward model is an *approximation* of human values, and optimizing hard against an approximation finds its cracks. This is **reward hacking** — the same shape as the Chapter-22 positive cycle (optimize a proxy, discover its loopholes), though the cure differs: we can't reshape true human values, so the effort goes into making the *learned* reward robust to adversarial inputs.
{{% /notice %}}

---

## Two Ways to Read a Mind

In Chapters 23–24 we read minds the *Bayesian* way: enumerate the hypotheses, score each with a forward planner, normalize. It is interpretable and sample-efficient — but it requires running the planner at inference time, once per hypothesis, which does not scale to rich worlds.

The frontier's alternative is **amortized** inference: pay the cost *once*, up front, by training a neural network to map behavior straight to the answer. **ToMnet** (Rabinowitz et al. 2018) does exactly this — a "Theory-of-Mind network" watches many agents act and *learns* to predict their behavior and their (possibly false) beliefs, in a single forward pass, with no explicit planner inside. It is the **learned, scalable cousin** of the explicit Bayesian inversion you built: same inverse problem — recover an agent's mental state from behavior — traded from *exact-but-slow* to *learned-but-opaque*.

![Two panels. Left, labeled Bayesian inverse planning: an explicit planner inverted by Bayes, interpretable and sample-efficient, hand-built. Right, labeled Machine ToM (ToMnet, 2018): a neural net trained on many agents, learned and amortized and scalable but less interpretable. Both map behavior to a goal or belief.](../../images/intro2/amortized-vs-bayesian-tom.png)

This is the same exact-versus-amortized tradeoff you will see everywhere in modern probabilistic ML: enumeration and importance sampling (what we ran above) are exact but slow; a trained network is fast but inherits whatever its training distribution taught it.

---

## World Models: Planning by Imagining

[Chapter 22](../22_q_learning/) ended on **simulation-based RL** — learn a model of the world, then *plan by simulating rollouts* inside it (Dyna, then Monte-Carlo Tree Search). That is the engine of the most capable agents today, and the frontier pushes it in one direction: make the learned model **abstract**.

- **MuZero** (Schrittwieser et al. 2020) learns a model not of the literal environment but of a **latent state** that is *only good enough to predict reward, value, and policy* — then runs MCTS in that latent space. It mastered Go, chess, and Atari **without being told the rules**, learning a world model purely from interaction.
- **Dreamer** (Hafner et al. 2020, 2023) learns a latent world model and then trains its policy entirely *inside the model* — "**learning behaviors by latent imagination**," planning in a dream rather than in the costly real world.

Both are the Chapter-22 idea — *know a model, simulate to plan* — with the model **learned and compressed**. And the connection back to this unit is direct: a world model that must track a hidden environment state from partial observations is maintaining a **belief**, exactly the POMDP machinery of [Chapter 24](../24_pomdps_belief_inference/), now learned by a network rather than written by hand — which means the belief-updating and decision tools from that chapter (filtering the hidden state, α-vectors, posterior sampling) apply, in principle, to planning *inside* the learned model — Dreamer's latent state, for instance, is refreshed from each new observation, a learned belief filter.

---

## Do LLMs Have a Theory of Mind?

The unit opened by inferring a goal from a few footsteps. It ends on the question that inference makes unavoidable: when a large language model passes the tasks we use to *measure* Theory of Mind, does it actually *have* one? Here the honest answer is **contested**, and this book takes the **skeptical** side — not because the capabilities aren't striking, but because the evidence does not yet license the strong claim.

The case *for* is real: GPT-4-class models pass a large fraction of classic **false-belief** vignettes, often at adult or near-adult level, and clear irony, hinting, and faux-pas batteries (Kosinski 2024; Strachan et al. 2024). Taken at face value, that looks like mentalizing.

But face value is exactly what is in doubt:

- **Minimal, belief-preserving perturbations break them** (Ullman 2023). Take a standard false-belief story — Anne hides a toy in a box, leaves, the toy is moved — which a capable model resolves correctly by reporting Anne's *false* belief. Now make the box **transparent**, so Anne could plainly see the toy move: a mind that tracks beliefs handles this trivially (Anne now knows), yet models often *still* report the old false belief. Failing on a change that is irrelevant to the belief itself points to surface pattern-matching, not mental-state inference — a robust Theory of Mind would not break on a cosmetic tweak.
- **The wins may be artifacts.** Apparent faux-pas competence can reflect response bias rather than inference; and because these vignettes and their templates are all over the training data, high scores may reflect **memorization / contamination** rather than reasoning (Pang 2025). Morgan's Canon applies — do not attribute a higher faculty when a lower one suffices — which is a call for *conservatism*: the current evidence does not *require* genuine mental-state inference, not that it rules such inference out.

![A two-column figure. Left, capability claims: Kosinski 2024 reports GPT-4 near a six-year-old level; Strachan 2024 at or above human on false belief, irony, hinting. Right, skeptical rebuttals: Ullman 2023 shows trivial perturbations flip success to failure; Pang 2025 cites Morgan's Canon and training-data contamination. A banner reads: behavioral pass does not equal mechanism.](../../images/intro2/llm-tom-debate.png)

The throughline is one sentence: **a behavioral pass is not the mechanism.** Passing the test that was designed to *detect* a capability in humans is not the same as having the capability, when the system could be matching patterns instead. The defensible 2026 position is the cautious one: *current evidence does not show that LLMs have a human-like Theory of Mind* — impressive, ToM-shaped pattern-matching, not demonstrated mental-state representation. That is not a permanent verdict; it is a statement about what today's evidence supports.

---

{{% notice style="success" title="What you can do now" %}}
You can see the modern frontier as the inverse-RL machine at scale. You built the **RLHF / DPO** reward-modeling step in GenJAX: a **Bradley–Terry** preference model $P(i\succ j)=\sigma(r_i-r_j)$, conditioned on $90$ human "this is better" judgments, recovers the hidden quality (A > B > C, up to the usual additive constant) — *preference-based inverse RL*, the engine that aligns today's models — and you know its failure mode, **reward hacking**, is Chapter 22's positive cycle at scale. You can contrast the two ways to read a mind: the **exact Bayesian** inversion you coded versus **amortized** inference (**ToMnet**), the learned-but-opaque cousin. You can place modern **world models** — **MuZero**, **Dreamer** ("latent imagination") — as Chapter 22's simulation-based RL with a *learned, compressed* model that tracks a belief. And you can argue the **LLM Theory-of-Mind** debate on the evidence: behavioral passes, belief-preserving-perturbation failures, contamination — and the cautious bottom line that *a behavioral pass is not the mechanism.*

This closes the agency-and-minds arc: from one decision ([Chapter 20](../20_statistical_decision_theory/)), to planning a known world, to learning an unknown one, to **inverting** behavior to read goals, beliefs, rewards, and — at the edge of what we can verify — minds.

*Glossary:* [RLHF](../../glossary/#rlhf-), [DPO](../../glossary/#dpo-), [Bradley–Terry model](../../glossary/#bradley-terry-model-), [reward model](../../glossary/#reward-model-), [reward hacking](../../glossary/#reward-hacking-), [amortized inference](../../glossary/#amortized-inference-), [ToMnet](../../glossary/#tomnet-), [world model](../../glossary/#world-model-), [MuZero](../../glossary/#muzero-).
{{% /notice %}}

---

## Exercises

{{% notice style="info" title="Try it yourself" %}}
1. **How many preferences is enough?** In the RLHF code, drop `n_each` from $30$ to $3$ (only $9$ comparisons) and rerun a few times with different `key`s in `make_dataset`. Does the recovered ranking stay correct? Does the *spacing* get noisier? Connect this to why frontier reward models need enormous preference datasets.
2. **The unidentifiable shift.** Add a constant (say $+5$) to `TRUE_R` and rerun. The recovered *centered* reward should be unchanged — verify it, and explain in one sentence why preferences can never reveal the absolute level of reward, only differences.
3. **Design a perturbation.** Pick a classic false-belief story and describe a *belief-preserving* change (like Ullman's transparent container) that a true Theory of Mind should be unaffected by. What would you conclude if a model's answer changed, and what if it didn't?
{{% /notice %}}

A companion notebook works through all of this interactively:

**📓 [Open in Colab: `25_modern_rl_world_models.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/25_modern_rl_world_models.ipynb)**

---

## References

- Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika, 39*(3/4), 324–345. <https://doi.org/10.2307/2334029>
- Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems (NeurIPS), 30*. <https://arxiv.org/abs/1706.03741>
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to control: Learning behaviors by latent imagination. *International Conference on Learning Representations (ICLR)*. <https://arxiv.org/abs/1912.01603>
- Kosinski, M. (2024). Evaluating large language models in theory of mind tasks. *Proceedings of the National Academy of Sciences, 121*(45), e2405460121. <https://doi.org/10.1073/pnas.2405460121>
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems (NeurIPS), 35*. <https://arxiv.org/abs/2203.02155>
- Pang et al. (2025). On evaluating theory of mind in large language models (Morgan's Canon; training-data contamination). *Proceedings of the National Academy of Sciences*. <https://doi.org/10.1073/pnas.2507080122>
- Rabinowitz, N. C., Perbet, F., Song, H. F., Zhang, C., Eslami, S. M. A., & Botvinick, M. (2018). Machine theory of mind. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, 4218–4227. <https://arxiv.org/abs/1802.07740>
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems (NeurIPS), 36*. <https://arxiv.org/abs/2305.18290>
- Schrittwieser, J., Antonoglou, I., Hubert, T., et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model. *Nature, 588*(7839), 604–609. <https://doi.org/10.1038/s41586-020-03051-4>
- Strachan, J. W. A., et al. (2024). Testing theory of mind in large language models and humans. *Nature Human Behaviour, 8*, 1285–1295. <https://doi.org/10.1038/s41562-024-01882-z>
- Ullman, T. (2023). Large language models fail on trivial alterations to theory-of-mind tasks. *arXiv:2302.08399*. <https://arxiv.org/abs/2302.08399>

---

Special thanks to [JPPCA](https://jpcca.org/) for their generous support of this tutorial series.
