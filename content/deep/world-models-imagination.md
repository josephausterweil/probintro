+++
date = "2026-07-04"
title = "World Models & Imagination"
weight = 50
tracks = ["beginner", "cogsci"]
+++

## The Trainee Rehearses Lunch

The robot mascot trainee has developed a strange new habit, and Chibany has been
watching it for three days now. At the doorway of Cafeteria A, just before the
lunch rush, the trainee **stops**. For half a second it goes still — no whir, no
step — and then it walks in and threads the crowd to the bento kiosk without a
single wrong turn. Chibany, who navigates the same crowd by bumping into people
and apologizing, finds this deeply suspicious. What is the trainee *doing* in
that half-second?

> **Jamal:** "It's running the walk in its head first. It has a little model of
> the cafeteria — where the tables are, what happens if it steps left, steps
> right — and before it moves for real, it plays a few possible walks forward
> and keeps the best one."
>
> **Alyssa:** "Which is a genuinely new thing for this Part. Back in the
> [MDP chapter](../../decisions/mdps/) we *handed* the agent the world — every
> transition probability, every reward, printed in a matrix. It planned by
> reading numbers it already had. Nobody handed the trainee a matrix of the
> cafeteria. It had to **learn** what happens next from bumping around, and
> *then* plan inside what it learned."

That learned "what happens next" has a name. A **world model** is a learned
generative model of an environment's dynamics: give it a state and an action,
and it predicts the next state (and the reward that comes with it). It is the
transition function $T(s' \mid s, a)$ from the [MDP chapter](../../decisions/mdps/) —
except that chapter *knew* $T$, and here we have to fit it from experience, the
same way the [kiosk](../from-rules-to-weights/) fit its photo classifier from
labeled pictures. Once you have such a model, planning becomes **imagination**:
roll the model forward in your head, score the futures it predicts, and pick the
action sequence that looks best — all before touching the real world.

This chapter is the learning-side twin of
[Modern RL & World Models](../../decisions/modern-rl-world-models/), which met the
same frontier from the inference side (recovering rewards from preferences,
reading minds). Here we stay on the *learning* side: fit a model of the world,
then plan by dreaming inside it.

{{< lecture week=11 label="World models" >}}

---

## Two Ways to Get Good at Acting

The [Q-learning chapter](../../decisions/q-learning/) drew the line this chapter
lives on, so let us re-anchor it. Faced with a world whose rules you do not know,
there are two families of strategy.

- **Model-free.** Never build a model at all. Just act, and **cache a value** for
  each situation — a running estimate of how good it is. Q-learning is the
  archetype: it stores $Q(s,a)$ and nudges it toward reward, and it never once
  represents *how the world works*. It learns *what to do* without ever learning
  *what happens next*.
- **Model-based.** Learn the dynamics — a **world model** — and then **plan** by
  simulating it. **Dyna** (Sutton, 1991), which the Q-learning chapter introduced
  in passing, is the clean example: count where your actions actually led, build
  a transition model $\hat T$ from those counts, then run value iteration on
  $\hat T$ as if it were the real thing. This chapter takes that idea seriously.

The trade between them is a familiar shape — the **bias–variance** tension of
[Part VII](../../complexity/bias-variance/) wearing new clothes. A world model is
enormously **sample-efficient**: every real step teaches it a general fact about
the dynamics ("stepping right raises my position"), and it can then *imagine*
thousands of futures for free, wringing many decisions out of little experience.
That is the payoff. The cost is that the model is only ever an *approximation*,
and planning **compounds** its errors: to imagine six steps ahead you feed the
model's own guess back into itself six times, so a small per-step error snowballs
into a large per-plan one. Model-free learning has the opposite profile — it
needs a mountain of real trials (no imagination to shortcut them), but it never
trusts a wrong model, because it has no model to be wrong.

You can watch the sample-efficiency half of that trade directly. The
Dyna-vs-Q-learning widget from the Q-learning chapter runs both agents on the
*same* stream of experience from Chibany's wellbeing MDP; the only difference is
that Dyna spends a few **planning** steps imagining with its learned model after
each real step. Drag the planning steps up and Dyna reaches the answer in a
fraction of the real experience Q-learning needs:

{{< widget src="dyna-vs-qlearning" height="560" title="Q-learning (model-free) vs Dyna (model-based) on the Chibany MDP: both see the same experience, but Dyna plans with a learned model and converges in far fewer real steps" >}}

---

## A World Model You Can Watch Learn

Enough words — let us fit one. We will keep it genuinely small and honest: a
**toy**, labeled as such, so that every moving part is visible. The world is a
seven-cell corridor. The trainee starts at the door (cell $0$) and wants the bento
counter at the far end (cell $6$); it can step **left** or **right**, and the walls
clip it if it tries to leave. The trainee does *not* know these rules — it will
learn them, then plan a route to the counter entirely in imagination.

First the setup: the real dynamics (`true_step`, hidden from the learner) and our
imports.

```python
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, lax
import itertools

key = jr.PRNGKey(0)

N = 7                                       # a 7-cell corridor: cells 0 .. 6
GOAL = N - 1                                # the counter the trainee wants to reach
ACTIONS = jnp.array([-1.0, 1.0])            # action 0 = step left, action 1 = step right

def true_step(pos, a):                      # the REAL dynamics -- hidden from the learner
    return jnp.clip(pos + a, 0.0, float(N - 1))

print("corridor of", N, "cells; door at 0, counter (goal) at cell", GOAL)
```

**Output:**
```
corridor of 7 cells; door at 0, counter (goal) at cell 6
```

**Collect experience.** Like the trainee bumping around, we gather transitions by
following a **random policy** — flip a coin, step left or right, record where we
were, what we did, and where we ended up. This is the raw material a world model
is fit to.

```python
def collect(key, n_steps=800):
    def walk(carry, _):
        pos, key = carry
        key, ka = jr.split(key)
        a = ACTIONS[jr.bernoulli(ka).astype(jnp.int32)]     # random policy: left/right 50/50
        pos2 = true_step(pos, a)
        return (pos2, key), (pos, a, pos2)
    _, (P, A, P2) = lax.scan(walk, (jnp.array(0.0), key), None, length=n_steps)
    return P, A, P2

pos, act, nxt = collect(key)
print("collected", pos.shape[0], "transitions by wandering at random")
print("first 5 (pos, action -> next):")
for i in range(5):
    print(f"  cell {float(pos[i]):.0f}, step {float(act[i]):+.0f}  ->  cell {float(nxt[i]):.0f}")
```

**Output:**
```
collected 800 transitions by wandering at random
first 5 (pos, action -> next):
  cell 0, step +1  ->  cell 1
  cell 1, step +1  ->  cell 2
  cell 2, step +1  ->  cell 3
  cell 3, step +1  ->  cell 4
  cell 4, step +1  ->  cell 5
```

**Fit the model.** Now the world model itself — a tiny **one-hidden-layer
network** predicting the next position from the current position and action. It
is exactly the training pattern from
[From Rules to Weights](../from-rules-to-weights/): features $\mathbf{x}$, a ReLU
hidden layer $g(A\mathbf{x}+\mathbf{c})$, a linear read-out, a squared-error loss
(which, recall, *is* a Gaussian negative log-likelihood), and `jax.grad` walking
the parameters downhill. Nothing new — we are just pointing that machine at a new
target: *predict the next state* instead of *classify a bento*.

```python
def predict(params, pos, a):                          # a tiny 1-hidden-layer dynamics model
    x = jnp.array([pos / GOAL, a])                    # features: normalized position, action
    h = jnp.maximum(0.0, params["A"] @ x + params["c"])   # learned features (ReLU)
    return jnp.dot(params["w"], h) + params["b"]       # predicted NEXT position, in [0,1] units

def loss(params):
    pred = vmap(lambda p, a: predict(params, p, a))(pos, act)
    return jnp.mean((pred - nxt / GOAL) ** 2)          # squared error = Gaussian NLL (Part VIII)

def typical_error(params):                             # root-mean-square error, back in CELLS
    return jnp.sqrt(loss(params)) * GOAL

k1, k2 = jr.split(jr.PRNGKey(1))
H = 8
init_params = {"A": 0.3 * jr.normal(k1, (H, 2)), "c": jnp.zeros(H),
               "w": 0.3 * jr.normal(k2, (H,)), "b": jnp.array(0.0)}

print(f"prediction error BEFORE fitting: {float(typical_error(init_params)):.2f} cells (random model)")
params = init_params
grad_loss = grad(loss)
for step in range(2000):
    g = grad_loss(params)
    params = jax.tree.map(lambda p, gp: p - 0.2 * gp, params, g)   # gradient descent
print(f"prediction error AFTER  fitting: {float(typical_error(params)):.2f} cells (learned model)")
```

**Output:**
```
prediction error BEFORE fitting: 3.53 cells (random model)
prediction error AFTER  fitting: 0.15 cells (learned model)
```

Before training, the random model's guesses are off by about three and a half
cells — worse than useless in a corridor only seven long. After two thousand
gradient steps, it predicts the next cell to within about a *sixth* of a cell.
(That residual is not a bug: it is the **walls**. Stepping left at cell $0$ leaves
you at cell $0$, not at cell $-1$, and a smooth little network can only approximate
that hard clip — an honest reminder that a learned model is never quite the world.)

**Plan by imagination.** Now the payoff. The trainee never touches the real
corridor to plan. It enumerates every six-step plan (each step left or right —
$2^6 = 64$ plans), **rolls the *learned* model forward** from the door for each one
— feeding the model's own prediction back in as the next input, snapping it to the
nearest cell — and keeps the plan whose *imagined* endpoint lands nearest the
goal. To make the stakes clear, we do this twice: once with the useless
**unfitted** model, once with the **fitted** one. Then, and only then, do we run
the chosen plan for real.

```python
HORIZON = 6
plans = jnp.array(list(itertools.product([0, 1], repeat=HORIZON)))   # every 6-step plan (64 of them)

def imagine(params, plan):                             # roll the LEARNED model forward from the door
    def step(pos, a_idx):
        nxt_hat = predict(params, pos, ACTIONS[a_idx]) * GOAL
        pos2 = jnp.clip(jnp.round(nxt_hat), 0.0, float(N - 1))    # snap prediction back onto the grid
        return pos2, pos2
    final, _ = lax.scan(step, jnp.array(0.0), plan)
    return final

def choose(params):
    finals = vmap(lambda pl: imagine(params, pl))(plans)
    i = jnp.argmax(-jnp.abs(finals - GOAL))            # pick the plan whose IMAGINED end is nearest goal
    return plans[i], finals[i]

def run_for_real(plan):
    def step(pos, a_idx):
        return true_step(pos, ACTIONS[a_idx]), None
    final, _ = lax.scan(step, jnp.array(0.0), plan)
    return final

plan_bad, imag_bad = choose(init_params)               # plan inside the UNFITTED model
plan_good, imag_good = choose(params)                  # plan inside the FITTED model
real_bad = run_for_real(plan_bad)
real_good = run_for_real(plan_good)

lbl = lambda pl: "".join("R" if int(a) else "L" for a in pl)
print(f"UNFITTED model -> plan {lbl(plan_bad)}, imagined end {float(imag_bad):.0f},"
      f" REAL end {float(real_bad):.0f}, goal reached: {bool(real_bad == GOAL)}")
print(f"FITTED   model -> plan {lbl(plan_good)}, imagined end {float(imag_good):.0f},"
      f" REAL end {float(real_good):.0f}, goal reached: {bool(real_good == GOAL)}")
```

**Output:**
```
UNFITTED model -> plan LLLLLL, imagined end 0, REAL end 0, goal reached: False
FITTED   model -> plan RRRRRR, imagined end 6, REAL end 6, goal reached: True
```

There it is, in miniature. The **same** planner, handed a **bad** model, imagines
that walking left is fine and marches straight into the wall; handed the **fitted**
model, it discovers — purely in its head, never having walked the corridor to plan
— that going right six times reaches the counter, and when run for real, it does.
Planning is only ever as good as the model you plan inside: the quality of the
*dream* decides the quality of the *act*. That single dependency is the whole
subject of this chapter, and its central danger.

The loop, drawn once:

```mermaid
graph LR
    E["real experience<br/>(state, action, next)"] --> M["fit world model<br/>predict next state"]
    M --> I["imagine: roll the<br/>model forward"]
    I --> P["score futures,<br/>pick best action"]
    P --> A["act in the<br/>real world"]
    A --> E
    classDef node fill:none,stroke:#9bbcff,stroke-width:2px,color:#fff
    class E,M,I,P,A node
    linkStyle default stroke:#9bbcff,stroke-width:2px,color:#fff
```

---

## The Modern Version: Dreaming in Latent Space

Our corridor model predicts *raw* positions. The frontier's big idea is to not
bother predicting the raw world at all — predict a compressed **latent state**
instead: a small learned summary of the situation, rich enough to plan with but
stripped of pixel-level detail nobody needs. Two systems define the landscape,
and both deserve to be described plainly, without hype.

- **MuZero** (Schrittwieser et al., 2020) learns a model of a *latent* state that
  is trained to be good for one thing only: predicting **reward, value, and the
  right move**. It never tries to reconstruct the actual board or screen — only
  to be *sufficient for planning* — and it plans by running Monte-Carlo Tree
  Search (the [MCTS](../../decisions/q-learning/) of the Q-learning chapter) inside
  that learned latent model. With nothing but this, it mastered Go, chess, and
  Atari **without being told the rules** of any of them.
- **Dreamer** (Hafner et al., 2020, 2023) learns a latent world model and then
  trains its policy *entirely inside the model* — "learning behaviors by latent
  imagination." Rather than take costly real actions to improve, it rolls
  thousands of imagined trajectories through its learned dynamics and improves the
  policy on those dreams, touching the real environment only to keep the model
  honest.

Both are exactly the loop you just ran — *learn a model, imagine forward, plan* —
with the model made **learned, latent, and large**. And both inherit the toy's
honest catch at scale: a latent world model that must track a hidden situation
from partial glimpses is maintaining a **belief**, the [POMDP](../../decisions/pomdps-belief/)
machinery from the decisions Part, now learned by a network instead of written by
hand; and its imagined rollouts still compound whatever error the model has. The
dream is powerful, and the dream can be wrong.

---

## One Substrate

Step back from the corridor and look at what the toy actually *was*, because it
quietly used the entire book.

- Its world model is a **generative model** — "given a state and action, here is a
  distribution over what happens next" — the same object you have been writing
  since [Part I](../../foundations/) counted possible bento days and
  [Part II](../../genjax/) taught a laptop to imagine them.
- It was **fit by gradient descent**, the [Part VIII](../from-rules-to-weights/)
  move of turning a squared-error loss (a Gaussian likelihood in disguise) over
  learnable parameters and letting `jax.grad` walk downhill.
- And it was **used to make decisions** by rolling forward and scoring futures —
  the [Part VI](../../decisions/mdps/) machinery of value, rollout, and plan.

That is the thesis this book has been earning, chapter by chapter, stated plainly
just once: **one substrate runs from counting bentos to imagining futures.** A
generative model of the world, fit to data, queried to decide — the pieces never
changed; only what we pointed them at did. Chibany's very first question, *will
there be tonkatsu today?*, and the trainee's half-second rehearsal at the
cafeteria door are the same computation at two ends of a year.

---

## People Plan by Imagining Too

The cognitive-science reading is not a stretch here — it may be the *original* of
the idea. "Plan by imagining" is a leading account of how **people** think:
cognitive scientists argue the mind runs an internal simulator — a noisy mental
model of how the world will unfold — and uses it to decide, so that you catch a
ball, judge whether a stack of dishes will topple, or rehearse a route by *running
the outcome forward in your head* before you act (Battaglia, Hamrick & Tenenbaum,
2013). And the field has a sharp tool for catching the mind doing exactly the
model-based thing our toy does. Recall the **two-step task** from the
[Q-learning chapter](../../decisions/q-learning/): it was built precisely to tell a
model-free learner (who only caches which choice paid off) from a model-based one
(who simulates the task's structure to decide) — and people show the model-based
signature, evidence that human brains, too, hold a model and plan inside it. The
machinery of this chapter is not just an engineering trick; it is one of our
better theories of a mind.

{{% notice style="warning" title="Tanaka-san's take: \"so the robot dreams like we do\"" %}}
Tanaka-san watches the trainee freeze at the doorway and nods knowingly: "It's
daydreaming, same as me picturing my walk to the station. The robot dreams like
we do."

It is a lovely metaphor — and it is doing more work than it should. What the
trainee runs is not a dream in any rich sense; it is a **fitted next-state
predictor rolled forward**, seventeen-odd numbers tuned by gradient descent,
scored by distance to a goal. And the honest limit is the one our two-column
result already showed: **model error compounds fast.** Our corridor model was off
by only a sixth of a cell per step, and even that would drift if we imagined far
enough; a model of a real cafeteria — or a real Atari screen — is wrong in
richer, correlated ways, and each imagined step feeds the last step's error back
in. "It imagines the future" and "its imagination is *accurate*" are different
claims, and only the first comes free. That gap between a confident dream and a
correct one is exactly what [Part IX](../../ethics/) starts poking at.
{{% /notice %}}

{{% notice style="tip" title="Capstone: imagine your way to the goal (and watch it break)" %}}
Put the whole idea together on a small **gridworld**, using the scaffolded
notebook below. **Part A:** collect random-policy transitions and fit a dynamics
model (the corridor's next-state predictor, one dimension up). **Part B:** plan by
**imagination** — random-shooting or short-horizon rollouts through the learned
model — and race it against a model-free **Q-learner** on the *same* task, scoring
each by how many **real** steps it needs to solve the maze (imagined steps are
free; that is the model-based agent's edge). **Part C (stretch):** deliberately
**inject model error** — corrupt a few of the learned transitions — and watch
planning degrade as the horizon grows, so you *feel* the compounding-error lesson
rather than just read it. Anchors: **Dyna** (Sutton, 1991) and **MuZero**
(Schrittwieser et al., 2020), honestly described.

**📓 [Open in Colab: `capstone_world_models_imagination.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/capstone_world_models_imagination.ipynb)** — setup, gridworld, and data generator provided; the model fit, the planner, and the error-injection are TODOs.
{{% /notice %}}

---

{{% notice style="success" title="What you can do now" %}}
You can say what a **world model** is — a *learned* generative model of dynamics,
$\text{state}, \text{action} \to \text{next state}$ — and why it is a step past the
[MDP chapter](../../decisions/mdps/), which was handed the model rather than
learning it. You can explain **planning as imagination**: roll the learned model
forward, score the futures, act — the **Dyna** idea taken seriously — and you can
place it against **model-free** learning as a sample-efficiency-versus-compounding-error
trade in the bias–variance family. You **built** one end to end in pure JAX: fit a
next-state predictor by `jax.grad` (error $3.53 \to 0.15$ cells), then plan a route
to the goal inside it — watching the plan **succeed with the fitted model and fail
with the unfitted one**, because planning is only as good as the model you dream
in. You can describe **MuZero** (latent model sufficient for planning, searched
with MCTS) and **Dreamer** (behaviors learned by latent imagination) accurately
and without hype, and you can connect all of it to how **people** plan by
simulation — the two-step task's model-based signature.

**Closing Part VIII.** Five chapters ago this Part opened two black boxes — a
kiosk and a trainee — with one question: *how does it see, and how does it act?*
We answered it as one arc. [Vectors](../vectors-and-spaces/) turned a bento into a
point in space; [From Rules to Weights](../from-rules-to-weights/) made the
features themselves learnable and revealed every loss to be a likelihood;
[Neural Net Fundamentals](../neural-net-fundamentals/) stacked those learned
features into depth; [Transformers & Attention](../transformers-attention/) and
[LLMs & In-Context Learning](../llms-in-context-learning/) scaled the idea into
sequence models that appear to learn from a prompt the way a hierarchical Bayesian
would; and here, world models closed the loop by letting a network *imagine* and
*act*. Underneath, it was never anything but the book's one substrate: a generative
model, fit by gradient descent, used to decide. Which raises the question
[Part IX](../../ethics/) is for — now that machines learn how to see, speak, and
act **from us**, what exactly did we teach them, and can we check?

*Glossary:* [world model](../../glossary/#world-model-), [MuZero](../../glossary/#muzero-), [simulation-based RL](../../glossary/#simulation-based-rl-), [rollout](../../glossary/#rollout-), [trajectory](../../glossary/#trajectory-). (*Dyna*, *Dreamer*, and *planning-by-imagination* are described in this chapter and the [Q-learning](../../decisions/q-learning/) chapter; they do not yet have their own glossary entries.)
{{% /notice %}}

---

## Exercises

{{% notice style="info" title="Try it yourself" %}}
1. **Starve the model.** Drop `n_steps` in `collect` from $800$ to $40$ and refit.
   Does the after-fitting error stay small? Does the fitted model still plan a
   winning route — and if not, which cells did the short random walk never visit,
   so the model never learned them?
2. **Compound the error on purpose.** After fitting, add a small constant (say
   $+0.4$) to the model's prediction inside `imagine`, then push `HORIZON` from $6$
   up to $12$. At what horizon does the chosen plan stop reaching the goal? This is
   the compounding-error lesson made quantitative.
3. **Move the door.** Start the rollouts from the middle of the corridor (cell $3$)
   instead of the door, so both left and right are live options. Does the planner
   still pick the shortest route to the counter? What does that tell you about
   whether the *plan* or the *model* is doing the work?
{{% /notice %}}

A companion notebook works through all of this interactively:

**📓 [Open in Colab: `capstone_world_models_imagination.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/capstone_world_models_imagination.ipynb)**

Backward links: this chapter continues from
[LLMs & In-Context Learning](../llms-in-context-learning/) and is the learning-side
twin of [Modern RL & World Models](../../decisions/modern-rl-world-models/).

---

## References

- Battaglia, P. W., Hamrick, J. B., & Tenenbaum, J. B. (2013). Simulation as an engine of physical scene understanding. *Proceedings of the National Academy of Sciences, 110*(45), 18327–18332. <https://doi.org/10.1073/pnas.1306572110>
- Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., & Dolan, R. J. (2011). Model-based influences on humans' choices and striatal prediction errors. *Neuron, 69*(6), 1204–1215. <https://doi.org/10.1016/j.neuron.2011.02.027>
- Ha, D., & Schmidhuber, J. (2018). World models. *Advances in Neural Information Processing Systems (NeurIPS), 31*. <https://arxiv.org/abs/1803.10122>
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to control: Learning behaviors by latent imagination. *International Conference on Learning Representations (ICLR)*. <https://arxiv.org/abs/1912.01603>
- Schrittwieser, J., Antonoglou, I., Hubert, T., et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model. *Nature, 588*(7839), 604–609. <https://doi.org/10.1038/s41586-020-03051-4>
- Sutton, R. S. (1991). Dyna, an integrated architecture for learning, planning, and reacting. *ACM SIGART Bulletin, 2*(4), 160–163. <https://doi.org/10.1145/122344.122377>

---

*This project is generously funded by the Japanese Probabilistic Computing
Consortium Association ([JPCCA](https://jpcca.org/)).*
