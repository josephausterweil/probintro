# Plan: Tutorial 3 Chapters 8–11 — Bayesian Networks, Causal Bayes Nets, and Information Theory

**Author:** Drafted by Claude (Opus 4.7) in conversation with Prof. Austerweil, 2026-05-26.
**For:** A fresh agent in a new session to build out, chapter by chapter.

**RENUMBERED 2026-05-31:** Bayesian Generalization took **Ch 7** (now built, as the page bundle
`content/intro2/07_generalization/`), so this spine was shifted up by one: **Bayes Nets = Ch 8, Conditional
Independence = Ch 9, Causal Bayes Nets = Ch 10, Information Theory = Ch 11**. Hierarchical Bayes is **Ch 12**
(see the sibling plan), sitting after this spine. The old 7–10 numbering in this doc has been reconciled to
8–11 throughout.

**Sibling planning doc:** [`CHIBANY_T3_GENERALIZATION_PLAN.md`](CHIBANY_T3_GENERALIZATION_PLAN.md) — Bayesian Generalization (Ch 7) + Hierarchical Bayes (Ch 12), from Week 4 of HML SP26. Ch 7 conceptually precedes this spine; Ch 12 follows it.
**Course context:** Week 5 of *Human and Machine Learning* (SP26, Chiba Tech SDS) covers Bayesian networks, causal Bayes nets, the do-operator, and a short info-theory closer. The Week 5 lecture is now built around opening on the Gaussian mixture from Clusters and revealing it as a Bayes net students have already been writing. **The lecture moves beyond what the existing textbook covers** (T3 currently ends at Ch 7: Bayesian Generalization), so this plan adds Chapters 8–11 to T3 to give students a textbook track that parallels the lecture.

---

## TL;DR for the new agent

Build four new chapters (the **Bayes-net spine, Ch 8–11**) in `/home/jausterw/work/hummachlearn/spring2026/textbook/content/intro2/`:

| File | Title | Weight | Builds on | Lecture block it parallels |
|---|---|---:|---|---|
| `08_bayes_nets.md` | Bayesian Networks: Drawing What You Already Know | 8 | Ch 5 (GMM) | Blocks 1–3 (GMM-as-BN, complications, formal definition) |
| `09_conditional_independence.md` | Conditional Independence and d-Separation | 9 | Ch 8 | Blocks 4–5 (d-separation, explaining away) |
| `10_causal_bayes_nets.md` | Causal Bayes Nets and the Do-Operator | 10 | Ch 9 | Blocks 6–7 (observation vs. intervention, do-operator) |
| `11_information_theory.md` | Information Theory: Surprise, Uncertainty, and the Collider | 11 | Ch 9–10 | Block 9 (entropy, mutual info, collider in info-theory clothing) |

Each chapter:
- **Opens with a Chibany bento scenario** (concrete, narrative, like Ch 1's "mystery bentos" or Ch 5's "20 measurements").
- **Develops the math** with the same level as existing T3 chapters — light on derivations, heavy on intuition + worked examples.
- **Ends with a GenJAX section** that implements a sampler / inference routine for the chapter's model. Include a fenced `python` block with runnable code (the `validate_code_blocks.py` pre-commit hook validates these — see the **Style Rules** section below).
- **Cross-references neighboring chapters** in the existing T3 style (e.g., "Remember in Chapter 5...").

---

## Style rules the new agent MUST follow

These are derived from `textbook/CLAUDE.md` and from reading the existing T3 chapters. **Do not skip any of them.**

### Frontmatter

Every new chapter starts with:

```markdown
+++
date = "YYYY-MM-DD"   # ← today's date; the textbook CLAUDE.md says to update on every edit
title = "Chapter Title"
weight = N            # 8, 9, 10, 11 respectively
+++
```

Update `content/intro2/_index.md`'s sidebar listing to add the new chapters and (if you renumber any existing prereq listing) the mermaid graph too.

### Voice

- **Narrate from Chibany's perspective.** Chibany is the Chiba Tech mascot — uses they/them, eats two bentos a day, has a particular fondness for tonkatsu. Chibany is **not** a professor. Aligns with `textbook/content/intro2/01_mystery_bentos.md` — read that chapter before writing anything.
- **Open with the concrete scenario; abstraction comes later.** The Ch 1 → Ch 5 pattern: "Chibany received 20 bentos and noticed something strange" *first*, math *second*, GenJAX *third*. Hold to that order in every new chapter.
- **Math is set in `$$...$$` with $...$ inline.** Don't use `\[...\]` or `\(...\)` — the Hugo theme uses dollar-sign math.
- **Use `{{% notice style="info" title="..." %}}` boxes** for sidebars (info / warning / success / tip variants — pick by content; see existing chapters for samples).
- **Include at least one mermaid diagram per chapter** (a DAG, a flow, a dependency graph). The existing Ch 5 and `_index.md` show the dark-themed mermaid style the textbook uses.

### GenJAX code rules

- **Every chapter ends with a GenJAX section** titled "GenJAX Implementation" or similar.
- **Code blocks must validate** via `python validate_code_blocks.py` (see `textbook/CODE_VALIDATION.md`). The validator checks:
  1. Python syntax parses (every block must be syntactically valid Python).
  2. JAX-style: use `jnp.where(cond, a, b)` instead of Python `if cond: a else: b` for array-conditional logic.
  3. Imports are complete in every standalone block.
- **Use the standard import header** at the top of the first code block of each chapter:
  ```python
  import jax.numpy as jnp
  import jax.random as jr
  from genjax import gen, normal, flip, categorical
  ```
  (Adjust the `from genjax import` list per chapter as needed.)
- **Use the `@gen` decorator** for generative functions. Use `@ "name"` to name random choices. See `textbook/content/genjax/06_building_models.md` for the canonical pattern.

### Forward / backward references

- Each chapter explicitly references the previous one in its opening section ("In Chapter N, you learned...").
- Forward pointers are dim, optional, and short: "(Chapter 11 will show this in info-theoretic terms.)"
- Update `content/intro2/_index.md`'s mermaid graph + sidebar listing whenever a new chapter lands.

### Funding acknowledgment

Per `textbook/CLAUDE.md`: every new content page ends with a JPCCA acknowledgment. Match the existing pattern at the bottom of `_index.md`:

```markdown
Special thanks to [JPPCA](https://jpcca.org/) for their generous support of this tutorial series.
```

### Dates

`textbook/CLAUDE.md` is emphatic: **whenever you modify any content file, update its `date = "YYYY-MM-DD"` frontmatter to today**. Even small edits.

---

## Per-chapter plan

### Chapter 8: Bayesian Networks — Drawing What You Already Know

**Goal:** Reveal that the Gaussian mixture model from Ch 5 is already a Bayes net, then generalize to multi-parent networks. This is the textbook parallel of Week 5 Blocks 1–3.

**Chibany scenario:**

> Chibany is sitting at their desk staring at the histogram from Chapter 5. They've solved the mystery — bentos come from two clusters, and they can compute which cluster each bento came from. But now they're curious: what *exactly* did they just do? When they wrote `p(cluster | weight)`, they treated weight as the thing-they-saw and cluster as the thing-they-wanted-to-know. Where did that picture come from?
>
> Their friend Ira walks by and notices the histogram. "Oh, you're doing Bayesian-network inference!" "Bayesian what?" Chibany asks. "Let me show you," says Ira.

This sets up the recognition moment: the GMM was secretly a 3-node DAG ($\pi \to z_i \to x_i$) all along. The chapter then **builds up** to multi-parent networks (weather + day-of-week + restaurant → bento) so students see that the same picture-language scales to arbitrarily complex models.

**Sections (in order):**

1. **The Mystery Bentos, Drawn Differently.** Re-draw the GMM as a DAG with a plate. Identify each part: $\pi$ is the prior over cluster identity, $z_i$ is the cluster identity for bento $i$, $x_i$ is the observed weight. Single mermaid diagram + a paragraph naming each piece.
2. **What's a "parent"?** Define $\text{Pa}(X)$ as "the variables whose values directly determine $X$'s distribution." Walk through the GMM: $\text{Pa}(z_i) = \pi$, $\text{Pa}(x_i) = z_i$ (plus $\mu, \sigma$ which are global parameters).
3. **The factorization rule.** Introduce $P(X_1, \ldots, X_n) = \prod_i P(X_i \mid \text{Pa}(X_i))$ as the shorthand for "each node, given its parents, contributes one factor." Show that for the GMM, this gives $P(\pi) \prod_i P(z_i \mid \pi) P(x_i \mid z_i)$. Don't yet call it "the Markov factorization."
4. **Adding a hyperprior.** Show what happens when $\pi$ gets its own prior $P(\pi \mid \alpha)$. New DAG, new factorization. Connect explicitly to hierarchical Bayes (Ch 12) — *this is the same thing, drawn as a graph.* (One paragraph.)
5. **Multi-parent networks: Chibany's bento, revisited.** New scenario: Chibany realizes the bento weight depends on *which day* (cafeteria menu rotates), *which restaurant* the student came from, AND *the weather* (in hot weather, lighter bentos). Build the DAG: Weather → Bento, Day → Bento, Restaurant → Bento. Show the factorization: $P(W, D, R, B) = P(W) P(D) P(R) P(B \mid W, D, R)$.
6. **The parameter-counting argument.** For 4 binary variables, a full joint distribution needs $2^4 - 1 = 15$ numbers. The factored Bayes net needs $P(W)$ (1 num), $P(D)$ (1), $P(R)$ (1), and $P(B \mid W, D, R)$ (8 — one number per $(W, D, R)$ combination). Total: **11**, not 15. Modest gain at 4 nodes, exponential gain at scale.
7. **Naming what we just did: the Markov Factorization.** The formula is now named. Define DAG formally ($G = (V, E)$, acyclic). State the Markov factorization as a *theorem*: a graph $G$ is an **I-map** for $P$ iff $P$ factorizes per $G$. (One sentence on I-map; don't go deeper.)
8. **GenJAX Implementation.** Three samplers:
   1. **GMM as a Bayes net** — re-implement the Ch 5 mixture model with explicit node naming via `@ "..."`.
   2. **GMM with hyperprior** — add $\alpha \to \pi$.
   3. **Chibany's multi-parent bento** — full 4-node sampler with conditional probability tables.

**GenJAX skeleton (the third sampler, as an example for the new agent):**

```python
import jax.numpy as jnp
import jax.random as jr
from genjax import gen, flip, categorical

@gen
def chibany_bento_network():
    # Weather: 0 = cold, 1 = hot
    weather = flip(0.5) @ "weather"

    # Day-of-week: 0 = Mon-Wed, 1 = Thu-Fri
    day = flip(0.6) @ "day"

    # Restaurant: 0 = Cafeteria A, 1 = Cafeteria B
    restaurant = flip(0.7) @ "restaurant"

    # P(bento = tonkatsu | weather, day, restaurant) — CPT
    # Indexed by (weather, day, restaurant); 8 entries.
    cpt = jnp.array([
        # weather=0 (cold)
        [[0.7, 0.5], [0.6, 0.4]],  # day=0, day=1
        # weather=1 (hot)
        [[0.5, 0.3], [0.4, 0.2]],
    ])
    p_tonkatsu = cpt[weather, day, restaurant]
    bento = flip(p_tonkatsu) @ "bento"
    return bento
```

The chapter should explain each piece, walk through one ancestral-sample call, and show how to compute marginals via Monte Carlo.

---

### Chapter 9: Conditional Independence and d-Separation

**Goal:** Develop the structural rules (chain / fork / collider) that let students read independence statements off a graph. Cover the explaining-away pattern as the textbook's pedagogical centerpiece. Parallels Week 5 Blocks 4–5.

**Chibany scenario:**

> Last week, Chibany's friend Ira showed them how to draw their bento problem as a Bayes net. Now Chibany has a new puzzle: their colleague Yuki noticed that on days when the cafeteria runs out of tonkatsu, Chibany seems sleepier in the afternoon. "Maybe tonkatsu makes you sharp!" Yuki suggested. Chibany isn't so sure. Could there be a *third* variable explaining both?

This sets up the chain / fork / collider distinction through a concrete bento story:

- **Chain:** Bento type → Calories → Afternoon energy. If you condition on calories, bento type and energy become independent.
- **Fork:** A common cause (the cafeteria's daily menu) influences both what Chibany eats *and* what the other faculty members eat. Conditioning on the menu makes the two faculty members' bentos conditionally independent.
- **Collider:** The cafeteria's wet-floor sign comes out **either** when it rained outside **or** when someone spilled tea. The sign is a collider. Before you see it, rain and tea-spills are independent. *After* you see the sign and learn there was no spill, your belief about rain shoots up. (Concrete: the "explaining away" pattern.)

**Sections (in order):**

1. **A new puzzle.** The Chibany / Yuki tonkatsu-and-energy scenario from above.
2. **The three building blocks.** Chain, fork, collider — one mermaid DAG each, in three sub-sections. For each:
   - State the structural pattern.
   - Tell the bento-flavored story.
   - State the conditional-independence rule (chain & fork: conditioning on the middle node *blocks* dependence; collider: conditioning on the middle node *induces* dependence).
3. **The d-separation algorithm.** Given any DAG and any conditioning set, when is $A \perp B \mid C$? Walk through the algorithm informally — "trace every path between $A$ and $B$; if every path is blocked by $C$ or by an un-conditioned collider, they're d-separated." A worked example on a 5-node bento network.
4. **The Markov blanket.** Define it; show that conditioning on the Markov blanket renders a node conditionally independent from everything else. One mermaid showing the blanket highlighted.
5. **Explaining away in depth.** The Sprinkler / Rain / Wet-Grass example (or a Chibany-flavored version: Wet-floor-sign / Rain / Tea-spill). Walk through the numerical update:
   - $P(\text{rain})$ before seeing anything: 0.3.
   - After seeing the wet-floor sign: 0.6 (rain went up).
   - After *also* learning a student spilled tea: 0.35 (rain went back down — the spill "explained" the wet-floor sign).
   This is the textbook's standout pedagogical figure; spend the most space here.
6. **GenJAX Implementation.** Implement an explaining-away sampler with `flip` for binary nodes; show the inference (rejection sampling or importance sampling — whichever is cleanest in current GenJAX) that recovers the three posteriors.

**GenJAX skeleton:**

```python
import jax.numpy as jnp
import jax.random as jr
from genjax import gen, flip

@gen
def wet_floor_network():
    # Independent causes
    rain = flip(0.3) @ "rain"
    tea_spill = flip(0.1) @ "tea_spill"

    # Common effect — the wet-floor sign goes out if EITHER cause is true
    # Noisy-OR: each cause "fires" with probability 0.9 conditional on being true
    p_sign_if_rain = 0.9
    p_sign_if_tea = 0.8
    # P(sign = true | rain, tea_spill) = 1 - (1-p_rain)^rain * (1-p_tea)^tea
    p_sign = 1.0 - (1.0 - p_sign_if_rain) ** rain * (1.0 - p_sign_if_tea) ** tea_spill
    sign = flip(p_sign) @ "sign"
    return sign
```

The chapter then shows how to *condition* on observed sign + observed tea_spill via GenJAX's constraint / importance sampling API to recover $P(\text{rain} \mid \text{sign}, \text{tea\_spill})$.

---

### Chapter 10: Causal Bayes Nets and the Do-Operator

**Goal:** Distinguish observational from interventional probability. Introduce the do-operator and graph surgery. Parallels Week 5 Blocks 6–7.

**Chibany scenario:**

> Chibany is at the campus health center for a checkup. The doctor mentions that people with yellow teeth tend to have a higher risk of lung cancer. "Should I whiten my teeth to lower my risk?" Chibany asks, half-joking. The doctor laughs. "That's not how it works." But *why* isn't that how it works? Chibany has a Bayes net for it — Yellow teeth and Lung cancer are statistically associated. So shouldn't intervening on one affect the other?

This sets up the central distinction:

- $P(\text{lung cancer} \mid \text{yellow teeth}) > P(\text{lung cancer})$ — true, observationally.
- $P(\text{lung cancer} \mid do(\text{yellow teeth} = \text{whitened})) = P(\text{lung cancer})$ — true, interventionally.

Same notation, different operation, different answer.

**Sections (in order):**

1. **The dentist's puzzle.** The scenario above.
2. **Bayes nets that *mean* something.** Up to now, our DAGs have just expressed conditional independence — they encode statistical structure. A **causal Bayes net** is the same machinery, but the edges now mean "this causes that." The same statistical pattern (yellow teeth ↔ lung cancer) is compatible with three different causal stories:
   - Yellow teeth → Lung cancer
   - Lung cancer → Yellow teeth
   - Common cause (smoking) → both
   These three are **Markov-equivalent** under observation alone. *Observation cannot distinguish them.*
3. **Intervention as graph surgery.** When you *intervene* on $X$ (set it to a specific value by hand, e.g., by paying for tooth whitening), you cut all arrows pointing INTO $X$. The rest of the graph is unchanged. This is the do-operator: $do(X = x)$.
4. **$P(Y \mid X)$ vs. $P(Y \mid do(X))$.** Walk through the smoking / teeth / cancer example numerically:
   - Original: $S \to T$, $S \to L$. Confound.
   - Compute $P(L \mid T = \text{yellow})$: depends on $T$ through the back-door path $T \leftarrow S \to L$.
   - Compute $P(L \mid do(T = \text{yellow}))$: the do-operator cuts $S \to T$, so $T$ no longer carries information about $S$. Answer: $P(L)$.
   - Two different answers from the "same" question.
5. **Pearl's ladder of causation.** Brief mention: Level 1 (observation, $P(Y \mid X)$), Level 2 (intervention, $P(Y \mid do(X))$), Level 3 (counterfactuals, $P(Y_x \mid X = x', Y = y')$). Stay on Levels 1 and 2.
6. **The blicket detector.** A short cog-sci payoff section: Gopnik & Sobel's experiment showed that children's causal inferences require intervention to compute, and they get it right by age 3. Cite the paper but don't expand. One paragraph.
7. **GenJAX Implementation.** Show two functions: one that samples from the *observational* distribution (standard ancestral sampling), and one that samples from the *interventional* distribution (clamp the intervened node, ignore its parents). Compute $P(L \mid T = \text{yellow})$ and $P(L \mid do(T = \text{yellow}))$ by Monte Carlo from each, side by side.

**GenJAX skeleton:**

```python
import jax.numpy as jnp
import jax.random as jr
from genjax import gen, flip

@gen
def smoking_network_observational():
    smoking = flip(0.3) @ "smoking"
    # P(teeth_yellow | smoking)
    p_teeth = jnp.where(smoking, 0.7, 0.2)
    teeth_yellow = flip(p_teeth) @ "teeth_yellow"
    # P(lung_cancer | smoking)
    p_lung = jnp.where(smoking, 0.15, 0.01)
    lung_cancer = flip(p_lung) @ "lung_cancer"
    return lung_cancer

@gen
def smoking_network_intervened(teeth_value):
    """do(teeth_yellow = teeth_value) — teeth is now exogenous."""
    smoking = flip(0.3) @ "smoking"
    # Cut the arrow into teeth — set it directly.
    teeth_yellow = teeth_value  # NOT a random choice
    # Lung cancer still depends only on smoking, NOT on teeth.
    p_lung = jnp.where(smoking, 0.15, 0.01)
    lung_cancer = flip(p_lung) @ "lung_cancer"
    return lung_cancer
```

The chapter then walks through Monte Carlo estimates of $P(L \mid T = \text{yellow})$ from the observational model (conditioning on `teeth_yellow = True`) vs. $P(L \mid do(T = \text{yellow}))$ from the intervened model.

**Pedagogical note for the agent:** The hardest concept in this chapter is that **conditioning and intervening look syntactically identical but produce different distributions**. Use the side-by-side numerical comparison as the spine. The student should be able to compute both quantities and *see* that they differ.

---

### Chapter 11: Information Theory — Surprise, Uncertainty, and the Collider

**Goal:** A short closing chapter that introduces entropy, mutual information, and ties them back to d-separation. Parallels Week 5 Block 9.

**Chibany scenario:**

> Chibany has been keeping a journal of which bento they receive each day. Some days they correctly predict tonkatsu; some days they're surprised. Today, they got a hamburger when they were sure it'd be tonkatsu. "How surprised should I have been?" Chibany wonders. Their friend Ira walks by again — "There's a way to measure that, you know."

**Sections (in order):**

1. **The surprise question.** Chibany's journal scenario. Lead with the question: "how surprised should I have been by this outcome, given my beliefs going in?"
2. **Surprise = $-\log P(x)$.** Define surprise as a function of the probability you assigned to the outcome. Walk through: if you assigned $P = 0.99$ and the event happens, surprise ≈ 0; if you assigned $P = 0.01$, surprise ≈ 6.6 bits. Show why log is the right function (additive over independent events; tied to coding theory but don't lecture on it).
3. **Entropy = expected surprise.** $H(X) = -\sum_x P(x) \log P(x) = \mathbb{E}[-\log P(X)]$. Walk through:
   - Fair coin: $H = 1$ bit.
   - 70/30 coin: $H ≈ 0.88$ bits.
   - Deterministic outcome: $H = 0$.
   - Why this is "uncertainty" or "average code length."
4. **Mutual information.** $I(X; Y) = H(X) - H(X \mid Y)$. Define as "how much knowing $Y$ reduces uncertainty about $X$." Show the symmetry ($I(X; Y) = I(Y; X)$). One-paragraph derivation of the symmetric form.
5. **Independence in info-theoretic terms.** $X \perp Y \iff I(X; Y) = 0$. Conditional version: $X \perp Y \mid Z \iff I(X; Y \mid Z) = 0$.
6. **The collider, in info-theoretic clothing.** Take the wet-floor-sign / rain / tea-spill collider from Chapter 9. Before conditioning on the sign, rain and tea_spill are independent: $I(R; T) = 0$. After conditioning on the sign: $I(R; T \mid \text{sign}) > 0$. **Conditioning on a collider creates mutual information from nothing** — and that's exactly explaining-away, viewed through a different lens.
7. **A note on KL divergence.** Brief: $D_{KL}(P \| Q) = \sum_x P(x) \log(P(x)/Q(x))$ — "how much does $Q$ underfit $P$ when measured in surprise units." Mention as a forward pointer (will come up in Week 11's neural-network chapter; don't develop).
8. **GenJAX Implementation.** Compute entropy and mutual information by Monte Carlo. Implementation: sample many ancestral traces from a Bayes net, estimate $H(X)$ as the empirical $-\sum \log p(x)$ averaged over samples, and $I(X; Y \mid Z)$ as the difference $H(X \mid Z) - H(X \mid Y, Z)$.

**GenJAX skeleton:**

```python
import jax.numpy as jnp
import jax.random as jr
from genjax import gen, flip

@gen
def wet_floor_with_loglikelihood():
    rain = flip(0.3) @ "rain"
    tea_spill = flip(0.1) @ "tea_spill"
    p_sign = 1.0 - (1.0 - 0.9) ** rain * (1.0 - 0.8) ** tea_spill
    sign = flip(p_sign) @ "sign"
    return rain, tea_spill, sign

def monte_carlo_entropy(model, key, n_samples=10000):
    """Estimate H(X) by averaging -log P(x) over ancestral samples."""
    keys = jr.split(key, n_samples)
    log_probs = []
    for k in keys:
        trace = model.simulate(k, ())
        log_probs.append(trace.get_score())
    return -jnp.mean(jnp.array(log_probs))
```

(The new agent will need to adapt this to the current GenJAX API — `simulate` and `get_score` are illustrative; the agent should verify against the latest GenJAX docs.)

---

## Cross-chapter assets to create / update

### `_index.md` updates

After all four chapters are written, update `content/intro2/_index.md`:

1. Bump the `date` field.
2. Extend the mermaid learning-path diagram (Ch 7 = Bayesian Generalization already lands before this spine):
   ```mermaid
   graph TB
       A[1. Mystery Bentos] --> B[2. Continuous Probability]
       B --> C[3. Gaussian Distribution]
       C --> D[4. Bayesian Learning]
       D --> E[5. Mixture Models]
       E --> F[6. Dirichlet Process]
       F --> G[7. Bayesian Generalization]
       G --> H[8. Bayesian Networks]
       H --> I[9. Conditional Independence]
       I --> J[10. Causal Bayes Nets]
       J --> K[11. Information Theory]
       K --> L[12. Hierarchical Bayes]
   ```
3. Add the four new sections ("Chapter 8: Bayesian Networks", "Chapter 9: ...", etc.) following the existing format.
4. Update the prerequisites for the new chapters — Ch 8 requires Ch 5; Ch 9 requires Ch 8; Ch 10 requires Ch 9; Ch 11 requires Ch 9.

### New images (optional, low-priority for the agent)

The existing chapters include Chibany illustrations (`images/chibanyplain.png`, `images/chibanylayingdown.png`). The new chapters could each have a Chibany illustration matching the chapter mood (Chibany drawing a graph, Chibany at the doctor's office, Chibany measuring surprise). **These do not exist** — the agent should NOT block on them. Write the chapters without illustrations; flag in a TODO for Joe to commission them later.

### Validation

After writing each chapter, the new agent MUST run:

```bash
cd /home/jausterw/work/hummachlearn/spring2026/textbook
python validate_code_blocks.py
```

…and resolve any failures before committing. The pre-commit hook will refuse the commit otherwise.

### Tests (optional but recommended)

The textbook repo has `test_building_models.py`, `test_ch5_code.py`, etc. — pattern after these for the new chapters. One test per chapter that exercises the canonical generative function.

---

## Suggested order of operations for the new agent

1. **Read `textbook/CLAUDE.md` and `textbook/CODE_VALIDATION.md` in full.** These contain rules not duplicated here.
2. **Read `content/intro2/01_mystery_bentos.md` and `content/intro2/05_mixture_models.md` in full.** These are the canonical Chibany-narrative + math + GenJAX templates.
3. **Read `content/genjax/06_building_models.md` in full.** This is the canonical "build a GenJAX model from scratch" template for the GenJAX sections of each chapter.
4. **Confirm the current GenJAX API.** Before writing any code blocks, run `python -c "import genjax; help(genjax)"` (or equivalent) to check that `gen`, `flip`, `categorical`, `normal`, `simulate`, `get_score`, etc. work as the skeletons above assume. The GenJAX API has been moving; the skeletons may need tweaking.
5. **Write Chapter 8 first**, validate code blocks, ask Joe for review before writing Chapter 9.
6. **Sequence the remaining chapters**: 8 → 9 → 10 → 11.
7. **Update `_index.md` once** at the end (or incrementally after each chapter).
8. **Run `validate_code_blocks.py` after every chapter.** Do not skip.
9. **Update dates on every modified file.**

---

## Things to ask Joe about before writing

Before the new agent commits the first chapter, surface these to Joe:

1. **Friend characters.** The plan introduces *Ira* (already in `textbook/TimCommentsOnChibany.pdf`?) and *Yuki* as supporting characters. Joe should confirm these names fit, or substitute.
2. **CPT values.** The CPTs in the Chibany bento and smoking-network skeletons are made-up. Joe may want to tune them so the Monte Carlo estimates have particular pedagogical properties (e.g., $P(L \mid T)$ noticeably > $P(L \mid do(T))$).
3. **How much info theory.** Chapter 10 is the lightest of the four. If Joe wants more (e.g., a fuller KL-divergence treatment, or cross-entropy as loss function), expand the chapter; otherwise leave it as a short capstone.
4. **Notebook companions.** The existing T3 chapters often link to Colab notebooks. Should each new chapter get a companion notebook? If yes, the agent will need to author them under `textbook/notebooks/` and link them via the standard `[📓 Open in Colab: ...]` pattern. If no (lower lift), the GenJAX code lives only in the markdown.

---

## What this plan does NOT cover

- **No new exercises with solutions.** The existing T3 chapters have light exercises; the new ones could too, but this plan does not enumerate them. Defer to the agent to draft exercises that match the chapter's level.
- **No coverage of HMMs or temporal Bayes nets.** That belongs in Week 6's Markov-chain content, not here.
- **No coverage of structure learning.** "Where does the DAG come from?" is its own topic (and a hard one). Out of scope for these four chapters; mention as a forward pointer in Ch 9 if it fits.
- **No coverage of variable elimination, junction trees, or belief propagation.** Inference algorithms beyond Monte Carlo are out of scope; the chapters use GenJAX-driven sampling/importance for all inference.

---

## Handoff complete

This document is the full plan. A new agent in a fresh session, given **(a)** this file, **(b)** read-access to the repo, and **(c)** the briefing "build out the Bayes-net spine, Chapters 8–11, per `textbook/CHIBANY_T3_CH7-10_PLAN.md`, starting with Chapter 8, ask Joe before moving to Chapter 9," has everything they need to begin. (The filename still says `CH7-10` for git-history continuity; the content is the Ch 8–11 spine.)

— Drafted by Claude (Opus 4.7) with Prof. Austerweil, 2026-05-26.
