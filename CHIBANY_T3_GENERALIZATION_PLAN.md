# Plan (REVISED): Tutorial 3 — Bayesian Generalization (Ch 7) + Hierarchical Bayes (Ch 12)

**Original draft:** Claude (Opus 4.7) with Prof. Austerweil, 2026-05-27.
**Revised:** Claude (Opus 4.8) with Prof. Austerweil, 2026-05-30 — rewritten against the *built* Week-4 deck
(`course/week04_generalization_hier_bayes/week4-slides.qmd`, 2299 lines) and the *shipped* assignment
(`course/assignments/generalization/generalization.tex`). The earlier draft predated both; this version
integrates the slides' pedagogy, figures, and exact worked numbers, and weaves GenJAX through every section.
**Sibling planning doc:** [`CHIBANY_T3_CH7-10_PLAN.md`](CHIBANY_T3_CH7-10_PLAN.md) — the Bayes-net spine (now Ch 8–11).

---

## What changed in this revision (read first)

The original plan was written before the lecture existed. Now that the deck is built and the assignment is
shipped, the plan is revised on five axes:

1. **Chapter order decided (with Prof. Austerweil).** Generalization is a *standalone* chapter that comes
   **before** the Bayes-net spine; the GMM→Bayes-net spine is Ch 8–11; **Hierarchical Bayes is Ch 12** (end),
   not adjacent to generalization. Files to write now: **`07_generalization.md`** and
   **`12_hierarchical_bayes.md`**; **reserve 08–11** for the sibling plan. (See "Numbering" below — the old
   A/B options are obsolete.)

2. **PRIVACY — Shohei's materials are excluded.** The deck's "student presentation" block is Tenenbaum & Xu
   (2000), *Word learning as Bayesian inference*, presented by a student. **The textbook must not use
   Tenenbaum & Xu, the dax/Dalmatian word-learning bridge, or any "student presenter" framing.** Where the
   deck bridges to that paper, the chapter instead develops the **six-animals property-induction example**
   (which is *the assignment*, and is the professor's own material). This keeps the chapter self-contained and
   privacy-clean while losing nothing pedagogically — the size principle still does all the work.

3. **Three on-ramps, one equation.** Per Prof. Austerweil: keep the original "golden-sticker bento" discrete
   scenario **and** add the deck's two canonical games. Order: **golden sticker (gentle discrete intro) →
   number game (discrete, the size-principle arithmetic) → rectangle game (continuous, Shepard's law
   derived)**. All three are the *same* equation with a different $\mathcal{H}$ — that unification is the
   chapter's spine, exactly as in the deck.

4. **GenJAX throughout (not one appendix).** Every conceptual section gets a small, runnable GenJAX/JAX cell,
   and at least one cell **regenerates a lecture figure** so code ↔ figure ↔ math line up. The original plan
   put GenJAX in a single closing section; that's demoted in favor of inline integration. All code validates
   via `validate_code_blocks.py` and is *executed* against GenJAX 0.10.3 (importable in the default env and in
   `textbook/.ch5_test_venv`) before commit.

5. **Worked example = the six animals, aligned to Assignment 2.** The shipped `generalization.tex` is
   **Assignment 2: Bayesian Generalization**, six animals (Cow, Dolphin, Chicken, Seal, Penguin, Bat), **due
   Fri Jun 19, 2026, 8:00 PM** — *not* the Clusters assignment (Jun 5) the deck's Block 1 tours. Ch 7 is the
   prep-reading for Assignment 2, and its running worked example is the six animals, matching the assignment's
   Problems 1–5 one-to-one (define $\mathcal{H}$ → prior → posterior weak/strong → predictive → "break your
   model"). The `.tex` already says the chapter is "in preparation … lecture slides are the canonical
   reference until then." Ch 7 fills that and the 404'd `…/intro2/06_generalization/` link.

---

## Reusable assets already on disk (don't recreate)

**Figures** — `course/week04_generalization_hier_bayes/images/`, dark-themed (the textbook is dark-themed too,
so they drop in). Copy the ones used into `textbook/content/intro2/images/`:

| File | Shows | Ch 7 section |
|---|---|---|
| `shepard_decay.png` | $g(d)=e^{-d}$ generalization gradient + vertical-line stimulus strip | §3 Shepard's law |
| `tg_vote.png` | One datum; stacked candidate intervals (thickness $=1/|h|$) over the posterior-weighted vote | §5 the vote |
| `tg_vote_y0/y1/y2.png` | The $y=x,\,x{+}1,\,x{+}2$ walk-through (greyed dropouts) | §5 build-up |
| `cc_1d.png` | 1-D interval hypotheses over observed points | §8 rectangle game (1-D) |
| `cc_1d_gradient.png` | Multi-example 1-D gradient: flat over data, decays outside | §8 gradient |
| `cc_2d.png` | Nested 2-D rectangles with $r$, $d$, $n$ labelled | §8 rectangle game (2-D) |
| `cc_exp_prior.png` | The exponential density $\lambda e^{-\lambda s}$ | §8 exponential prior |
| `tg_results.png` | Human (solid) vs. model-no-prior (dashed), $d$ vs $r$ by $n$ — model over-extends | §8 the gap |
| `tg_results_prior.png` | Same axes, model **with** exponential prior — curves bend onto human data | §8 the fit |

Figure-build scripts live under `course/week04_generalization_hier_bayes/scripts/`
(`build_tg_integration_plot.py`, `build_continuous_concept_plots.py`, `build_suspicious_coincidence_plot.py`).
Use them as the basis for the GenJAX "regenerate a figure" cells (rewrite to call the chapter's own sampler so
the code in the chapter *produces* the figure rather than just displaying a PNG — at minimum for the 1-D
gradient and the six-animals predictive).

**Exact worked numbers from the deck** (use verbatim so textbook ⇄ lecture agree):
- Number game, $\mathcal{H}=\{$mult-10, even$\}$, flat prior, **strong** sampling:
  $X=\{60\}\Rightarrow 0.83/0.17$ (5:1); $X=\{60,80,10,30\}\Rightarrow 0.998/0.002$ ($5^4=625{:}1$).
- **Weak** sampling, same $X$: $0.5/0.5$ — and the subtle, correct point the deck makes: weak sampling **can**
  eliminate (a datum outside $h$ ⇒ likelihood 0) but **cannot rank** survivors; the $0.5$ is an artefact of
  *two* survivors + flat prior. Carry this nuance into the chapter; it is a common over-simplification.
- Rectangle game: exponential prior $\sigma=5$ in a 24-unit window gives Tenenbaum's (1999) excellent fit.

**Style/voice templates** — `01_mystery_bentos.md` (Chibany narrative + misconception boxes + practice
problems), `05_mixture_models.md` (math + GenJAX `@gen`/`ChoiceMap.d`/`generate`+importance-weights, the
`N_OBS`-closed-over gotcha), `genjax/06_building_models.md` (the canonical model-building patterns + the
math↔code translation table). Read all three before writing.

**Characters:** Chibany (mascot, they/them, two bentos a day, tonkatsu fan — *not* a professor). Supporting
friends **Jamal** and **Alyssa** (these replace the old plan's Ira/Yuki). No student-presenter character.

---

## Numbering (decided — supersedes the old A/B options)

T3 currently ends at Ch 6 (DPMM). Final order:

| Ch | File | Title | From | Write now? |
|---:|---|---|---|:--:|
| 7 | `07_generalization.md` | Bayesian Generalization | *this doc* | **yes** |
| 8 | `08_bayes_nets.md` | Bayesian Networks (built from the Ch 5 GMM) | sibling plan | reserve |
| 9 | `09_conditional_independence.md` | Conditional Independence & d-Separation | sibling plan | reserve |
| 10 | `10_causal_bayes_nets.md` | Causal Bayes Nets & the Do-Operator | sibling plan | reserve |
| 11 | `11_information_theory.md` | Information Theory | sibling plan | reserve |
| 12 | `12_hierarchical_bayes.md` | Hierarchical Bayes | *this doc* | **yes** |

The sibling `CHIBANY_T3_CH7-10_PLAN.md` must be renumbered 7→8, 8→9, 9→10, 10→11 in lockstep **before** any
spine chapter is committed. (Not part of *this* task; flagged for whoever writes the spine.) Ch 12 is written
now to reference **Ch 7** as its conceptual hook (NFL → "you need a prior" → "so learn it") and to treat the
Ch 8–11 spine as dim forward/back pointers, so it reads correctly whether or not the spine exists yet.

---

# Chapter 7 — Bayesian Generalization

**Goal.** Develop the Shepard → Tenenbaum & Griffiths generalization framework — hypothesis space, the
posterior-weighted vote, weak vs. strong sampling, the size principle, the predictive distribution, and No
Free Lunch — as a direct continuation of Ch 4 (Bayesian learning). Parallels the entire pre-break Week-4
lecture **plus** the NFL capstone, and is the prep-reading for Assignment 2 (six animals). GenJAX is woven
through every section; the six-animals example is the running thread.

## Scaffolding analysis: what the student arrives with (revision driver)

Added 2026-05-30 after re-reading what Tutorials 1–3 actually teach. **Build the chapter on the reader's real
prior knowledge, not an idealized one.** A student who has worked T3 Ch 1–6 and T2 (GenJAX) arrives with:

**Conceptual anchors to build on:**
- Bayes' rule as **posterior ∝ likelihood × prior** — in *every* prior chapter (T1 Ch 5, T3 Ch 4, T2 Ch 4).
  Reuse the exact phrase and shape.
- **Prior, likelihood, posterior, and predictive distribution** — all explicitly named; T3 Ch 4 teaches the
  posterior-predictive ("what's the next bento?"). Ch 7's predictive has a real anchor — name it.
- **Conditioning = restricting the outcome space** (T2 Ch 4 taxicab; T1 Ch 4). The anchor for "which
  hypotheses survive the data."
- **Categorization** P(category | x) over two Gaussians (T3 Ch 4 preview, Ch 5). The six-animals "which
  hypotheses contain $y$" is its sibling — make the link explicit.

**The ONE big conceptual leap (must be scaffolded, not assumed):** in every prior chapter the unknown $h$/$H$
is a **parameter** (μ) or a **binary event** (`is_blue`, `is_tonkatsu`). The reader has **never** seen $h$ be
a **set of items**. Jumping straight to "$\mathcal{H}$ is a space of 63 sets we enumerate" is the cliff. **A
dedicated bridge section converts $h$=event into $h$=set on a two-hypothesis toy, before any sum over
$\mathcal{H}$ appears.**

**GenJAX knowledge (the code ladder must start here):**
- KNOWN: `@gen`, `flip`, `normal`, `uniform`, `categorical`, naming via `@ "x"`, `simulate`,
  `ChoiceMap.d({...})`, `generate(...) → (trace, weight)`, and the **importance-weights idiom**
  (`w = jnp.exp(logw - logw.max()); w /= w.sum()`). T2 Ch 6 has the math↔code table.
- NEVER seen: `jax.vmap` over a **hypothesis matrix** (every prior vmap is over PRNG keys/particles, never a
  data array), **enumeration as exact inference** (all prior inference is sampling), `jnp.where` to switch a
  *whole likelihood formula* (prior `jnp.where` only ever picks a per-element value — a mean or a
  probability), `itertools` (never appears anywhere), and **building a 0/1 hypothesis matrix from scratch**.
  Introduce enumeration as "a simpler inference that works *because* $\mathcal{H}$ is a finite list" — a
  simplification of the sampler they know, not a new burden.
- **Not reliably known:** `beta` — appears ONLY in T3 Ch 6 (DPMM stick-breaking), which many readers skip.
  (Ch 7 doesn't use `beta`; Ch 12 does and defines Beta from scratch — see Ch 12 §1b.) *An earlier draft of
  this analysis said `beta` was "never seen" — wrong; corrected per audit finding m1.*
- **Known inconsistency #1 (imports):** older T3 chapters write `jnp.normal`/`jnp.bernoulli` (not the real
  API); newer chapters + T2 write `normal`/`flip` from `genjax`. Ch 7 uses the **correct `genjax` imports
  throughout**; do not replicate the `jnp.normal` typo.
- **Known inconsistency #2 (audit finding M1 — `generate` argument order):** the student just read `generate`
  written TWO ways in adjacent chapters — `generate(key, observations, args)` (obs **second**: taxicab T2
  Ch 4, mixture T3 Ch 5) and `generate(key, args, observations)` (obs **last**: the coin models in T2 Ch 6).
  Ch 7 must pick **one** order everywhere; use **observations-second** (matches the most-recently-read Ch 5
  and the taxicab). First `generate` (§2 bridge cell) adds a one-line note: "earlier chapters varied the
  order; we use observations-second consistently here." The enumeration core avoids `generate`, but §2 and the
  §5 "show they agree" callback use it — so the order must be settled.
- **Known inconsistency #3 (`simulate` signature):** T3 Ch 2 uses `model.simulate(subkey)`, Ch 1/3 use the
  free-function `simulate(model)(subkey)`, Ch 5 uses `model.simulate(k, ())`. Ch 7 adopts the Ch 5 form
  (`model.simulate(key, args)`) throughout; mention once if a `simulate` call appears.

**Three design rules that follow:** (1) **bridge before formalism** — event → set on a two-hypothesis toy
before the Σ; (2) **discrete before continuous, and say why** — the reader was trained (T3 Ch 2) that
"continuous is the hard case"; here discreteness makes $\mathcal{H}$ a finite list you can enumerate, the
*easy* case, so signpost the inversion; (3) **GenJAX ladder reuse → extend → new** — known `@gen` idioms
first, then the `generate`+weights conditioning they know, only *then* enumeration as the finite-$\mathcal{H}$
shortcut, each new idiom prefaced "you already did X; this is X with a vector."

## Chapter 7 narrative arc (scaffolded order)

A concrete two-set example comes *before* the abstract framework, and the discrete number game *before* the
continuous rectangle game (so "enumerate a finite list" comes first). The six-animals example threads from §4
onward as the running worked case, in lockstep with Assignment 2's Problems 1→5.

```
0. Recall box — what you already know (prior/likelihood/posterior/predictive; conditioning = restriction).
1. The golden sticker  — informal hook; "a rule is a SET of bentos"; no math. (on-ramp 1)
2. Bridge: from "which event?" to "which set?"  — the ONE leap, on a 2-hypothesis toy. (KEYSTONE — NEW)
3. Shepard's law  — the empirical target the framework must reproduce (why we need a model).
4. The framework, formally  — H, prior, likelihood, posterior; "H IS a prior". Safe now: sets are familiar.
5. Generalization = a posterior-weighted vote  — the chapter equation + the single-datum vote build-up.
6. Weak vs strong sampling + the size principle  — the likelihood, made concrete.
7. On-ramp 2: the number game  — discrete H; size-principle arithmetic; graded vs rule-like.
8. On-ramp 3: the rectangle game  — continuous H; Shepard's law DERIVED; the exponential prior.
9. No Free Lunch  — why the prior is unavoidable; the "break your model" capstone (Assignment 2 Problem 5).
10. Summary + practice  — takeaways; assignment-shaped (not -identical) problems; forward pointers.
```

**Opening Chibany scenario (the golden sticker — on-ramp 1).**

> Chibany has been receiving bentos for months and notices that *some* have a tiny **golden sticker** on the
> side and others don't. What does the sticker mean — the day of the week? The chef? A price tier? Chibany
> doesn't know the rule; they only get to see which bentos have the sticker. They decide to figure it out the
> way a probabilist would: list every plausible rule ("only Mondays", "only tonkatsu", "bentos over 400 g",
> "every bento"), and watch which rules survive as more sticker'd bentos show up.

The sticker is a *novel property*; each rule is a *hypothesis* (a set of bentos); the surviving rules after
data are the *posterior*. This is the gentlest possible entry to the same machinery the chapter then runs on
the number game, the rectangle game, and the six animals.

### Section plan (each section opens by naming what is REUSED / NEW; then narrative/math → runnable GenJAX cell)

0. **Recall box — "what you're bringing with you"** (a `{{% notice style="info" %}}` before the hook, not a
   full section). Inventory the four anchors with the existing `[← Review …]` cross-links: posterior ∝
   likelihood × prior (T3 Ch 4), the predictive distribution (T3 Ch 4), conditioning = restricting to what's
   consistent with the data (T2 Ch 4 taxicab), categorization P(category | x) (T3 Ch 5). End with the promise:
   "This chapter changes *one* thing — what a hypothesis *is*. Everything else you already know."

1. **The golden sticker (informal hook).** The scenario above. Frame the question ("I see one sticker'd bento
   — what should I believe about the next?"). Establish the one new idea in words, no formulas: *a hypothesis
   here is a **rule**, and a rule is a **set** of bentos* ("only Mondays" = the set of Monday bentos). *No
   GenJAX yet* — keep the hook pure narrative so the new idea lands before any code. (Original plan put a `@gen`
   cell here; deferred to §2 so the first code is on the familiar two-hypothesis bridge, not a hypothesis
   matrix.)

2. **Bridge: from "which event?" to "which set?" (KEYSTONE — the section the original plan lacked).** Walk the
   one conceptual leap on a tiny, holdable example:
   - **Start where they are.** Taxicab (T2 Ch 4): the unknown was one of *two events*, $h\in\{\text{blue},
     \text{green}\}$; Bayes ranked them. Recall in two lines.
   - **Reinterpret as sets.** "Blue" *is* the set of blue taxis (callback to T1 "event = subset of the outcome
     space"). So $h$ was always a set — we just never needed to see it that way.
   - **Let the sets overlap and vary in size.** Generalization is the case where candidate sets **overlap** (an
     item can be in many hypotheses) and **differ in size** (broad vs. narrow rules). That's the only genuinely
     new wrinkle.
   - **2-hypothesis worked example** (two rules over a few bentos): compute the posterior over the *two* by
     hand (reusing posterior ∝ likelihood × prior), then ask "is the next bento sticker'd?" by checking which
     surviving rule contains it. Previews the vote (§5) and size principle (§6) on a 2-element $\mathcal{H}$
     before either is named. *GenJAX cell — the first code of the chapter, using ONLY known idioms:* a `@gen`
     model that does `categorical` over the two rules then `flip`/`categorical` for the datum, plus the
     `generate`+weights conditioning from T2 Ch 4. Output: the 2-hypothesis posterior. Anchors everything in
     inference they already trust. **This is where the `generate` argument order gets pinned (audit finding
     M1):** use `generate(key, observations, args)` (observations second) and add the one-line note that
     earlier chapters varied the order — so every later `generate` in the chapter is consistent and the reader
     isn't second-guessing the call signature they saw written two ways in T2 Ch 4 vs T2 Ch 6.

3. **Shepard's universal law (the empirical target).** Shepard (1987): generalization decays *exponentially*
   with distance in **psychological space** (define — perceived, not physical, distance). $g(d)=e^{-d}$; the
   law is *descriptive* — exponential, but not *why*. State the promise: the framework will **derive** it
   (paid off in §8). *Figure:* `shepard_decay.png`. Short and motivational — it sets the bar so §8's
   derivation has a target the reader remembers. **Scaffolding note (audit finding M5):** the reader has met
   $e^{-x}$ only *inside* the Gaussian PDF (T3 Ch 3), never as a standalone decay law. So gloss $e^{-d}$ here
   as intuition only — "a curve that starts at 1 and decays smoothly toward 0 as the distance $d$ grows" — and
   explicitly defer the exponential *distribution* (rate $\lambda$, mean $1/\lambda$) to §8. Don't let a bare
   $e^{-d}$ read as an undefined object. *Cell:* optional one-liner plotting $e^{-d}$; no inference.

4. **The framework, formally.** *Now* safe, because §2 made "hypothesis = set" concrete on two sets. Posit a
   hypothesis space of candidate sets; a *feature* ("has stripes") is a hypothesis too (it carves out a set).
   **Notation lock-in** (per CLAUDE.md), each tied back to §2 so no symbol is abstract: $h$ ("like 'blue' was,
   but now it can overlap others"), $\mathcal{H}$ (calligraphic — the *list of candidate sets*; new symbol,
   say so), $X=\{x_1,\dots,x_n\}$, $y$ (a novel item to judge), and **$C$ — the unknown true set the property
   actually picks out; our whole job is to predict the event $y\in C$ from $X$** (audit finding m2: define $C$
   relative to $y$ and $h$ here, or §5's $p(y\in C\mid X)$ is unreadable — $C$ is *not* one of the $h$; it's
   the truth the $h$'s are guesses at). Three ingredients, each labeled "same as Ch 4, new object": prior
   $p(h)$, likelihood $p(X\mid h)$, posterior $p(h\mid X)\propto p(X\mid h)\,p(h)$. **"The hypothesis space IS
   a prior"** — anything not in $\mathcal{H}$ has $p(h)=0$ (NFL seed for §9). **Bind the indicator symbol HERE
   (audit finding B1 — BLOCKER):** the student has never seen $\mathbf{1}[\cdot]$ notation (T1 used set-builder
   `{ω : f(ω)=1}`, never the bracket). Introduce $\mathbf{1}[\,y\in h\,]$ in this section, *defined as exactly
   the 0/1 entry of the membership matrix* ("does row $h$ have a 1 in column $y$?"), so that when it lands
   inside a Σ in §5 it is already a known symbol, not a new one buried in a dim caption. *GenJAX cell:* the
   six-animals $\mathcal{H}$ as a binary matrix (Cow, Dolphin, Chicken, Seal, Penguin, Bat), uniform prior —
   Assignment 2 Problem 1's exact object. Scaffold the matrix: "each row is one hypothesis-set as a 0/1
   membership vector — the same 0/1 membership you used to define an event in T1, just stacked into rows."
   **Introduce `jax.vmap`-over-a-matrix HERE, not in §5 (audit finding M3):** the reader's only vmap experience
   is "run a model once per PRNG key"; mapping over *rows of a static array* (no key) is a new mental model.
   Show one trivial example in this section — e.g. `jax.vmap(jnp.sum)(H)` returns the size $|h|$ of every
   hypothesis at once — so that by §5 the *only* new idiom is enumeration-as-inference, not vmap **and**
   enumeration at once.

5. **Generalization = a posterior-weighted vote.** The chapter equation:
   $$p(y\in C\mid X)=\sum_{h\in\mathcal{H}}\mathbf{1}[\,y\in h\,]\,p(h\mid X).$$
   $\mathbf{1}[\,y\in h\,]$ and $C$ were both bound in §4, so the equation now contains *no* unintroduced
   symbol. Plain-English reading *first* ("every hypothesis votes; its vote is its posterior weight; it votes
   yes iff it contains $y$"), then the Σ. The **single-datum build-up** from the deck. **Audit finding M2 — fix
   the forward reference:** the deck's `tg_vote*.png` figures use 1-D *interval* hypotheses (and label bar
   thickness $1/|h|$), but intervals are §8 and $|h|$ is §6 — both undefined here. Two options, **(a)
   preferred:** lead the build-up with the **six-animals discrete** $\mathcal{H}$ from §4 (already defined) —
   walk the vote for one observed animal, greying hypotheses that exclude the candidate $y$ — and move the
   interval `tg_vote*.png` figures to §8 where intervals live. **(b) fallback:** if the interval figures must
   lead, pre-define in one line at the top of §5 "an interval hypothesis = the set of all points between two
   endpoints; its size is its length," and drop the $1/|h|$ thickness label until §6. Do NOT show an
   interval-with-$1/|h|$-thickness figure before either concept exists. *GenJAX cell:* the six-animals vote by
   **enumeration** — and this is now the *only* new idiom in the cell, because vmap-over-a-matrix was moved to
   §4 (M3). Introduce enumeration explicitly: "because $\mathcal{H}$ is a finite list, we don't need sampling —
   score every hypothesis and normalize; that's exact." Reuse the §4 vmap to get all posteriors, then sum the
   indicator-weighted posterior per animal $y$ (Assignment 2 Problem 4). Two-line callback: "the
   `generate`+weights sampler from §2 gives the *same* answer, approximately — enumeration is the
   finite-$\mathcal{H}$ shortcut"; optionally show both agree (using the obs-second `generate` order fixed in
   §2 — see M1).

6. **Weak vs. strong sampling + the size principle.** Frame as "we've always had a likelihood $p(X\mid h)$;
   now we ask what it should *be* when $h$ is a set." Weak: $p(X\mid h)=1$ if all $x_i\in h$ else $0$
   (size-blind). Strong: each $x$ uniform from $h$, $p(x\mid h)=1/|h|$, $p(X\mid h)=(1/|h|)^n$. Define $|h|$
   (set size — and note §4's `jax.vmap(jnp.sum)(H)` already computed it); reuse $n$ (the example-count from
   Ch 4). **The size principle:** under strong sampling smaller hypotheses get exponentially higher likelihood
   as $n$ grows. **Suspicious coincidence** intuition ($\{60,80,10,30\}$ under "even" vs "multiples of 10").
   *GenJAX cell (audit finding M4 — make the prose match the code):* `log_likelihood(h, X, sampling)`. The
   genuinely **new** idiom here is **branching on a Python bool (`strong`) outside the traced path to choose a
   whole formula** — that is the one thing to flag ("first time you switch the *model's math* with a plain
   Python `if`, not a `jnp.where`"). Do **not** advertise `jnp.where` as doing something new: in the code
   `jnp.where` is still doing its familiar Ch-5 job (pick a per-element value, here $-\log|h|$ vs $-\infty$),
   and the formula switch is the surrounding Python `if`. Run `posterior(X, sampling)` (enumerate+normalize) on
   the six animals weak and strong, print both (Assignment 2 Problem 3).

7. **On-ramp 2 — the number game (discrete H; size principle by the numbers).** Tenenbaum's number game:
   concept = a set of numbers 1–100; observe "yes" examples, judge novel numbers. Phenomenon first
   (show-before-tell): $\{60\}$ → diffuse; $\{60,80,10,30\}$ → "multiples of 10"; $\{60,52,57,55\}$ → "near
   60". Two things to explain (graded vs. rule-like; few examples). Discrete $\mathcal{H}$ = math properties
   (~24) ∪ magnitude intervals. **Size-principle arithmetic, verbatim from the deck:**
   $p(60\mid\text{mult-2})=1/50$ vs $p(60\mid\text{mult-10})=1/10$ (5×); four examples → $(1/50)^4$ vs
   $(1/10)^4$ → $625\times$ → a *rule*. Then the **two-hypothesis posterior** ($\{$mult-10, even$\}$, flat
   prior): strong $0.83/0.17$ then $0.998/0.002$; weak $0.5/0.5$ with the eliminate-but-can't-rank nuance.
   **Scaffolding callback:** this is the SAME two-set move as the §2 bridge, now with numbers and the size
   principle switched on — say so, so the reader sees the bridge was the number game in miniature. *Figures:*
   `suspicious_strong_1.png`, `suspicious_strong_4.png`, `suspicious_weak.png`. *GenJAX cell:*
   `number_game_posterior(X, sampling)` over the two-hypothesis space **reproducing the deck's exact numbers**
   (0.83/0.17, 0.998/0.002, 0.5/0.5) — the "code reproduces the figure" requirement here.

8. **On-ramp 3 — the rectangle game (continuous H; Shepard's law derived).** Open by signposting the
   discrete→continuous step *and* the T3-Ch-2 inversion: "last tutorial, continuous was the hard case; here we
   did discrete first because we could list $\mathcal{H}$. Now $\mathcal{H}$ goes continuous — infinitely many
   intervals — but the *equation doesn't change*, only the sum becomes an integral that enumeration-on-a-grid
   handles for us." Same equation, $\mathcal{H}$ = intervals (1-D) then axis-aligned rectangles (2-D). 1-D
   first: $[\ell,u]$, strong-sampling likelihood $\propto(1/(u-\ell))^n$, size = length ("$|h|$ is now a
   length, not a count"). The **generalization gradient** built from votes — flat over data, decaying outside,
   and the decay is *exponential*: **this is Shepard's law from §3, now derived, not assumed** (close the loop
   the chapter opened). Effect of $n$. Then 2-D rectangles, $r$, $d$, Tenenbaum's (1999) experiment: human vs.
   model. **Flat-prior model over-extends** (`tg_results.png`); add an **exponential prior** over size,
   $p(s)=\lambda e^{-\lambda s}$. **Scaffolding note (kept from original):** this is the **first appearance of
   the exponential *distribution*** (the reader has seen $e^{-x}$ inside the Gaussian PDF, never the
   exponential as its own distribution), so define it fully — rate $\lambda>0$, mean $1/\lambda$, support
   $s\ge0$, monotonically decreasing — with the same "define before use" care T3 Ch 3 used for the Gaussian.
   Model **bends onto the human data** ($\sigma=5$; `tg_results_prior.png`). Lesson: likelihood (size
   principle) + prior together, neither alone. *Figures:* `cc_1d.png`, `cc_1d_gradient.png`, `cc_2d.png`,
   `cc_exp_prior.png`, `tg_results.png`, `tg_results_prior.png`. *GenJAX/JAX cell (headline figure
   reproduction):* the 1-D interval learner — **regenerate `cc_1d_gradient.png`** by enumerating intervals on a
   grid, weighting by strong-sampling posterior, summing the indicator vote across a grid of $y$, plotting; the
   exponential gradient *emerges from code*. (Optional second cell: the $d$-vs-$r$ curve with/without the
   exponential prior, reproducing the bend.) **Scaffolding note:** the grid-enumeration is "the same
   enumerate-and-normalize from §5, now over a grid of intervals instead of a list of animal-sets" — one-line
   bridge so the continuous code reads as a small step, not a new method.

9. **No Free Lunch (the capstone).** Wolpert (1996), concretely (the deck's mirror-world bit-prediction
   argument): predict $x_3$ from $0,1$; worlds $0,1,0$ and $0,1,1$ are equally possible; every rule is right in
   one and wrong in its mirror, so averaged over all worlds every learner scores exactly $1/2$. The point:
   generalization is impossible without a **non-flat prior** — callback to "the hypothesis space IS a prior"
   (§4). Connect to **Assignment 2 Problem 5 ("break your model")**: expanding $\mathcal{H}$ to all $2^6-1=63$
   hypotheses with a uniform prior makes the predictive collapse — under weak sampling to exactly $1/2$ for
   unobserved animals; this *is* NFL in the assignment. *Scaffolding note:* this also pays off the §4 "H IS a
   prior" seed — the reader planted the idea, watched it run the games, and now sees *why* it was load-bearing.
   *GenJAX/JAX cell:* generate the full $2^6-1$ hypothesis space with `itertools.product`, recompute the
   six-animals predictive, and show the collapse — the exact computation Problem 5 asks for.

10. **Summary + practice.** Key-takeaways notice box (one equation, two sampling assumptions, size principle,
   prior is unavoidable). Practice problems mirroring the assignment's structure but *not* identical (so the
   chapter supports the assignment without solving it). A short "what's next" pointer: a hierarchical learner
   can *learn* the prior NFL says you need (forward pointer to Ch 12), and structured models that say how
   variables depend on each other are coming (dim pointer to Ch 8).

**Excluded (privacy):** no Tenenbaum & Xu (2000), no dax/Dalmatian word-learning, no student-presenter
framing. Where the deck bridged to that, §9's six-animals "break your model" + NFL carries the same load.

**GenJAX skeleton (the enumeration core — §5/§6, runs on the six animals):**

```python
import jax
import jax.numpy as jnp
import jax.random as jr
from genjax import gen, categorical

# Six animals, columns in this order:
# 0 Cow, 1 Dolphin, 2 Chicken, 3 Seal, 4 Penguin, 5 Bat
# Each ROW of H is a hypothesis: 1 if that animal has the property, else 0.
H = jnp.array([
    [1, 1, 1, 1, 1, 1],   # catch-all (required by Assignment 2 Problem 1)
    [1, 0, 0, 1, 0, 1],   # mammals (cow, seal, bat)  -- illustrative
    [0, 1, 0, 1, 0, 0],   # lives in water (dolphin, seal)
    [0, 0, 1, 0, 1, 0],   # has wings/feathers (chicken, penguin)
    [1, 1, 0, 1, 0, 1],   # warm-blooded non-bird ...
], dtype=jnp.float32)

prior = jnp.ones(H.shape[0]) / H.shape[0]          # uniform (Problem 2)

def log_likelihood(h, x_idxs, strong):
    """log p(X | h). strong is a Python bool chosen OUTSIDE the traced path."""
    in_h = h[x_idxs]                                # 1 if animal in h else 0
    size = h.sum()
    # jnp.where here does its FAMILIAR Ch-5 job: pick a per-element value (log 1/|h| vs log 0).
    per_x_strong = jnp.where(in_h > 0, -jnp.log(size), -jnp.inf)
    per_x_weak   = jnp.where(in_h > 0, 0.0,            -jnp.inf)
    # The NEW idiom (flag this for the reader): switching the whole FORMULA with a plain Python
    # `if` on `strong` — legal because `strong` is a Python bool, decided outside the traced path.
    per_x = per_x_strong if strong else per_x_weak
    return per_x.sum()

def posterior(x_idxs, strong, hyp=H, pri=prior):
    logp = jnp.log(pri) + jax.vmap(lambda h: log_likelihood(h, x_idxs, strong))(hyp)
    logp -= logp.max()
    p = jnp.exp(logp)
    return p / p.sum()

def predictive(x_idxs, strong, hyp=H, pri=prior):
    """p(animal y has property | X) for every y -- Assignment 2 Problem 4."""
    post = posterior(x_idxs, strong, hyp, pri)      # (n_hyp,)
    return post @ hyp                               # sum_h 1[y in h] p(h|X), per column

# Example: observe that the Seal (idx 3) has the property.
print("strong:", predictive(jnp.array([3]), strong=True))
print("weak:  ", predictive(jnp.array([3]), strong=False))
```

The `@gen`/`categorical` generative version (sample a hypothesis index from the prior, then an animal uniformly
from inside it) appears in §2 as the *generative story* on the two-hypothesis bridge; §5–§6 use the enumeration
above because $\mathcal{H}$ is finite (exact, and it mirrors what the assignment stencil does). Show the two
agree.
Verify `gen`, `categorical`, `simulate`, `ChoiceMap.d`, `generate` against GenJAX 0.10.3 before finalizing —
the API has moved; `05_mixture_models.md` and `genjax/06_building_models.md` show the current calls.

---

# Chapter 12 — Hierarchical Bayes

**Goal.** Priors have priors: hyperparameters, partial pooling, the Beta-Binomial hierarchy, shrinkage. The
conceptual hook is **Ch 7's NFL result** — "a learner needs a prior; a hierarchical learner can *learn* one."
Parallels Week-4 Block 10 (both the teaser and the full Beta-Binomial variant). Sits at the end of T3 after
the Bayes-net spine, so its DAG cross-references point *back* to Ch 8 as dim pointers.

**Opening Chibany scenario (the deck's, verbatim spirit).**

> Chibany tracks how often each student brings tonkatsu vs. hamburger. Some students have brought many bentos
> (lots of data), others only one or two (almost none). For the heavy bringers Chibany can confidently say
> "Alyssa is about a 70%-tonkatsu person." For the one-bento bringers the estimate is hopeless — one tonkatsu
> means 100%, one hamburger means 0%, and neither is believable. **Could Chibany use what the *other* students
> do to sharpen the guess for the light-data ones?**

## Scaffolding analysis for Ch 12 (what the student arrives with)

By Ch 12 the reader has Ch 7 *and* (in textbook order) the Bayes-net spine Ch 8–11. But because Ch 12 is
written before the spine exists, **assume only Ch 7 and T3 Ch 1–6 as hard prerequisites**; treat Ch 8–11 as
dim pointers. Concretely the reader brings:
- **Beta-Binomial conjugacy** is *partially* there: T3 Ch 4 taught Gaussian-Gaussian conjugacy and the
  precision-weighted "posterior is a compromise between prior and data" intuition — the perfect anchor for
  shrinkage. But the **Beta distribution itself is NOT reliably known**: it appears only in T3 Ch 6 (DPMM
  stick-breaking), which many readers skip. **Ch 12 must define Beta($a,b$) from scratch** (support $[0,1]$,
  mean $a/(a+b)$, "a soft count of $a$ prior tonkatsu and $b$ prior hamburger") — same define-before-use care
  as Ch 7's exponential.
- **The single-rate Bernoulli/flip model** is solid (T2 throughout, T3 Ch 4 "learn μ"). Hierarchy is "the
  Ch 4 learn-a-parameter move, done for many students at once, with the parameters sharing a prior."
- **Importance sampling + weights** is known (T2 Ch 4, T3 Ch 4/5). Ch 12's inference-over-$(a,b)$ reuses it;
  no new inference machinery.
- The reader does NOT know plate notation or DAGs yet (those are Ch 8). So §2's diagram must be introduced
  gently as "a picture of who-depends-on-whom," not assume graphical-model fluency.

Design rule: Ch 12 = "Ch 7 said you *need* a prior; Ch 4 showed how to *learn one parameter*; now learn the
*prior itself* from many parameters at once." Every section ties to one of those two known moves.

---

### Section plan (each opens by naming what is REUSED / NEW; then narrative/math → runnable GenJAX cell)

1. **Two extremes that both feel wrong** — the deck's three-way framing. *No pooling:* separate $\theta_i$,
   unrelated → absurd estimates for light-data students (one tonkatsu ⇒ 100%). *Complete pooling:* one shared
   $\theta$ → throws away real differences. *Hierarchical (partial pooling):* the principled middle. *Anchor:*
   this is the same "prior vs. data compromise" the reader met as precision-weighting in T3 Ch 4 — name it.
   *GenJAX cell:* simulate six students' raw fractions $k_i/n_i$ with wildly different $n_i$ to show the
   no-pooling pathology concretely (uses only `flip`/`simulate`).

1b. **Define the Beta distribution (NEW — define-before-use).** Brief dedicated beat before the generative
   process: Beta($a,b$) lives on $[0,1]$, mean $a/(a+b)$; read $(a,b)$ as "a soft count of $a$ prior tonkatsu
   and $b$ prior hamburger." Show three shapes (uniform $a=b=1$, peaked, skewed). This is the symbol the rest
   of the chapter leans on; the reader has not reliably met it. *Cell:* plot three Beta densities.

2. **The hierarchical generative process.** $(a,b)\sim\text{prior}$; $\theta_i\mid a,b\sim\text{Beta}(a,b)$;
   $k_i\mid\theta_i\sim\text{Binomial}(n_i,\theta_i)$. Define $k_i$ (tonkatsu count), $n_i$ (that student's
   bentos). Mermaid plate diagram: $(a,b)\to\theta_i\to k_i$, $\theta$ plate size $J$, $k$ plate size $N_j$.
   *GenJAX cell:* a `@gen` `student_tonkatsu` model — forward-simulate a population, show the empirical
   $\theta$'s match the Beta. (Use `jax.vmap`, not Python loops, in the final version.)

3. **Partial pooling / shrinkage, in pictures.** Posterior over $\theta_j$ for a heavy bringer (e.g. 70/100),
   a moderate (3/5), a light (1/2): the light one is **pulled toward the population mean**, the heavy one
   barely moves. This is the payoff plot — "automatically borrowing strength." *GenJAX cell:* the Beta-Binomial
   conjugate posterior for fixed $(a,b)$ (closed form), plotted as raw fraction vs. posterior mean per student
   — **regenerate a shrinkage figure** (this is the chapter's "code reproduces a figure" requirement; the deck
   describes it as Slide B7 but ships no PNG, so make one).

4. **Where does the prior on $\theta$ come from?** Zoom out: $(a,b)$ are themselves inferred from a
   weakly-informative hyperprior (one default, e.g. a broad prior on $a+b$; state that the specific choice is
   research-active and move on). "The prior has its own prior" is coherent, not infinite regress. *GenJAX
   cell:* `@gen` model with $(a,b)$ latent + importance sampling over $(a,b)$ — the gentlest intro to
   inference-over-hyperparameters. Honest note (as in the deck and the Ch 5 importance-sampling caveat):
   importance sampling here is noisy; that's the point, and smarter inference is a later topic.

5. **The NFL connection (the hook).** Explicit callback to **Ch 7's No Free Lunch**: NFL proves a learner
   needs inductive bias; the hierarchy is where a learner **acquires** it from data instead of being born with
   it. This is the chapter's reason to exist. Mention **overhypotheses** (Kemp, Perfors & Tenenbaum, 2007) in
   one paragraph — shape bias, object-vs-substance — as *second-level* hypotheses; name-drop, don't derive.

6. **Connections (dim pointers).** One paragraph each, not re-derived: the DPMM (Ch 6) is a hierarchical model
   whose hyperparameter is the DP concentration; the DAG in §2 *is* a Bayes net (back-pointer to Ch 8, written
   so it reads fine whether or not Ch 8 exists yet).

7. **Summary + practice.** Takeaways box; a partial-pooling practice problem with a GenJAX stencil.

**GenJAX skeleton (forward simulation, §2 — rewrite with `vmap` for the final):**

```python
import jax
import jax.numpy as jnp
import jax.random as jr
from genjax import gen, beta, flip

@gen
def student_tonkatsu(a, b, n):
    """One student: draw a rate from Beta(a,b), then n bento outcomes."""
    theta = beta(a, b) @ "theta"
    # n bentos; flip(theta) is Bernoulli. Final version vmaps this instead of looping.
    outcomes = jnp.array([flip(theta) @ f"y_{i}" for i in range(n)])
    return theta, outcomes.sum()
```

Verify `beta`, `flip`, `Binomial`/`binom` availability and the conjugate-update API against GenJAX 0.10.3
before finalizing. If `Binomial` isn't a primitive, model each bento as a `flip(theta)` and sum, as above.

---

## Style rules (unchanged from the sibling plan; restated for self-containment)

1. **Frontmatter** — `+++` with `date = "2026-05-30"` (today, per textbook CLAUDE.md), `title`, `weight = N`.
2. **Voice** — Chibany scenario first, math second, GenJAX woven through (not relegated to one closing block).
3. **Math** — `$$...$$` block, `$...$` inline (dollar-sign math; the Hugo theme uses KaTeX).
4. **Notice boxes** — `{{% notice style="info|warning|success|tip" title="…" %}}`; misconception boxes and
   key-takeaways boxes per the Ch 1 template.
5. **≥1 mermaid diagram per chapter** (Ch 7: the framework as prior→likelihood→posterior→vote flow, or the
   strong-vs-weak generative processes; Ch 12: the $(a,b)\to\theta_i\to k_i$ plate DAG).
6. **GenJAX code MUST validate** — `python validate_code_blocks.py` — and **must execute** against GenJAX
   0.10.3 (default env or `.ch5_test_venv`). Use `jnp.where` for array-conditionals; only branch on Python
   bools (like the `strong` flag) outside the traced path.
7. **Reproduce a figure in code** — at least one GenJAX/JAX cell per chapter regenerates a lecture figure (Ch
   7: the 1-D gradient and/or the six-animals predictive and the NFL collapse; Ch 12: the shrinkage plot).
8. **Forward/backward references** — Ch 7 opens referencing Ch 4; Ch 12 opens referencing Ch 7's NFL. Dim,
   short forward pointers. Update `content/intro2/_index.md` (date, sidebar listing, learning-path mermaid) when
   chapters land — add Ch 7 and Ch 12, leaving 8–11 as a documented gap until the spine is written.
9. **JPCCA acknowledgment** at the bottom of every new page (match `_index.md`).
10. **Privacy** — no Shohei materials anywhere (no Tenenbaum & Xu, no dax/Dalmatian, no presenter framing).

---

## What this plan does NOT cover

- **Specific hyperprior choice** for Ch 12 — give one weakly-informative default, note it's research-active.
- **Exchangeability theorems / empirical Bayes / MCMC** — one sentence each at most; the tools here are exact
  enumeration (Ch 7, finite $\mathcal{H}$) and importance sampling (Ch 12). Forward-point MCMC, don't develop.
- **Renumbering the sibling spine plan** — owed before any spine chapter is committed; not this task.
- **Writing Ch 8–11** — sibling plan's job.

---

## Suggested order of operations for the writing agent

1. Re-read `textbook/CLAUDE.md`, `CODE_VALIDATION.md`, `01_mystery_bentos.md`, `05_mixture_models.md`,
   `genjax/06_building_models.md`. Skim the built `week4-slides.qmd` for exact wording/numbers/figures.
2. Confirm the GenJAX 0.10.3 API (`gen`, `categorical`, `beta`, `flip`, `simulate`, `ChoiceMap.d`,
   `generate`) by running a one-cell smoke test before writing any code block.
3. Copy the Ch-7 figures listed above into `textbook/content/intro2/images/`.
4. Write **Ch 7** section by section, each with its runnable GenJAX cell; build the figure-reproduction cells
   against the six-animals model and the 1-D interval learner. Run `validate_code_blocks.py` and *execute* the
   cells. **Stop for Prof. Austerweil's review before Ch 12.**
5. Write **Ch 12**; validate + execute; build the shrinkage figure.
6. Update `_index.md` (date, sidebar, mermaid) — add 7 and 12, document the 8–11 gap.
7. Confirm: no Shohei materials; dates set to today; nothing under `archive/` touched; assignment link
   (`…/intro2/07_generalization/`) now resolves; `generalization.tex`'s "in preparation" note can later be
   updated to point at the live chapter.

---

## Funding acknowledgment (bottom of every new page)

```markdown
Special thanks to [JPCCA](https://jpcca.org/) for their generous support of this tutorial series.
```

— Revised by Claude (Opus 4.8) with Prof. Austerweil, 2026-05-30.
