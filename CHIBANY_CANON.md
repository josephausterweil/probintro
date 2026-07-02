# CHIBANY_CANON.md — the single source of truth for the book's narrative

This document supersedes the character/style fragments scattered across
`CHIBANY_T3_CH7-10_PLAN.md`, `CHIBANY_T3_GENERALIZATION_PLAN.md`,
`CHIBANY_T3_MARKOV_NETWORKS_PLAN.md`, and `CLAUDE.md`. Those docs remain as
historical build plans; **when they disagree with this file, this file wins.**
Every chapter rewrite must be checked against this canon before it ships, and
must append its newly-established facts to the Continuity Ledger at the bottom.

---

## 1. Character sheet — Chibany

- **Who:** the beloved mascot of Chiba Institute of Technology. A mascot, not a
  person: Chibany attends no classes, holds no degrees, and is **NOT a
  professor** — they never lecture, and they never explain math to the reader.
  Chibany *does* things (weighs, counts, tallies, wanders, tastes) and *asks*
  the naive-smart question that opens every chapter.
- **Pronouns:** they/them, always. (Audit note: "his beloved tonkatsu" appears
  in some lecture speaker notes — the book must never slip.)
- **Diet & tradition:** students bring Chibany **two bentos every day** (lunch
  and dinner) — a campus tradition. Chibany loves **tonkatsu** above all;
  hamburger is the acceptable second.
- **Temperament:** curious empiricist, optimist, slightly gluttonous, patient.
  Chibany's superpower is *measuring* rather than assuming.
- **The Bento Journal:** Chibany's notebook of every bento received — dates,
  weights, contents when known, who brought it. Established in the hierarchical
  Bayes and information theory chapters; **promoted here to the book's central
  data artifact.** Whenever a chapter needs data, the data comes from (or goes
  into) the Journal. New measurements types (weights, temperatures, arrival
  times, web clicks) are new *columns*; new semesters are new *pages*.

## 2. Supporting cast (closed — no new recurring characters)

| Character | Role in the story | Role in the prose |
|---|---|---|
| **Jamal** (he/him, grad student) | pragmatic modeler, owns the laptop | writes/explains the GenJAX code; "let's just simulate it" |
| **Alyssa** (they/them, grad student) | theory-minded | supplies the *why*; states the math idea in words before the formula |
| **Tanaka-san** (he/him, Chiba Tech alum) | friendly campus fixture, brings marbles and opinions | the **voice of every misconception box**: his takes are always plausible and always subtly wrong; the chapter's job is to say *why* |
| Derek | — | **lecture-only** (appears in course decks, never in the book) |
| Ira, Yuki | — | retired; never introduce |

Dialogue rules: exposition dialogue is Jamal/Alyssa; wrong-but-reasonable takes
are Tanaka-san; Chibany owns curiosity and data, never explanation. Students
who bring bentos stay anonymous ("a student") unless the chapter's math needs
them distinguishable (then they get single-letter names that match the math
symbols, e.g. students $A$–$F$ in the shrinkage table).

## 3. World rules

- **Setting:** Chiba Tech campus — Cafeteria A and Cafeteria B, the SDS
  building, Chibany's nap spot, the health center, the campus wiki/web (Part
  V's network), and (from Part VIII) the new **bento kiosk** and the **robot
  mascot trainee**.
- **The bento rules (the continuity fix):** bentos arrive **wrapped in
  furoshiki — opaque**. Contents are unknown until opened. Any chapter where
  contents are known in advance must be *explained by the story*: from the
  Markov-chains chapter onward, students **label their bentos** (a canonized
  timeline event — see Ledger), and in flashback chapters ("last semester")
  boxes were transparent. Never silently switch between known and hidden
  contents.
- **Weights are informative:** tonkatsu bentos ≈ 500 g, hamburger ≈ 350 g
  (established in Mystery Bentos). Numbers reused across chapters must match
  the Ledger.
- **The base rate:** students bring tonkatsu ~70% of the time (the overheard
  70/30 conversation in Mystery Bentos). Chapters may perturb this *knowingly*
  ("this semester the cafeteria changed suppliers…"), never accidentally.

## 4. Timeline — one academic year, one driving mystery per Part

Each Part is an **arc** with a single driving question; its chapters are
episodes that make progress on that question. A chapter opens *inside* its
Part's arc (no cold resets), and closes by advancing it.

| Part | Season | Driving mystery | Episode logic |
|---|---|---|---|
| I Foundations | April — a new school year | *Will there be tonkatsu today?* | counting possible days → shrinking the possibilities (conditioning) → updating on evidence (Bayes) |
| II The Tools | Golden Week | Jamal lends Chibany a laptop: *can we teach it to imagine days?* | code as imagination: simulate → trace → condition → infer |
| III Continuous & Bayesian Learning | early summer | **The Mystery of the Two Peaks** — opaque bentos, a weigh-station, an average that matches no bento | weights → densities → Gaussians → learning → mixtures |
| IV Structure | rainy season | **The Bento Provenance Project** — not just *what* arrives but *the rules behind it* | which concepts generalize → what causes what → how surprising is each day → each student's true rate |
| V Chains, Walks & Sampling | summer wandering | *The question you can't sum* — Chibany wants a semester-long answer no formula gives | wander campus (chains) → the campus web (walks/PageRank) → wander memory (search) → answer by sampling (MC/PF/MCMC) → do minds sample? |
| VI Decisions & RL | autumn | **Chibany learns to act** — the health-kick arc | losses → plans (MDPs) → learning by trial (Q) → reading *others'* goals (IRL) → acting under hidden state (POMDPs/Tiger) → the frontier |
| VII Model Complexity | winter — **the Model Fair** | Chibany judges the student modeling contest: *which model is honestly best?* | overfit entries (bias-variance) → "how many bento types exist, really?" (DPMM) → the smooth-curve stall (GPs) |
| VIII Deep Networks | new year | **The kiosk & the trainee** — a photo-recognizing bento kiosk and a robot mascot trainee arrive: *how does it see?* | features → nets → attention → in-context learning → imagination (world models) |
| IX Ethics & Safety | year's end | **The Mascot's Code** — Chibany audits what the campus systems learned | can it be fooled? → is it fair? → whose priors did it inherit? → what did we actually teach the trainee? |

## 5. Voice & style rules (consolidated)

1. **Scenario first, math second, GenJAX third** — every chapter opens with a
   concrete Chibany scene *inside its Part's arc*, derives the math from the
   scene, then closes the loop in code. The narrative is not a garnish on the
   first page: each major section re-enters the scene.
2. **Chibany never explains.** Explanations belong to Alyssa (ideas), Jamal
   (code), or the narrator. Misconceptions belong to Tanaka-san.
3. ≥ 1 dark-theme mermaid diagram per chapter.
4. Math delimiters `$...$` / `$$...$$` only (goldmark passthrough handles the
   rest); never a literal prose dollar sign.
5. Cross-references are **linked chapter names, never bare numbers** — "the
   [Inverse RL](../inverse-rl/) chapter (Part VI)". Sidebar numbers are
   computed and will shift as chapters are added.
6. Every chapter ends with a **"What you can do now"** box (skills, glossary
   links) and the JPCCA funding acknowledgment.
7. **Standalone code:** the *first* code block of every chapter is a complete
   setup cell (all imports + key). Later blocks may continue the namespace,
   but any block hidden in an `{{% expand %}}` must be independently runnable
   or carry `<!-- validate: skip -->` with a one-line reason.
8. American spellings in English prose.
9. Terms defined at first use; a term returning after a long gap gets a
   one-line re-anchor, not an assumption of retention.

## 6. Retcon list (known inconsistencies → canonical resolutions)

| Inconsistency | Resolution |
|---|---|
| Bentos sometimes opaque, sometimes known in advance | Furoshiki rule (§3) + the labeled-bento shift canonized at the start of the Markov-chains chapter; earlier "known contents" scenes are last-semester flashbacks (transparent boxes) |
| The "second meal" ambiguity in the conditional-probability chapter (1/3 vs 2/3 readings) | Book-wide convention: **in any two-meal outcome pair, the first letter is lunch, the second is dinner** (stated in a convention box at first use). The famous question is asked about *dinner*; the event is $\{HT, TT\}$, the answer 2/3. The $\{TT\}$ reading survives as an explicit Tanaka-san contrast exercise |
| "Cardinality" used for weighted sums | *Cardinality* $|A|$ strictly for counts; weighted totals introduce *weight* $w(A)$ |
| Pronoun slips ("his tonkatsu") | they/them audit is part of every Part pass |
| Alyssa/Jamal appear mid-book with no introduction | Both get a proper one-line introduction in Part I (they help Chibany count) and reappear by name in every Part |

## 7. Continuity Ledger

Facts established per chapter. **Append when a rewrite lands; check before
writing.** Format: `fact — where established`.

- Tonkatsu ≈ 500 g, hamburger ≈ 350 g — Mystery Bentos (Part III).
- ~70/30 tonkatsu/hamburger base rate, overheard — Mystery Bentos (Part III).
- Two bentos/day tradition — Part I, Chibany is hungry.
- The Bento Journal exists and has per-student pages — Part IV (information
  theory & hierarchical Bayes chapters).
- Students begin labeling bentos — Part V (Markov chains opening; canonized
  retcon).
- *(append below as rewrites land)*
