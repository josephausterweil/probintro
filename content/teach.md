+++
date = "2026-07-16"
title = "Teach With This Book"
linkTitle = "Teach With This Book"
weight = 8
+++

This book is free, open, and built to be taught from. If you run a course in
probability, probabilistic machine learning, computational cognitive science,
or probabilistic programming, you are welcome to adopt any part of it — whole,
or a Part at a time. This page is the orientation for instructors.

## What it is

A single narrative textbook that carries a reader from **counting** all the
way to **LLMs and alignment**, using one continuous story (Chibany, the Chiba
Tech mascot, and a year of bento mysteries) and one continuous toolkit (a
generative model, a prior, an inference). Every chapter is *scenario first,
math second, runnable code third*.

Three things distinguish it from a lecture-note PDF:

- **Runnable, validated code in every chapter.** Code cells are executed in CI
  and checked against their printed output, so what you see on the page is what
  runs. The probabilistic-programming spine is [GenJAX](genjax/) (a JAX-based
  PPL); a dedicated chapter explains [why GenJAX rather than Stan or
  PyMC](genjax/why-genjax/), honestly, including where Stan/PyMC are the better
  tool.
- **~25 interactive in-browser widgets** — conditioning by crossing out
  outcomes, an MCMC explorer, a Q-learning gridworld, a POMDP belief updater, a
  matrix-transform / ReLU-fold visualizer, an attention lookup, and more —
  driven live in lecture or explored by students.
- **A Colab capstone per chapter**, modeled on real assignments and papers
  (e.g. the in-context-learning chapter's capstone runs *both sides* of the
  "is ICL Bayesian?" debate — the Xie/Ye exact-Bayes toy and the Falck
  martingale test on it).

It also has an **in-page English ⇄ Japanese toggle** (press `L`) — the concept
chapters are translated, which some students genuinely need.

## Two reading tracks

The same book serves two audiences; a sidebar switch dims off-track chapters
and each chapter's footer points to the next stop on your track.

- **🤖 Beginner** — probability theory → probabilistic ML → probabilistic
  computing, through to neural networks, LLMs, and ML ethics, from a light
  background. [Roadmap](start/beginner/).
- **🧠 CogSci** — Bayesian cognitive science: probabilistic models as theories
  of minds (generalization, memory search, sampling hypotheses, goal
  inference, theory of mind). [Roadmap](start/cogsci/).

## The nine Parts

| Part | Covers | Capstones anchor on |
|---|---|---|
| **I. [Foundations](foundations/)** | events, counting, conditioning, Bayes' rule | counting the outcome space |
| **II. [The Tools (GenJAX)](genjax/)** | simulate, trace, condition, infer in a PPL | the taxicab problem in code |
| **III. [Continuous & Bayesian Learning](learning/)** | densities, Gaussians, conjugacy, mixtures | Gaussian conjugate updating |
| **IV. [Structure](structure/)** | generalization, Bayes nets, causality, information, hierarchies | Tenenbaum & Griffiths, causal inference |
| **V. [Chains, Walks & Sampling](sampling/)** | Markov chains, random walks, MC, particle filters, MCMC | Sanborn & Griffiths MCMC-with-people |
| **VI. [Decisions & RL](decisions/)** | decision theory, MDPs, Q-learning, inverse RL, POMDPs | Baker goal inference, Daw two-step |
| **VII. [Model Complexity](complexity/)** | bias–variance, Bayesian nonparametrics, GPs | double descent, DPMM |
| **VIII. [Deep Networks](deep/)** | vectors→nets→transformers→LLMs, in-context learning as hierarchical Bayes | FGSM, attention, the ICL debate |
| **IX. [Ethics, Fairness & Safety](ethics/)** | adversarial examples, fairness formalisms, bias, alignment | the fairness impossibility, RLHF reward hacking |

Part I assumes **no math background** and Part VIII opens with a
[linear-algebra-from-zero chapter](deep/vectors-and-spaces/), so the beginner
track is genuinely self-contained.

## Adopting it in a course

The book grew alongside a 12-week graduate course, **[Human and Machine
Learning](https://henkaku-center.github.io/hml/)** (Chiba Institute of
Technology, School of Design & Science), so it maps cleanly onto a semester —
roughly one Part every one to two weeks. The course homepage has the full
schedule, slide decks, readings, and assignments — a worked example of
teaching from this book. Where a chapter parallels a lecture, its closing
"From the lecture" box links the week's deck, and the in-chapter self-check
quizzes are the same commit-before-reveal polls used live.

Mix and match freely: a one-term probabilistic-ML course can run Parts I–III
then VII–IX; a cognitive-modeling seminar can run the CogSci track; a single
guest lecture can lift one chapter and its widget.

## Coming from probmods?

If you have taught with [Probabilistic Models of Cognition](https://probmods.org/),
much of the Bayesian-cognitive-science material here covers the same ground —
generalization, sampling as a process model, structured inference — but with
runnable GenJAX (rather than WebPPL), interactive widgets, and a from-zero
on-ramp. The [CogSci track](start/cogsci/) is the closest map.

## Feedback and reuse

The book is **[CC-BY-4.0](https://github.com/josephausterweil/probintro)** — reuse,
remix, and adapt with attribution; a [`CITATION.cff`](https://github.com/josephausterweil/probintro/blob/main/CITATION.cff)
is included. If you adopt it, find an error, or want a topic covered, please
reach out to Joe — feedback from other instructors is exactly what the next
revision is for.

---

*This project is generously funded by the Japanese Probabilistic Computing
Consortium Association ([JPCCA](https://jpcca.org/)).*
