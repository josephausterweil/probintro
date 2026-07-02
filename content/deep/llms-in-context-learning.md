+++
date = "2026-07-02"
title = "LLMs & In-Context Learning"
weight = 40
tracks = ["beginner", "cogsci"]
+++

{{% notice style="warning" title="Chapter in preparation" %}}
This chapter accompanies Week 11 of the course and is being written now.
{{% /notice %}}

**Premise.** The campus chatbot guesses Chibany's favorite from three
examples — no retraining, no gradient steps. This chapter makes the case that
**in-context learning is implicit hierarchical Bayesian inference**: the
prompt plays the role of the prior, the examples are observations, and the
model's pretraining distribution is the hyperprior — a direct callback to
[Hierarchical Bayes](../../structure/hierarchical-bayes/). The CogSci payoff
chapter of this Part.

{{< lecture week=11 label="LLMs and in-context learning" >}}

---

*This project is generously funded by the Japanese Probabilistic Computing
Consortium Association (JPCCA).*
