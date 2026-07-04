+++
date = "2026-07-02"
title = "Foundations: Counting Lunches"
linkTitle = "Foundations"
weight = 10
book_part = true
tracks = ["beginner", "cogsci"]
aliases = ["/intro/"]
+++

**Will there be tonkatsu today?** It is April, a new school year at Chiba
Tech, and the campus mascot
[Chibany](https://admission.chibatech.ac.jp/about/chibany/) has one consuming
question about the two bentos students bring them every day. This Part builds
all of probability out of that question — by **counting**. Sets of possible
days, events as subsets, conditioning as crossing out possibilities, and
Bayes' rule as the inevitable arithmetic of updating.

![Chibany laying down](../images/chibanylayingdown.png)

Most probability courses jump straight into formulas. Here, **probability is
fancy counting**: What are all the possibilities? Which ones include tonkatsu?
What's the ratio? That perspective makes conditional probability, Bayes' rule,
and everything after them feel natural instead of mysterious — and it is
exactly the mental model that [probabilistic programming](../genjax/) runs on.
No prior math background required.

```mermaid
graph LR
    A[Chibany<br>is Hungry] --> B[Probability<br>& Counting]
    B --> C[Conditional<br>Probability]
    C --> D[Bayes'<br>Theorem]
    D --> E[Glossary]

    style B fill:#4a9eff,color:#fff
    style C fill:#4a9eff,color:#fff
    style D fill:#4a9eff,color:#fff
```

## Chapters

{{% children sort="weight" %}}

New to the book? The [Start Here](../start/) guide explains the two reading
tracks and how the Parts fit together.

---

*This project is generously funded by the Japanese Probabilistic Computing
Consortium Association ([JPCCA](https://jpcca.org/)).*
