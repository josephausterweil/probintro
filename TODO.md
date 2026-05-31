# TODO — Probability & Probabilistic Computing Textbook (probintro)

Future work and open questions for the Hugo textbook. Keep entries short; link to the
relevant `content/` path or plan doc.

## Structure / organization

- [ ] **Reconsider the overall organization of the tutorial material — Tutorial 3 (`content/intro2/`)
      has become packed relative to the others.** As of 2026-05-31, Tutorial 3 carries Chapters 1–7
      (continuous probability → Gaussians → Bayesian learning → mixtures → DPMM → Bayesian
      generalization), and Ch7 alone grew large enough that it was split into a 4-part page bundle
      (`content/intro2/07_generalization/`). With the Bayes-net spine (Ch8–11) and Hierarchical Bayes
      (Ch12) still to land, Tutorial 3 will be far heavier than Tutorials 1 and 2.
      Decide on the best top-level organization before adding more — options to weigh:
    - Split Tutorial 3 into two tutorials (e.g. "Continuous Probability & Bayesian Learning" =
      Ch1–6, and a new "Models & Structured Inference" tutorial = generalization + Bayes nets +
      hierarchical Bayes).
    - Keep one Tutorial 3 but lean harder on page bundles so each heavy chapter is a navigable group
      (as Ch7 now is), and make sure the sidebar weights/ordering stay coherent across ~12 chapters.
    - Rebalance content *between* tutorials (e.g. move a foundational continuous-probability chapter
      earlier) so no single tutorial dominates.
      Cross-refs to keep consistent if the numbering changes: the `_index.md` learning-path mermaid,
      every `§N` / "Chapter N" cross-reference, the reserved 08–11 numbering for the Bayes-net spine
      (see `CHIBANY_T3_GENERALIZATION_PLAN.md`), and the assignment ↔ chapter mapping.
