+++
date = "2025-11-11"
title = "Continuous Probability and Bayesian Learning"
weight = 3
toc = true
+++

## From Discrete to Continuous with GenJAX

In the first two tutorials, you learned:
1. **Tutorial 1**: Probability theory using discrete outcomes (sets and counting)
2. **Tutorial 2**: How to express probability in code using GenJAX

Now you'll learn to work with **continuous probability distributions** and perform **Bayesian learning** on real-valued data, all using the GenJAX tools you've already learned!

![Chibany with scale](images/chibanylayingdown.png)

## The New Challenge

In Tutorial 1, Chibany's lunch choices were **discrete**: tonkatsu OR hamburger. But what if we want to model **continuous measurements** like:

- The **weight** of his bento box
- The **temperature** of his office
- The **time** it takes students to arrive

These aren't discrete choices. They're **continuous values** that can fall anywhere on a number line!

## What You'll Learn

This tutorial builds directly on GenJAX (Tutorial 2) to explore:

### Chapter 1: Continuous Random Variables
- Probability density functions (PDFs)
- Cumulative distribution functions (CDFs)
- The uniform distribution
- **GenJAX**: Sampling from and conditioning on continuous distributions

### Chapter 2: The Gaussian Distribution
- The bell curve and its properties
- Mean and variance parameters
- The 68-95-99.7 rule
- **GenJAX**: Working with Normal distributions

### Chapter 3: Bayesian Learning with Gaussians
- Prior beliefs about parameters
- Updating beliefs with data (conjugate priors)
- Posterior and predictive distributions
- **GenJAX**: Implementing Gaussian-Gaussian models

### Chapter 4: Mixture Models
- Combining multiple distributions
- Gaussian Mixture Models (GMMs)
- Clustering with mixtures
- **GenJAX**: Building and inferring mixture models

### Chapter 5: Dirichlet Process Mixture Models
- Infinite mixture models
- The Dirichlet Process prior
- Automatic model selection
- **GenJAX**: Implementing DPMM for clustering

## Prerequisites

**Required:**
- ✅ Completed "A Narrative Introduction to Probability" (Tutorial 1)
- ✅ Completed "Probabilistic Programming with GenJAX" (Tutorial 2)
- ✅ Comfortable writing and running GenJAX generative functions
- ✅ Understand traces, conditioning, and inference in GenJAX

**Not Required:**
- ❌ Calculus (we provide intuition and use GenJAX for computation)
- ❌ Advanced statistics
- ❌ Mathematical proofs

## Learning Philosophy

**You already know** how to think probabilistically (Tutorial 1) and how to express probability in GenJAX code (Tutorial 2). This tutorial shows you how those same ideas extend to the continuous case!

**Key insight:** Moving from discrete to continuous isn't about learning entirely new concepts. It's about adapting what you know:
- **Probabilities** become **probability densities**
- **Sums** become **integrals** (but GenJAX handles this for you!)
- **Counting** becomes **measuring area under curves**

## What Makes This Tutorial Different

Unlike traditional probability courses, we:

1. **Use GenJAX throughout**: Every concept is illustrated with runnable code
2. **Leverage simulation**: When math gets complex, we approximate with samples
3. **Build intuition first**: Visual understanding before mathematical details
4. **Connect to Tutorial 1**: Every concept links back to discrete probability
5. **Interactive notebooks**: Adjust parameters and see results update live

## Why Continuous Probability Matters

Many real-world phenomena are naturally continuous:

- **Scientific measurements**: Temperature, weight, time, distance
- **Financial data**: Stock prices, returns, volatility
- **Machine learning**: Most input features are continuous
- **Natural phenomena**: Heights, speeds, concentrations

Understanding continuous probability lets you model the real world more accurately!

## Tutorial Structure

Each chapter includes:
- 📖 **Concept explanation** connecting to discrete probability
- 💻 **GenJAX implementation** with runnable code
- 🎮 **Interactive Colab notebooks** to explore parameters
- 📊 **Visualizations** showing PDFs, posteriors, and predictions
- ✅ **Exercises** with solutions

## Ready to Begin?

Let's start by understanding what changes when we move from discrete outcomes to continuous measurements!

[Next: Chapter 1 - Continuous Random Variables →](./01_continuous.md)

---

## A Note on Mathematics

This tutorial includes some mathematical notation (PDFs, integrals, etc.), but **you don't need to be a math expert!**

**When you see an integral** (∫): Think "area under the curve" (GenJAX computes it for you)
**When you see a derivative**: Think "rate of change" (rarely needed, GenJAX handles it)
**When you see Σ vs ∫**: Think "sum vs continuous sum" (same idea, different notation)

**Focus on:**
- What the code does (run it and see!)
- What the plots show (visual intuition)
- How it connects to discrete probability (concepts you know)

Mathematics provides precise definitions, but GenJAX lets you **compute without deriving**!

---

{{% notice style="tip" title="Learning Tip" %}}
**Start with visuals, then code, then math.** When you see a new concept:

1. Look at the plots first (what does it look like?)
2. Run the GenJAX code (what happens?)
3. Read the math (why does it work?)

Understanding beats memorization. You can always reference the math later!
{{% /notice %}}

---

Special thanks to [JPPCA](https://jpcca.org/) for their generous support of this tutorial series.
