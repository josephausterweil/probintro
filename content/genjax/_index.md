+++
date = "2025-11-06"
title = "Probabilistic Programming with GenJAX"
weight = 2
toc = true
+++

## From Probability Theory to Probabilistic Code

In the previous tutorial, you learned to think about probability using **sets and counting**. Chibany showed you that probability questions are really about:
1. What's possible? (Define the outcome space)
2. What am I interested in? (Define the event)
3. Count them! (Calculate the ratio)

Now you'll learn to express those same ideas in **code** using GenJAX — a probabilistic programming language that lets computers do the counting for you!

![Chibany is happy](images/chibanyplain.png)

## What is GenJAX?

GenJAX is a **probabilistic programming language** that lets you:

1. **Define generative processes** — Write code that describes how outcomes are produced
2. **Perform inference** — Find probable explanations given observations
3. **Leverage powerful computation** — Run fast on GPUs for complex problems

**The best part?** You already understand the concepts! GenJAX just translates what you learned about sets into code that computers can execute.

## No Coding Experience? No Problem!

This tutorial is designed for **complete beginners** to programming. We'll:

✅ Use **Google Colab** — Run code in your browser, no installation needed
✅ Provide **interactive notebooks** — Adjust sliders and see results change instantly
✅ Teach **just enough Python** — Only what you need to understand the code
✅ Connect **everything to sets** — Every line of code relates to concepts you know

**You won't become a programmer from this tutorial** — but you'll be able to use probabilistic programming tools to explore probability and build models!

## Two Ways to Follow Along

### Option 1: Google Colab (Recommended for Beginners)

**Pros:**
- ✅ No installation required
- ✅ Runs in your web browser
- ✅ Interactive widgets and visualizations
- ✅ Works on any computer (Windows, Mac, Linux, Chromebook)
- ✅ Free GPU access

**Cons:**
- ⚠️ Requires internet connection
- ⚠️ Sessions timeout after inactivity

**Perfect for:** Complete beginners, trying things out, classroom settings

### Option 2: Local Installation (Optional)

**Pros:**
- ✅ Works offline
- ✅ Faster for large computations
- ✅ Full control over environment

**Cons:**
- ⚠️ Requires installation and setup
- ⚠️ More technical troubleshooting needed

**Perfect for:** Those comfortable with software installation, serious projects

---

## Tutorial Structure

### Chapter 0: Getting Started
Set up your environment (Google Colab or local installation)

### Chapter 1: Python Essentials
Just enough Python to read and run GenJAX code

### Chapter 2: Your First Generative Function
Chibany's meals in code — from sets to simulation

### Chapter 3: Understanding Traces
What GenJAX records when programs run

### Chapter 4: Conditioning and Observations
How to ask "what if I know this happened?"

### Chapter 5: Inference in Action
The taxicab problem, now solved with code!

### Chapter 6: Building Your Own Models
Go beyond Chibany's meals

---

## Learning Philosophy

**You already know the concepts** from the probability tutorial. This tutorial just shows you how to:
- Express outcome spaces as generative functions
- Express events as filters on outcomes
- Let computers do the counting (simulation)
- Ask conditional probability questions (inference)

**Every chapter includes:**
- 📖 Explanation connecting to set-based probability
- 💻 Interactive Colab notebook
- 🎮 Widgets to play with parameters
- 📊 Visualizations that update automatically
- ✅ Exercises with solutions

---

## What You'll Build

By the end of this tutorial, you'll be able to:

1. **Write simple generative models** in GenJAX
2. **Run simulations** to approximate probabilities
3. **Perform inference** given observations
4. **Visualize results** with interactive plots
5. **Understand the connection** between theory and code

You'll see how the taxicab problem, Chibany's meals, and other examples from the probability tutorial can be solved computationally!

---

## Prerequisites

**Required:**
- ✅ Completed "A Narrative Introduction to Probability"
- ✅ Understand sets, events, and conditional probability
- ✅ Know what Chibany likes to eat 😊

**Not Required:**
- ❌ Programming experience
- ❌ Python knowledge
- ❌ Software installation (if using Colab)

---

## Ready to Start?

Let's set up your environment and write your first probabilistic program!

**Choose your path:**
- [Chapter 0: Getting Started with Google Colab →](./00_getting_started.md) (Recommended)
- [Chapter 0b: Local Installation →](./00b_local_install.md) (Optional)

**Or jump to Python basics:**
- [Chapter 1: Python Essentials →](./01_python_basics.md)

---

{{% notice style="tip" title="Learning Tip" %}}
**Don't try to memorize Python syntax!** Focus on understanding:
- What the code is trying to do (the purpose)
- How it connects to probability concepts (the mapping)
- What happens when you run it (the result)

You can always copy-paste and modify examples. Understanding beats memorization!
{{% /notice %}}
