+++
title = "Getting Started with Google Colab"
weight = 1
+++

## What is Google Colab?

Google Colab (short for "Colaboratory") is a **free online tool** that lets you write and run Python code in your web browser. Think of it like Google Docs, but for code!

**Why we love it for beginners:**
- ✅ No installation needed
- ✅ No setup required
- ✅ Pre-installed with common libraries
- ✅ Free GPU access (for fast computation)
- ✅ Easy sharing with others
- ✅ Automatic saving to Google Drive

## Step 1: Open Your First Notebook

Click this link to open the first GenJAX notebook:

**[📓 Open: Chapter 2 - Your First GenJAX Model](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/02_first_genjax_model.ipynb)**

{{% notice style="info" title="What just happened?" %}}
You opened a **Jupyter notebook** — an interactive document that mixes:
- **Text** (explanations, like this)
- **Code** (that you can run)
- **Visualizations** (graphs and plots)

Think of it as a lab notebook for probability experiments!
{{% /notice %}}

## Step 2: Make a Copy (So You Can Edit)

When the notebook opens, you'll see a yellow banner that says "You are using Colab in playground mode."

**Click: "Copy to Drive"** (top right, or in the banner)

This creates your own copy that you can edit and save!

{{% notice style="warning" title="Important!" %}}
Without copying to Drive, your changes **won't be saved** when you close the browser!
{{% /notice %}}

## Step 3: Understanding the Interface

Let me show you around:

### The Notebook Structure

A Colab notebook has **cells** — little boxes that contain either:
- **Text cells** (like this explanation)
- **Code cells** (Python code you can run)

### Running Code Cells

See a cell with code like this?

```python
print("Hello, Chibany!")
```

**To run it:**
1. Click on the cell
2. Press **Shift + Enter** (or click the ▶️ play button on the left)

Try it! You should see "Hello, Chibany!" appear below the cell.

### Interactive Widgets

Throughout the notebooks, you'll see **sliders and controls** that let you change values and instantly see updated results. We'll use these to explore probability!

## Step 4: Install GenJAX

The first code cell in each notebook will look like this:

```bash
# Install GenJAX (this takes about 1-2 minutes the first time)
pip install genjax
```

**What this does:**
- In Colab notebooks, you use `!pip install` to run shell commands
- `pip install` means "download and install"
- `genjax` is the library we're installing

**To run it:**
1. Click the cell
2. Press Shift + Enter
3. Wait for it to finish (you'll see progress messages)

{{% notice style="tip" %}}
You only need to install GenJAX **once per session**. If you come back later and restart the notebook, you'll need to run this cell again.
{{% /notice %}}

## Step 5: Import Required Libraries

After installation, you'll typically see a cell like:

```python
import jax
import jax.numpy as jnp
import genjax
from genjax import gen, bernoulli
import matplotlib.pyplot as plt
```

**What this does:**
- `import` means "load this library so we can use it"
- We're loading JAX (for computation), GenJAX (for probability), and matplotlib (for plotting)

**Just run it!** You don't need to understand every line. Think of it like turning on the lights before you start working.

## Step 6: Your First Code!

Now you're ready to run real GenJAX code! In the next chapter, you'll:
- Write a generative function for Chibany's meals
- Generate thousands of random days
- Visualize the results
- Use sliders to change probabilities and see what happens

## Common Issues & Solutions

### "Runtime disconnected"
**Problem:** Colab disconnects after ~90 minutes of inactivity
**Solution:** Just reconnect and re-run the cells (start to finish)

### "Restart Runtime" button
**Problem:** Something went wrong and you need a fresh start
**Solution:** Click Runtime → Restart runtime → Re-run all cells

### Code won't run
**Problem:** Cells need to be run in order
**Solution:** Click Runtime → Run all (to run from top to bottom)

## Keyboard Shortcuts (Optional)

These can make you faster, but they're optional:

| Action | Shortcut |
|--------|----------|
| Run current cell | Shift + Enter |
| Insert cell below | Ctrl + M, B |
| Delete cell | Ctrl + M, D |
| Comment/uncomment line | Ctrl + / |

---

## Quick Reference Card

Save this for later:

**Running Code:**
1. Click cell
2. Shift + Enter

**Saving Work:**
- Auto-saves to Google Drive (if you copied to Drive)
- Manual save: File → Save

**Fresh Start:**
- Runtime → Restart runtime

**Getting Help:**
- In code cell, type `?function_name` and run to see documentation
- Or just Google your question!

---

## You're Ready!

Now you have:
✅ Google Colab open
✅ Your own copy of the notebook
✅ GenJAX installed
✅ Basic understanding of the interface

**Next steps:**
- [Chapter 1: Python Essentials →](./01_python_basics.md) - Learn just enough Python
- [Chapter 2: Your First GenJAX Model →](./02_first_model.md) - Jump right into coding!

{{% notice style="success" %}}
**Pro tip:** Keep this tab open as a reference while you work through the notebooks!
{{% /notice %}}

---

|[← Previous: Introduction](./index.md) | [Next: Python Essentials →](./01_python_basics.md)|
| :--- | ---: |
