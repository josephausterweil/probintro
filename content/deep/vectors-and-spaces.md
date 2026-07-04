+++
date = "2026-07-04"
title = "From Bentos to Vectors: the Linear Algebra You Need"
weight = 5
tracks = ["beginner", "cogsci"]
+++

## The Kiosk That Cannot See

The new year brought two arrivals to campus. In Cafeteria A there is now a **bento kiosk** — a sleek box with a camera, said to recognize any bento you show it. And trailing three steps behind Chibany everywhere is the **robot mascot trainee**, assigned to learn the mascot trade by watching.

Chibany unwraps the lunchtime bento (the label says tonkatsu — students have labeled their bentos since the [Markov chains](../../sampling/markov-chains/) chapter) and holds it up to the kiosk's camera. The screen thinks for a moment, then chirps: *tonkatsu*. Chibany turns the box around, inspects the camera lens, peers behind the kiosk, and finds... no eyes. No taste buds. No nose.

Chibany looks at the trainee robot. The trainee looks back. Neither of them says anything, but the question hangs in the air, and it is the question of this whole Part: **how does a machine see a bento?**

> **Alyssa:** "It doesn't — not the way you do. A machine can't eat a photo and it can't smell pork cutlet. Machines eat exactly one thing: *numbers*. So before anything clever happens, the bento has to become a list of numbers."
>
> **Jamal:** "And the good news is that Chibany has been turning bentos into numbers all year. That's what the Journal *is*."

So Chibany does what Chibany always does: measures. Onto the scale — **500 grams**. A tap of the cutlet with a chopstick, a listen, and a rating on the campus crunch scale — **crunch 8 out of 10**. Two measurements, written top to bottom as a column:

$$\begin{bmatrix} 500 \\ 8 \end{bmatrix} \quad \begin{matrix}\leftarrow \text{weight (grams)} \\ \leftarrow \text{crunch (0–10)}\end{matrix}$$

That column of measurements has a name. It is a **vector**: an ordered list of numbers. Each number in the list is a **component** (the first component is 500, the second is 8), and the number of components is the vector's **dimension** — this is a 2-dimensional vector, or just "a vector in 2-D."

*Ordered* matters: the vector $[500, 8]$ means "500 grams, crunch 8," while $[8, 500]$ would describe a feather-light bento with an impossible crunch score. The position in the list carries the meaning.

And here is the picture that makes vectors friendly. A vector of two numbers **is a point in a plane**: put weight on the horizontal axis, crunch on the vertical axis, and the tonkatsu bento sits at the point $(500, 8)$. A second bento — the dinner hamburger, weighing **350 grams** with a limp **crunch of 3** — becomes the vector

$$\begin{bmatrix} 350 \\ 3 \end{bmatrix},$$

a different point in the same plane. Every bento the kiosk will ever see is a point in this plane. Recognizing bentos is going to mean doing *geometry* on those points.

{{% notice style="note" title="The Journal's columns were dimensions all along" %}}
Chibany's Bento Journal — the book's central data artifact since the [hierarchical Bayes](../../structure/hierarchical-bayes/) chapter — records each bento as a row: weight, arrival time, who brought it. Each *column* of the Journal is one measurement type. In this chapter's language: each column is a **dimension**, and each row of the Journal was secretly a vector the whole time. Nothing new has to be measured; we are just learning what the table already was. (The 500 g and 350 g are not new either — tonkatsu ≈ 500 g and hamburger ≈ 350 g were established back in [Mystery Bentos](../../learning/mystery-bentos/).)
{{% /notice %}}

---

## Arrows, and Similarity as Direction

Back at the kiosk, Chibany presents bento after bento, and the trainee robot dutifully plots each one as a point. But the kiosk does something more than plot: shown a new bento, it says *which known bento it resembles*. How do you compute "resembles" from two lists of numbers?

> **Alyssa:** "Draw each vector as an **arrow** — from the corner of the plot, the point $(0,0)$, out to the bento's point. Now every bento has a direction and a length. And here's the idea the whole chapter turns on: two bentos are *similar in kind* when their arrows **point the same way**."

To see the arithmetic clearly, Alyssa copies two bentos out of the Journal and rescales the weights to hundreds of grams so both components are small, friendly numbers:

- Bento $\mathbf{u}$: 300 g and crunch 4 $\;\rightarrow\; \mathbf{u} = [3, 4]$
- Bento $\mathbf{v}$: 400 g and crunch 3 $\;\rightarrow\; \mathbf{v} = [4, 3]$

(Bold letters like $\mathbf{u}$ are the traditional costume for vectors; ordinary letters stay plain numbers.)

The tool for comparing directions is the **dot product**, written $\mathbf{u} \cdot \mathbf{v}$: *multiply the matching components, then add the results.* That is the entire definition. With our two bentos, fully worked:

$$\mathbf{u} \cdot \mathbf{v} \;=\; \underbrace{3 \times 4}_{\text{weights}} + \underbrace{4 \times 3}_{\text{crunches}} \;=\; 12 + 12 \;=\; 24.$$

The dot product is big and positive when the arrows agree, near zero when they are unrelated, and negative when they oppose. Don't take that on faith — try two arrows at a right angle: $[1,0] \cdot [0,1] = 1 \times 0 + 0 \times 1 = 0$. The first arrow lives entirely in the weight direction, the second entirely in the crunch direction; they share nothing, and multiply-matching-components-and-add reports exactly that. The general fact (worth keeping — it is *why* this whole section works) is

$$\mathbf{u} \cdot \mathbf{v} \;=\; \|\mathbf{u}\|\,\|\mathbf{v}\|\cos\theta,$$

where $\theta$ (theta) is the angle between the arrows and $\|\mathbf{u}\|$ is a length we define next: the dot product literally *is* "how long they are, times how much they agree." That also means "24" alone mixes together *agreement* and *size* — a huge bento produces big products just by being huge. To isolate pure direction, divide out the lengths.

The **length** of a vector (also called its **norm**, written $\|\mathbf{u}\|$) is just the Pythagorean theorem — the straight-line distance from $(0,0)$ to the arrow's tip:

$$\|\mathbf{u}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5, \qquad \|\mathbf{v}\| = \sqrt{4^2 + 3^2} = \sqrt{16 + 9} = \sqrt{25} = 5.$$

Now divide the identity above through by both lengths — the sizes cancel, and what remains is pure agreement. This is the **cosine similarity**:

$$\cos\theta \;=\; \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|} \;=\; \frac{24}{5 \times 5} \;=\; \frac{24}{25} \;=\; 0.96.$$

Here $\theta$ (theta) is the angle between the two arrows, and $\cos\theta$ lands on a fixed scale: $1$ means the arrows point in exactly the same direction, $0$ means perpendicular (no relationship), $-1$ means exactly opposite. Our two bentos score $0.96$ — nearly identical in direction. Whatever kind of bento $\mathbf{u}$ is, $\mathbf{v}$ is very much the same kind.

> **Jamal:** "Let's make the laptop agree with the chalkboard."

```python
import jax.numpy as jnp

u = jnp.array([3.0, 4.0])   # bento u: 300 g -> 3 hundred-grams, crunch 4
v = jnp.array([4.0, 3.0])   # bento v: 400 g -> 4 hundred-grams, crunch 3

dot = jnp.dot(u, v)                       # 3*4 + 4*3
length_u = jnp.sqrt(jnp.sum(u ** 2))      # Pythagoras
length_v = jnp.sqrt(jnp.sum(v ** 2))
cosine = dot / (length_u * length_v)

print("u . v       =", float(dot))
print("length of u =", float(length_u))
print("length of v =", float(length_v))
print(f"cosine      = {float(cosine):.2f}")
```

**Output:**
```
u . v       = 24.0
length of u = 5.0
length of v = 5.0
cosine      = 0.96
```

Every number matches the hand calculation: $24$, $5$, $5$, $0.96$. (`jnp` is JAX's version of numpy; `jnp.dot` is the multiply-matching-components-and-add we just did by hand.)

{{% notice style="warning" title="Tanaka-san's take: \"closer points are always more similar\"" %}}
Tanaka-san watches the plot fill up and nods sagely: "So similarity is easy — whichever two points are *closest together* are the most similar bentos."

Plausible — and subtly wrong, and the reason is the difference between **length** and **direction**. Take a mini tonkatsu, $[2.5, 4]$ (250 g, crunch 4), and a jumbo tonkatsu, $[5, 8]$ (500 g, crunch 8). The jumbo is exactly the mini doubled — same proportions, same *kind* — so their arrows point in **exactly** the same direction: cosine $= 1.0$. Yet as points they are far apart: a distance of $\sqrt{2.5^2 + 4^2} \approx 4.7$ units. Meanwhile the hamburger $[3.5, 3]$ sits only $\sqrt{1^2 + 1^2} \approx 1.4$ units from the mini tonkatsu — three times *closer* — while pointing in a genuinely different direction.

Distance-as-similarity says the mini tonkatsu's nearest kin is the hamburger. Direction-as-similarity says it is the jumbo tonkatsu. For "what kind of bento is this?", direction wins: **length says how much bento; direction says what kind of bento.** That is precisely why the kiosk uses cosine similarity, and why the normalize button in the widget below is so satisfying to press.
{{% /notice %}}

Try all of this with your hands instead of your eyes:

<iframe src="../../widgets/vector-space.html" width="100%" height="560" frameborder="0" style="background:#111111; border-radius:6px; margin:1rem 0;" title="Vector space explorer: plot bentos as arrows, click two to see the worked dot product and cosine similarity"></iframe>

{{% notice style="info" title="Drive it yourself" %}}
Click any **two dishes** in the plane — the widget draws their arrows and shows the dot product worked term by term (its axes are rescaled to the plot, so the numbers differ from our hundred-gram units, but the recipe is identical). The two tonkatsu load preselected: nearly parallel arrows, cosine $\approx 0.997$. Now click **tonkatsu-B and the salad** — the angle opens and the cosine drops to $\approx 0.78$. Then toggle **normalize** to shrink every arrow to length 1: same-kind dishes land close together on the unit circle while different kinds spread apart — with length divided out, only direction remains. (One teachable wart: because all our measurements are positive, every pair scores *somewhat* similar — real systems center their features first.)
{{% /notice %}}

---

## More Measurements, Same Math

The kiosk's screen, it turns out, logs more than two numbers per bento. Chibany, delighted, adds columns: temperature and price. The tonkatsu bento is now *four* measurements — sticking with friendly rescaled units (hundreds of grams, crunch 0–10, tens of degrees Celsius, hundreds of yen):

$$\text{tonkatsu} = \begin{bmatrix} 5 \\ 8 \\ 6 \\ 7 \end{bmatrix} \quad \begin{matrix}\leftarrow \text{weight: } 500\text{ g} \\ \leftarrow \text{crunch: } 8 \\ \leftarrow \text{temperature: } 60\,^\circ\text{C} \\ \leftarrow \text{price: } 700\text{ yen}\end{matrix} \qquad \text{hamburger} = \begin{bmatrix} 3.5 \\ 3 \\ 4 \\ 5 \end{bmatrix}$$

A vector with four components is a point in **4-D**, and nobody — not Chibany, not the trainee, not you — can draw 4-D. Here is the liberating fact: *you never need to.* The math does not care. The dot product in 4-D is the same recipe as in 2-D — multiply matching components, add:

$$5 \times 3.5 \;+\; 8 \times 3 \;+\; 6 \times 4 \;+\; 7 \times 5 \;=\; 17.5 + 24 + 24 + 35 \;=\; 100.5.$$

Lengths are still square-root-of-sum-of-squares; cosine is still dot-over-lengths. The pictures stop at three dimensions, but the arithmetic — and the *intuition* of "arrows pointing the same way" — carries on unchanged into 4-D, 100-D, 10,000-D.

This is worth naming properly. The set of all possible measurement lists — every point the plane (or 4-D, or 10,000-D) could hold — is a **vector space**. When the dimensions are measured features of objects, as here, it is called a **feature space**: the space where weight, crunch, temperature, and price live. And when a machine *learns* where in such a space to place each object — rather than being handed measurements by Chibany — the object's learned position is called an **embedding**. An embedding is nothing mysterious: it is a vector, a point in a feature space, whose coordinates were chosen by training instead of by a scale and a chopstick.

The most famous embeddings are for *words*. The word2vec system (Mikolov et al., 2013) learned a vector for every English word, and something remarkable fell out: **directions in the space meant something.** Subtracting and adding word vectors did analogy:

$$\text{king} - \text{man} + \text{woman} \;\approx\; \text{queen}.$$

(Vector addition and subtraction, first use: component-wise, just like the dot product's "matching components" — $[a_1, a_2] - [b_1, b_2] = [a_1 - b_1,\; a_2 - b_2]$.)

You can see *why* this works with a toy 2-D embedding. Give words two coordinates, $[\text{royalty}, \text{masculinity}]$, scored 0–10:

$$\text{king} = \begin{bmatrix} 9 \\ 8 \end{bmatrix}, \quad \text{man} = \begin{bmatrix} 1 \\ 8 \end{bmatrix}, \quad \text{woman} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}.$$

Then $\text{king} - \text{man} = [9-1,\; 8-8] = [8, 0]$ — subtracting *man* cancels the masculinity and leaves a pure "royalty direction." Adding *woman* puts back a person: $[8+1,\; 0+2] = [9, 2]$ — a maximally royal, low-masculinity word. Queen. The same trick works on bentos (jumbo tonkatsu $-$ mini tonkatsu $=$ a pure "more of the same" direction) and, in the [Transformers & Attention](../transformers-attention/) chapter, on the embeddings inside a language model — there the coordinates number in the thousands and no human ever labels an axis "royalty"; the directions are discovered by training.

```python
import jax.numpy as jnp

# Four measurements per bento: weight, crunch, temperature, price (rescaled units)
tonkatsu  = jnp.array([5.0, 8.0, 6.0, 7.0])
hamburger = jnp.array([3.5, 3.0, 4.0, 5.0])

# The SAME code as 2-D -- jnp.dot doesn't care about the dimension
print("tonkatsu . hamburger =", float(jnp.dot(tonkatsu, hamburger)))

# Embedding arithmetic in the toy 2-D word space [royalty, masculinity]
king  = jnp.array([9.0, 8.0])
man   = jnp.array([1.0, 8.0])
woman = jnp.array([1.0, 2.0])
print("king - man + woman   =", king - man + woman)
```

**Output:**
```
tonkatsu . hamburger = 100.5
king - man + woman   = [9. 2.]
```

The 4-D dot product is $100.5$, exactly as worked above, from the *identical* line of code that handled 2-D. And the analogy lands on $[9, 2]$ — queen's corner of the space.

---

## Matrices: Machines That Move Every Point

The trainee robot has been quietly plotting everything, and now it does something new. On its screen, the whole cloud of bento points suddenly *shifts* — stretched wider along one axis, every point at once, like a rubber sheet being pulled. Chibany taps the screen and looks at Alyssa.

> **Alyssa:** "That's the other half of linear algebra. A vector is a point. A **matrix** is a *machine that moves every point at once* — the same rule applied to the whole space."

A **matrix** is a rectangular table of numbers. Here is a 2×2 one (two rows, two columns), and the rule for applying it to a vector — **matrix-vector multiplication**: *each row of the matrix takes a dot product with the vector, and the results stack into a new vector.* Fully worked:

$$\begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \end{bmatrix} \;=\; \begin{bmatrix} 2 \times 3 + 0 \times 4 \\ 0 \times 3 + 1 \times 4 \end{bmatrix} \;=\; \begin{bmatrix} 6 \\ 4 \end{bmatrix}.$$

Row one, $[2, 0]$, dotted with $[3, 4]$ gives $6$: the new first component. Row two, $[0, 1]$, gives $4$: the new second component. In bento terms: this matrix **stretches the weight axis by 2** and leaves crunch alone. The bento at $(3, 4)$ slides to $(6, 4)$ — and so does *every other point in the plane*, each by the same rule. One small table of four numbers describes a transformation of infinitely many points.

Now the key intuition — the single most useful fact in this chapter, so it gets its own paragraph:

**The matrix's first column is where $[1, 0]$ lands. The second column is where $[0, 1]$ lands. That's all a matrix is.** The vectors $[1,0]$ and $[0,1]$ — one step along each axis — are called the **basis vectors**, written $\mathbf{e}_1$ and $\mathbf{e}_2$ (the widget labels its two colored arrows this way); every other vector is built from them ($[3,4]$ is 3 of the first plus 4 of the second). And a matrix machine *respects that recipe*: it sends $3\,\mathbf{e}_1 + 4\,\mathbf{e}_2$ to 3 of wherever $\mathbf{e}_1$ landed plus 4 of wherever $\mathbf{e}_2$ landed — that respect-the-recipe property is what the word **linear** means. So once you know where the two basis arrows go, you know where *everything* goes, and the matrix simply records those two destinations as its two columns. Read $\begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$ column by column: $[1,0]$ lands on $[2,0]$ (the weight axis stretched), $[0,1]$ lands on $[0,1]$ (crunch untouched).

Try reading a less obvious machine the same way:

$$R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \qquad \begin{matrix}\text{first column: } [1,0] \text{ lands on } [0,1] \\ \text{second column: } [0,1] \text{ lands on } [-1,0]\end{matrix}$$

The east-pointing arrow ends up pointing north; the north-pointing arrow ends up pointing west. Both basis arrows turned a quarter-turn counterclockwise — so $R$ **rotates the whole plane** a quarter turn. Applied to our bento:

$$R \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 0 \times 3 + (-1) \times 4 \\ 1 \times 3 + 0 \times 4 \end{bmatrix} = \begin{bmatrix} -4 \\ 3 \end{bmatrix}.$$

In code, matrix multiplication gets its own operator, `@` — so `W @ x` below performs exactly the rows-dot-vector recipe we just did by hand:

```python
import jax.numpy as jnp

W = jnp.array([[2.0, 0.0],
               [0.0, 1.0]])     # stretch the weight axis by 2
x = jnp.array([3.0, 4.0])       # a bento
print("W @ x =", W @ x)

R = jnp.array([[0.0, -1.0],
               [1.0,  0.0]])    # quarter-turn: columns say where [1,0] and [0,1] land
print("R @ x =", R @ x)
```

**Output:**
```
W @ x = [6. 4.]
R @ x = [-4.  3.]
```

(The `@` symbol is Python's matrix-multiplication operator: `W @ x` performs exactly the rows-dot-vector recipe.)

### Two machines in a row — and the punchline

What if the trainee runs *two* machines, one after another — first stretch with $A$, then rotate with $B$? You could apply them in sequence. Or you could multiply the matrices: **matrix-matrix multiplication** $BA$ builds the single machine that does "$A$, then $B$" in one step. The recipe extends the one you know — each entry of $BA$ is a *row of $B$* dotted with a *column of $A$* (a column of $A$ is just a vector, so this is the same rows-dot-vector move, done once per column). Worked in full, with $A$ the stretch and $B$ the quarter-turn:

$$BA = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 \times 2 + (-1) \times 0 & 0 \times 0 + (-1) \times 1 \\ 1 \times 2 + 0 \times 0 & 1 \times 0 + 0 \times 1 \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 2 & 0 \end{bmatrix}.$$

(Order matters: $BA$ means "$A$ first, then $B$" — read right to left, like function application. $AB$ would be a different machine.)

```python
import jax.numpy as jnp

A = jnp.array([[2.0, 0.0],
               [0.0, 1.0]])     # machine 1: stretch
B = jnp.array([[0.0, -1.0],
               [1.0,  0.0]])    # machine 2: quarter-turn
x = jnp.array([3.0, 4.0])

C = B @ A                       # ONE machine that does "A, then B"
print("C = B @ A gives:")
print(C)
print("B @ (A @ x) =", B @ (A @ x))   # run the two machines in sequence
print("(B @ A) @ x =", C @ x)         # run the single combined machine
```

**Output:**
```
C = B @ A gives:
[[ 0. -1.]
 [ 2.  0.]]
B @ (A @ x) = [-4.  6.]
(B @ A) @ x = [-4.  6.]
```

Identical answers, $[-4, 6]$ both ways. And that identity is not a party trick — it is the fact the entire next chapter leans on:

{{% notice style="note" title="Why depth alone buys nothing" %}}
Matrices are **linear** machines: they can rotate, stretch, squash, and shear the space, but they always keep the grid made of straight, evenly spaced lines. And as the code just showed, **two linear machines compose into one linear machine** — $B$ after $A$ is just the single matrix $BA$. Stack ten matrices, a hundred, a thousand: the result collapses into *one* matrix, no more expressive than where you started. A deep stack of purely linear layers is an expensive way to build a shallow one. Whatever makes deep networks powerful, it cannot be the matrices alone — something *nonlinear* has to happen between them. That something is next.
{{% /notice %}}

<iframe src="../../widgets/matrix-transform.html" width="100%" height="560" frameborder="0" style="background:#111111; border-radius:6px; margin:1rem 0;" title="Matrix transform playground: drag the matrix entries and watch the whole grid move; toggle ReLU to fold the plane"></iframe>

{{% notice style="info" title="Drive it yourself" %}}
Drag the four **sliders** — the four matrix entries — and watch the whole grid move at once. Set the matrix to $[[2,0],[0,1]]$ and see the stretch from the worked example; set $[[0,-1],[1,0]]$ and see the quarter-turn. Watch the two colored arrows: they are the basis vectors, and they always land exactly on the matrix's columns. Then press **ReLU** and watch the plane *fold*: everything that was pushed into negative territory gets flattened onto the axes. That fold is the nonlinearity the notice box promised — bend after bend is what a matrix alone can never do.
{{% /notice %}}

---

## One Layer of a Neural Network

Back to the kiosk, and to the payoff. However fancy its insides, the kiosk's basic move — the basic move of every neural network — is a one-line formula Chibany can now read word by word:

$$\mathbf{h} = g(W\mathbf{x})$$

- $\mathbf{x}$ is the **input vector** — the bento as a list of measurements. (For a real photo, the components are pixel brightnesses — thousands of dimensions, but as we saw, the math doesn't blink.)
- $W$ is a **matrix** — a machine that moves every point: rotating, stretching, and mixing the input's dimensions into new combinations.
- $g$ is a simple **nonlinear function** applied to each component of the result. The most popular choice is the **ReLU** (rectified linear unit): $g(z) = \max(0, z)$ — keep positive values, replace negatives with zero. Geometrically, ReLU **folds the space**: the parts of the plane that $W$ pushed negative get flattened onto the axes, creasing the rubber sheet instead of merely stretching it.
- $\mathbf{h}$ is the layer's output — a new vector, the bento's coordinates in a *new, warped* space. And because it is just a vector, it can be fed into another layer: $g(W_2\, \mathbf{h})$, and another after that.

One layer: multiply by a matrix, then bend. That's it — you already know both halves.

```python
import jax.numpy as jnp

W = jnp.array([[1.0, -1.0],     # one layer's matrix: mixes weight and crunch
               [1.0,  1.0]])
x = jnp.array([3.0, 4.0])       # the bento goes in

pre  = W @ x                    # move every point (linear)
post = jnp.maximum(0.0, pre)    # ReLU: fold the negative part away (nonlinear)

print("before the bend:", pre)
print("after  the bend:", post)
```

**Output:**
```
before the bend: [-1.  7.]
after  the bend: [0. 7.]
```

The matrix sends the bento to $[-1, 7]$; the ReLU folds the negative first component up to zero, leaving $[0, 7]$. The line `jnp.maximum(0.0, pre)` *is* $g$ — one bend, and the machine is no longer linear, so the collapse argument from the notice box no longer applies. Stacking now genuinely adds power.

That is the whole picture of a network doing recognition, and it is worth a diagram:

```mermaid
graph LR
    B["bento 🍱"] -->|measure| X["vector x<br/>[500, 8, ...]"]
    X --> W1["matrix W<br/>rotate & stretch<br/>the space"]
    W1 --> G1["ReLU g<br/>fold the space"]
    G1 --> W2["matrix W₂"]
    W2 --> G2["ReLU g"]
    G2 --> D["decision<br/>tonkatsu!"]
    classDef node fill:none,stroke:#9bbcff,stroke-width:2px,color:#fff
    class B,X,W1,G1,W2,G2,D node
    linkStyle default stroke:#9bbcff,stroke-width:2px,color:#fff
```

Warp, fold, warp, fold — each layer reshapes the feature space so that, by the last layer, the tonkatsu points and the hamburger points that started tangled together have been pulled to *opposite sides of the room*, where telling them apart is easy. A deep network is not doing anything more exotic than the moves in this chapter, repeated: matrices moving every point, folds creasing the sheet between them.

> **Alyssa:** "One honest confession, though. We *chose* every number in $W$ today. The kiosk's matrices contain millions of numbers, and nobody chose them."
>
> **Jamal:** "They were *learned*. And 'learned' is a word this book already knows how to cash out — that's where we go next."

How the numbers inside $W$ get learned — and why that learning is our old friend inference wearing a new costume — is the business of the next two chapters, [From Rules to Weights](../from-rules-to-weights/) and [Neural Net Fundamentals](../neural-net-fundamentals/). The trainee robot, who has been taking notes this entire time, seems oddly eager for that part.

---

{{% notice style="success" title="What you can do now" %}}
You can turn an object into a **vector** — an ordered list of measurements — and read it as a point (or an arrow) in a **feature space**, one **dimension** per measurement, each entry a **component**: the tonkatsu bento became $[500, 8]$, the hamburger $[350, 3]$. You can compute a **dot product** by hand (multiply matching components, add: $3 \times 4 + 4 \times 3 = 24$), a **length/norm** by Pythagoras ($\sqrt{3^2+4^2}=5$), and a **cosine similarity** ($24/(5 \times 5) = 0.96$) — and you can say why *direction* means "what kind" while *length* means "how much" (Tanaka-san's closest-point rule fails on the mini vs. jumbo tonkatsu). You know the same arithmetic runs unchanged in 4-D and beyond ($100.5$ for the 4-D bentos), and that an **embedding** is a learned position in such a space — with directions meaningful enough that $\text{king} - \text{man} + \text{woman} \approx \text{queen}$ works out to $[9,2]$ in the toy space. You can read a **matrix** as a machine that moves every point, run **matrix-vector multiplication** by rows ($W\mathbf{x}$: for $W = [[2,0],[0,1]]$ and $\mathbf{x} = [3,4]$, the result is $[6,4]$), and — the key — read its **columns as the landing spots of the basis vectors** $\mathbf{e}_1, \mathbf{e}_2$. You saw two machines compose into one ($BA$, giving $[-4,6]$ either way), so purely linear depth collapses — and why the **ReLU** fold ($[-1,7] \to [0,7]$) is what rescues stacking. And you can now decode $\mathbf{h} = g(W\mathbf{x})$ word by word: measure, move every point, bend, repeat.

Next, [From Rules to Weights](../from-rules-to-weights/) asks where the numbers in $W$ come from — and finds a likelihood hiding inside a loss function.

*Glossary:* [vector](../../glossary/#vector-), [component](../../glossary/#component-), [dimension](../../glossary/#dimension-), [dot product](../../glossary/#dot-product-), [norm (vector length)](../../glossary/#norm-vector-length-), [cosine similarity](../../glossary/#cosine-similarity-), [feature space](../../glossary/#feature-space-), [embedding](../../glossary/#embedding-), [matrix](../../glossary/#matrix-), [basis vectors](../../glossary/#basis-vectors-), [ReLU](../../glossary/#relu-).
{{% /notice %}}

---

## Exercises

{{% notice style="info" title="Try it yourself" %}}
1. **A third bento.** The Journal lists a karaage bento at 400 g with crunch 6: $\mathbf{w} = [4, 6]$. By hand, compute its cosine similarity with $\mathbf{u} = [3,4]$ and with the hamburger $[3.5, 3]$. Which is it more similar *in kind* to — and would plain distance have agreed?
2. **Read the machine from its columns.** Without multiplying anything, describe what $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$ does to the plane (where does each basis vector land?). Then verify on $[3,4]$ by the row recipe.
3. **Break the collapse.** In the composition cell, replace the quarter-turn $B$ with a second stretch $[[3,0],[0,1]]$ and confirm the two-machines-equal-one identity still holds. Then insert `jnp.maximum(0.0, ...)` between the two machines and show the identity *fails* — the fold is doing real work.
{{% /notice %}}

A companion notebook works through all of this interactively:

**📓 [Open in Colab: `vectors_and_spaces.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/vectors_and_spaces.ipynb)**

---

## References

- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*. <https://arxiv.org/abs/1301.3781>
- Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.

---

Special thanks to [JPPCA](https://jpcca.org/) for their generous support of this tutorial series.
