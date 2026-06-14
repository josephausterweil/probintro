+++
date = "2026-06-14"
title = "ガウス混合モデル"
weight = 5
+++

## 謎に立ち返る

第1章でのChibanyの最初の謎を覚えていますか？ 重さの分布に2つのピークを持つ謎のお弁当があったにもかかわらず、平均値はどの弁当も実際には存在しない谷間に落ちていました。

これを完全に解くためのツールがすべて揃いました：
- **第1章**：混合分布における期待値のパラドックス
- **第2章**：連続確率（PDF、CDF）
- **第3章**：ガウス分布
- **第4章**：パラメータのベイズ学習

これらを組み合わせます：**複数のガウス分布が混合されており、各観測値がどのコンポーネントに属するかと、各コンポーネントのパラメータの両方を明らかにする必要がある場合はどうすればよいでしょうか？**

これが**ガウス混合モデル（GMM）**です。

---

## 📚 前提知識：分類の理解

GMM学習問題の全体に取り組む前に、**既知のパラメータ**を用いた混合モデルにおける**分類**を理解していることを確認してください。

{{% notice style="warning" title="⚠️ 推奨される準備" %}}
まだ取り組んでいない方は、第4章の**ガウスクラスター**課題を通してください：

**📝 課題**：[Colabで開く：`solution_2_gaussian_clusters.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/solution_2_gaussian_clusters.ipynb)

**📓 インタラクティブな探索**：[Colabで開く：`gaussian_bayesian_interactive_exploration.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/gaussian_bayesian_interactive_exploration.ipynb)（パート2）

**この重要性**：
- **第4章 問題2** では、パラメータが**既知**の場合にP(カテゴリ | 観測値)を計算する方法を学びます
- **この章（第5章）** では、パラメータが**未知**の場合のパラメータ学習に拡張します
- 既知パラメータでの分類を理解することは、それらを学習しようとする前に不可欠です！

**練習すること**：
- ベイズの定理の使用：P(c|x) = p(x|c)P(c) / p(x)
- 周辺分布の計算：p(x) = Σ_c p(x|c)P(c)
- 決定境界と事前分布・分散がそれに与える影響の理解
- 二峰性対単峰性の混合分布の可視化
{{% /notice %}}

### 橋渡し：既知パラメータ → 未知パラメータ

**第4章 問題2** で学んだこと：
- 与えられたもの：μ₁、μ₂、σ₁²、σ₂²、θ（すべて既知）
- 推論するもの：各観測値のカテゴリ
- 公式：P(c=1|x) = θ·N(x;μ₁,σ₁²) / [θ·N(x;μ₁,σ₁²) + (1-θ)·N(x;μ₂,σ₂²)]

**この章では**、より難しい問題に取り組みます：
- 与えられたもの：観測値のみ x₁, x₂, ..., xₙ
- 推論するもの：カテゴリ**および**パラメータ μ₁、μ₂、σ₁²、σ₂²、θ
- 方法：期待値最大化（EM）アルゴリズム

次のように考えてみましょう：
1. **まず**（第4章 問題2）：「とんかつのレシピ（μ₁、σ₁²）とハンバーグのレシピ（μ₂、σ₂²）は分かっている。重さを見てどちらか判定できるか？」
2. **今度**（第5章）：「レシピが分からない！重さだけからレシピを割り出せるか？」

---

## 問題の全体像

Chibanyは20個の謎のお弁当を受け取りました。その重さを測定します：

```
[498, 352, 501, 349, 497, 503, 351, 500, 348, 502,
 499, 350, 498, 353, 501, 347, 499, 502, 352, 500]
```

ヒストグラムを見ると、350g と 500g 付近に2つの明確なクラスターが見えます。

**問い**：
1. お弁当の**種類はいくつ**ありますか？（ここでは2とします）
2. 各お弁当は**どの種類**ですか？（分類問題）
3. 各種類の**パラメータは何**ですか？（学習問題）

---

## ガウス混合モデル：数学

GMM は、各観測値が K 個のガウスコンポーネントのいずれかから生成されると仮定します：

$$p(x) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x | \mu_k, \sigma_k^2)$$

ここで：
- **π_k**：混合比率（コンポーネント k の確率）
- **μ_k**：コンポーネント k の平均
- **σ_k²**：コンポーネント k の分散

制約条件：$\sum_{k=1}^{K} \pi_k = 1$（確率の総和は1）

### 生成的なストーリー

1. **コンポーネントを選ぶ**：k ~ Categorical(π₁, π₂, ..., πₖ) からサンプリング
2. **観測値を生成する**：x ~ N(μₖ, σₖ²) からサンプリング

まさに GenJAX が得意とする場面です！

{{% notice style="info" title="📘 基礎概念：離散と連続の組み合わせ" %}}
**ここでの美しい組み合わせに注目してください！**

**ステップ1は離散**（Tutorial 1 と同様）：
- どのコンポーネントを選ぶか：k ~ Categorical(π₁, π₂, ..., πₖ)
- これはちょうど {ハンバーグ、とんかつ} から選ぶようなものです
- 離散的な結果（コンポーネント1、コンポーネント2、…）を**数えて**います
- Tutorial 1 より：**確率変数**は結果を値に対応させます

**ステップ2は連続**（Tutorial 3 と同様）：
- 実際の重さを生成する：x ~ N(μₖ, σₖ²)
- 第2章で学んだ**確率密度**を使います
- 連続的な値（350g、500g、…）を**測定**しています

**これが重要な理由：**
- 実際の問題ではしばしば両方が組み合わさっています！
- 離散的な選択（どのカテゴリか？）+ 連続的な測定（どんな値か？）
- **Tutorial 1 の論理**（離散的な数え方）と **Tutorial 3 のツール**（連続的な密度）が一緒に機能します
- GenJAX は同じモデルの中で両方をシームレスに扱います

**強み：** 混合モデルは、離散確率と連続確率が別々の世界ではなく、豊かな現実世界の現象をモデル化するために協力して機能することを示しています。

[← Tutorial 1 第3章で確率変数を復習する](../../intro/03_prob_count/#random-variables)

[← Tutorial 3 第2章で連続分布を復習する](../02_continuous/)
{{% /notice %}}

---

## 2コンポーネントのお弁当モデル

Chibanyのお弁当（K=2：とんかつとハンバーグ）の場合：

**コンポーネント1（とんかつ）**：
- π₁ = 0.7（お弁当の70%）
- μ₁ = 500g
- σ₁² = 4（標準偏差 = 2g）

**コンポーネント2（ハンバーグ）**：
- π₂ = 0.3（お弁当の30%）
- μ₂ = 350g
- σ₂² = 4（標準偏差 = 2g）

```python
import jax
import jax.numpy as jnp
import jax.random as random
from genjax import gen, flip, normal

@gen
def bento_mixture_model():
    """Two-component Gaussian mixture for Chibany's mystery bentos."""

    # Choose component with flip(0.7): True = tonkatsu (70%), False = hamburger (30%).
    # We use flip() because the choice is binary; flip(p) takes a probability directly.
    is_tonkatsu = flip(0.7) @ "component"

    # Pick the chosen component's parameters. jnp.where() selects without an if/else,
    # which keeps the model JAX-traceable.
    mu = jnp.where(is_tonkatsu, 500.0, 350.0)
    sigma = jnp.where(is_tonkatsu, 2.0, 2.0)   # both stds are 2.0 here

    # Generate the weight from the chosen component's Gaussian.
    weight = normal(mu, sigma) @ "weight"

    return weight, is_tonkatsu

# Simulate 20 bentos. GenJAX runs a model with model.simulate(key, args);
# here args is the empty tuple () because bento_mixture_model takes no arguments.
key = random.PRNGKey(42)
keys = random.split(key, 20)

# jax.vmap runs simulate once per key, in parallel.
traces = jax.vmap(lambda k: bento_mixture_model.simulate(k, ()))(keys)
weights, is_tonkatsu = traces.get_retval()

n_tonkatsu = jnp.sum(is_tonkatsu)
n_hamburger = jnp.sum(~is_tonkatsu)

print(f"Generated {n_tonkatsu} tonkatsu and {n_hamburger} hamburger bentos")
print(f"Weights: {weights}")
```

**出力（数値はランダムのため異なる場合があります）：**
```
Generated 14 tonkatsu and 6 hamburger bentos
Weights: [501.2 349.8 499.5 351.3 498.7 502.1 350.5 ...]
```

---

## 推論問題

**順方向（生成的）**：パラメータ（π、μ、σ²）が与えられたとき、観測値を生成する ✅
**逆方向（推論）**：観測値が与えられたとき、パラメータ（π、μ、σ²）と割り当てを推論する ❓

こちらの方が難しいです！次の問題を解く必要があります：
1. 各観測値はどの**コンポーネント**から来たか？
2. **パラメータ**（μ₁、μ₂、σ₁²、σ₂²）は何か？
3. **混合比率**（π₁、π₂）は何か？

これらの問題は相互依存しています：
- 割り当てが分かれば、パラメータを簡単に推定できる（各コンポーネントごとに平均・分散を計算するだけ）
- パラメータが分かれば、割り当て確率を計算できる（各点がどちらのガウス分布に近いか？）

典型的な鶏と卵の問題です！

---

## 推論の難しさを理解する

各お弁当の種類が分かっていれば、学習は単純です――各グループの平均と分散を計算するだけです。逆に、真のパラメータが分かっていれば、各観測値がどのコンポーネントから来た可能性が高いかを計算できます。

この鶏と卵の問題こそ、確率的推論が解決するよう設計されていることです。点推定ではなく、GenJAX を使ってパラメータと割り当ての両方にわたる完全な事後分布を推論します。

---

## GenJAX によるベイズ GMM

次に、GenJAX を使ってベイズ版を実装します。ここでは、コンポーネントの平均と各観測値の割り当ての両方を、推論すべき潜在変数として扱います。

*混合*構造に焦点を当てるため、2つの標準偏差は**既知**（各コンポーネントで σ = 2）として扱い、平均と割り当てのみを学習します。分散も学習することは簡単な拡張ですが（各分散に事前分布を追加する）、固定することでこの最初のモデルが読みやすくなります。

```python
import jax
import jax.numpy as jnp
import jax.random as random
from genjax import gen, flip, normal, ChoiceMap

# Mystery bento weights from earlier.
mystery_weights = jnp.array([
    498.0, 352.0, 501.0, 349.0, 497.0, 503.0, 351.0, 500.0, 348.0, 502.0,
    499.0, 350.0, 498.0, 353.0, 501.0, 347.0, 499.0, 502.0, 352.0, 500.0
])

# The number of observations is a *structural* constant of the model — the @gen
# function below loops `range(N_OBS)` to lay out one assignment + one observation
# per data point. It must be a plain Python int (not a traced model argument),
# because JAX cannot trace through `range()` of a traced value. We close over it.
N_OBS = len(mystery_weights)
SIGMA_KNOWN = 2.0  # known standard deviation, shared by both components

@gen
def bayesian_gmm():
    """Bayesian 2-component Gaussian Mixture Model.

    Latents: two component means (mu_0, mu_1) and one assignment per observation.
    The standard deviation is the fixed constant SIGMA_KNOWN. The number of
    observations is the fixed constant N_OBS — both are closed over, not passed
    as arguments, so the loop length is known at trace time.
    """
    # Priors on the two component means (vague Normal priors).
    mu_0 = normal(400.0, 50.0) @ "mu_0"
    mu_1 = normal(400.0, 50.0) @ "mu_1"

    # Generate each observation: pick a component, then sample from its Gaussian.
    for i in range(N_OBS):
        # Assignment for observation i: flip(0.5) — True = component 1, False = component 0.
        z_i = flip(0.5) @ f"z_{i}"

        # Pull the assigned component's mean with jnp.where (keeps the model traceable).
        mu_i = jnp.where(z_i, mu_1, mu_0)

        # The observation itself.
        x_i = normal(mu_i, SIGMA_KNOWN) @ f"x_{i}"

    return mu_0, mu_1

# Condition on the observed weights by building a ChoiceMap that fixes each "x_i".
# ChoiceMap.d({...}) builds a choice map from a plain Python dict.
observations = ChoiceMap.d({
    f"x_{i}": mystery_weights[i] for i in range(N_OBS)
})

# generate() runs the model with those choices forced, and returns a trace plus
# a log-importance-weight. Running it for many keys gives weighted posterior samples.
# The model takes no arguments, so the args tuple is empty: ().
key = random.PRNGKey(42)
num_particles = 1000
keys = random.split(key, num_particles)

def one_particle(k):
    trace, log_weight = bayesian_gmm.generate(k, observations, ())
    choices = trace.get_choices()
    return choices["mu_0"], choices["mu_1"], log_weight

mu_0_samples, mu_1_samples, log_weights = jax.vmap(one_particle)(keys)

# Convert log-weights to normalized weights (log-sum-exp trick for stability).
weights = jnp.exp(log_weights - jnp.max(log_weights))
weights = weights / jnp.sum(weights)

# Weighted posterior means.
post_mu_0 = jnp.sum(weights * mu_0_samples)
post_mu_1 = jnp.sum(weights * mu_1_samples)

print(f"Posterior mean for mu_0: {post_mu_0:.1f}")
print(f"Posterior mean for mu_1: {post_mu_1:.1f}")
```

**注意**：これは*重み付きサンプリング（importance sampling）*——最もシンプルな推論手法であり、意図的にそうしています。印字された事後平均が350や500にきれいに収まるとは思わないでください：20個の観測値と漠然とした事前分布では、ランダムに引き出されたほとんどの粒子がデータをうまく説明できず、ほぼゼロの重みを持つため、加重平均は事前分布の方向に引っ張られ、推定が不安定になります。これはバグではなく、*まさに*重み付きサンプリングだけでは不十分な理由です。実際の GMM 推論では、よりスマートなアルゴリズム（EM、MCMC、変分法）を使い、それらは後の章で学びます。それらはデータに実際に適合するパラメータ設定に計算を集中させます。ベイズの枠組みは特に DPMM（第6章）で強力になります。そこでは、コンポーネントの*数*さえも不確かです。

---

## モデル選択：コンポーネントはいくつ必要か？

K=2 とどうやって分かるのでしょうか？お弁当の種類が3つあるいは5つある場合は？

従来のアプローチでは、異なる K の値で複数のモデルを当てはめ、BIC（ベイズ情報量規準）などの基準を使って最良のモデルを選択します。

しかし、完全ベイズ推論（第6章でより深く探求します）では、K 自体を確率変数として扱い、事後分布を通じてデータがコンポーネントの数について教えてくれるようにすることができます。

---

## 実際の応用例

GMM はお弁当だけのものではありません。至るところで登場します：

### 画像セグメンテーション
- 各ピクセルは K 個のクラスター（例：前景対背景）のいずれかに属する
- ピクセルの輝度値からクラスターパラメータを学習する

### 話者識別
- 異なる話者からの音響特徴が異なるようにクラスター化される
- GMM が声の特性の分布をモデル化する

### 異常検知
- 正常なデータが典型的なパターンの混合に適合する
- 外れ値はすべてのコンポーネントで低い確率を持つ

### 顧客セグメンテーション
- 顧客が行動（高額消費者、時々購入する人など）によってクラスター化される
- 各セグメントが特徴空間においてガウス分布としてモデル化される

---

## 練習問題

### 問題1：3種類のコーヒーブレンド

あるカフェで3種類のコーヒーブレンドを提供しています。30杯のカフェイン量（mg/カップ）を測定します：

```
[82, 118, 155, 80, 120, 158, 79, 115, 160, 83, 121, 157,
 81, 119, 156, 84, 117, 159, 78, 122, 154, 82, 116, 158,
 80, 120, 155, 81, 118, 157]
```

**a)** ベイズ GMM コードを K=3 コンポーネントに拡張してください。

**b)** カフェイン量が50〜200mgの範囲であることが分かっている場合、平均に対してどのような事前分布が適切でしょうか？

**c)** コンポーネント割り当てにわたる事後分布をどのように解釈しますか？

<details>
<summary>解答を表示</summary>

```python
import jax
import jax.numpy as jnp
from genjax import gen, categorical, normal

# Coffee caffeine data.
coffee_data = jnp.array([
    82.0, 118.0, 155.0, 80.0, 120.0, 158.0, 79.0, 115.0, 160.0, 83.0,
    121.0, 157.0, 81.0, 119.0, 156.0, 84.0, 117.0, 159.0, 78.0, 122.0,
    154.0, 82.0, 116.0, 158.0, 80.0, 120.0, 155.0, 81.0, 118.0, 157.0
])

# Structural constants — closed over by the @gen function, NOT passed as
# arguments, so the loop length is a concrete int at trace time. (See the
# bento GMM above for why this matters.)
COFFEE_N_OBS = len(coffee_data)
COFFEE_SIGMA = 5.0  # known standard deviation, shared by all components

@gen
def coffee_gmm():
    """3-component GMM for coffee blends.

    a) Extends the 2-component model to K=3 by using categorical() for the
       component choice instead of flip().
    Latents: three component means; one assignment per observation.
    The standard deviation is fixed (COFFEE_SIGMA) — same simplification as the
    bento GMM above; learning the variances is a straightforward extension.
    """
    K = 3

    # Equal mixing proportions, fixed. categorical(probs) takes probabilities
    # directly (not log-probabilities).
    mixing_probs = jnp.full(K, 1.0 / K)

    # Priors on the three means — centered on the expected low/medium/high range.
    mu_0 = normal(80.0, 20.0) @ "mu_0"    # Low caffeine
    mu_1 = normal(120.0, 20.0) @ "mu_1"   # Medium caffeine
    mu_2 = normal(160.0, 20.0) @ "mu_2"   # High caffeine
    means = jnp.array([mu_0, mu_1, mu_2])

    # Generate observations with component assignments.
    for i in range(COFFEE_N_OBS):
        z_i = categorical(mixing_probs) @ f"z_{i}"   # 0, 1, or 2
        mu_i = means[z_i]
        x_i = normal(mu_i, COFFEE_SIGMA) @ f"x_{i}"

    return mu_0, mu_1, mu_2

# Conditioning + importance sampling follows the same pattern as the bento GMM:
# build a ChoiceMap fixing each "x_i" to coffee_data[i], then call
# coffee_gmm.generate(key, observations, ())  — the model takes no arguments.

# b) The priors above use Normal(expected_mean, 20.0), which allows reasonable
#    variation while keeping the means inside the plausible 50-200 mg range.

# c) The posterior over each z_i tells us the probability that cup i belongs to
#    each blend, accounting for uncertainty in both the assignments and the means.
```
</details>

---

### 問題2：不確かさを理解する

お弁当データに対するベイズ GMM を使って：

**a)** 特定の観測値がどのコンポーネントに属するかについての不確かさをどのように定量化しますか？

**b)** これは割り当ての点推定とどのように異なりますか？

<details>
<summary>解答を表示</summary>

**a)** ベイズアプローチでは、コンポーネント割り当てにわたる完全な事後分布が得られます。各観測値 i について、次のことを計算できます：
- P(z_i = 0 | データ) - コンポーネント0である確率
- P(z_i = 1 | データ) - コンポーネント1である確率

決定境界付近の観測値では P(z_i = 0) ≈ 0.5 となり、高い不確かさを示す可能性があります。

**b)** 点推定では、各観測値を最も可能性の高いコンポーネントに単純に割り当て、確信度に関する情報を捨ててしまいます。ベイズアプローチはこの不確かさを保持します。これは以下のために重要です：
- 曖昧なケースの特定
- 下流のタスクへの不確かさの伝播
- 不確かさの下でのより良い意思決定

例えば、425g（2つのクラスターのちょうど中間）のお弁当は、無視すべきでない高い割り当て不確かさを持ちます。
</details>

---

## 次は何？

これで理解できました：
- ガウス混合モデルは複数のガウス分布を組み合わせる
- GMM は離散的な選択（コンポーネント割り当て）と連続的な観測を優雅に組み合わせる
- GenJAX は確率的プログラムとして生成プロセスを自然に表現する
- ベイズ推論はパラメータと割り当ての両方にわたる不確かさを保持する

しかし、**K**（コンポーネント数）を事前に**指定しなければなりませんでした**。クラスターがいくつ存在するか分からない場合はどうすればよいでしょうか？

第6章では、**ディリクレ過程混合モデル（DPMM）**について学びます：データから自動的にコンポーネント数を学習するベイズアプローチです！

---

{{% notice style="tip" title="重要なまとめ" %}}
1. **GMM**：混合比率 π を持つ K 個のガウス分布の混合
2. **生成プロセス**：まずコンポーネントを選ぶ（離散）、次に観測値を生成する（連続）
3. **ベイズ推論**：パラメータと割り当てにわたる完全な事後分布を推論する
4. **GenJAX**：確率的プログラムとして GMM を宣言的に表現する
5. **不確かさ**：コンポーネントのメンバーシップに関する不確かさを保持・定量化する
6. **応用**：クラスタリング、セグメンテーション、異常検知
{{% /notice %}}

---

**次の章**：[ディリクレ過程混合モデル →](./06_dpmm.md)
