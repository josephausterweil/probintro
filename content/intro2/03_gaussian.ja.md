+++
date = "2026-06-14"
title = "ガウス分布"
weight = 3
+++

## ベルカーブ

第2章で一様分布について学んだ後、Chibany はあることに気づきます。**実際の測定値が範囲全体に均等に広がることはほとんどない**のです。1000食のとんかつ弁当を丁寧に計測しても、重量が 495g から 505g の間に一様に分布するわけではありません。むしろ、ほとんどの値が 500g 付近に集まり、その中心値から離れるほど測定値は少なくなります。

このパターンは自然界のいたるところで見られます:
- 人の身長
- 測定誤差
- テストの点数
- 日々の気温
- そして、弁当の重量も！

これが**ガウス分布**（**正規分布**とも呼ばれます）であり、統計学において最も重要な確率分布と言っても過言ではありません。

特徴的な「ベルカーブ」形状は、基本的なパターンを捉えています。ほとんどの値が平均値付近に集まり、そこから離れるにつれてなめらかに対称的に減少するというパターンです。

---

## ガウス分布の確率密度関数

ガウス分布の確率密度関数（PDF）は次のように表されます:

$$p(x|\mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{1}{2\sigma^2}(x-\mu)^2\right)$$

**慌てないでください！** この数式を暗記する必要はありません。GenJAX が代わりに処理してくれます。しかし、パラメータが何を意味するかを理解しておきましょう。

### 形状を決める2つのパラメータ

**1. 平均値（μ、「ミュー」）**: ベルカーブの中心
- ここにピークがあります
- これは期待値でもあります: E[X] = μ
- μ を変えると、曲線全体が左右にシフトします

**2. 分散（σ²、「シグマ二乗」）**: 曲線の広がり
- 分散が大きい → 幅広く平たいベル
- 分散が小さい → 幅が狭く高いベル
- 標準偏差（σ）はその平方根です: σ = √(σ²)

### わかりやすく言うと

ガウス分布の PDF が意味することは: **「μ に近い値が最も起こりやすく、そこから離れるほど尤度はなめらかに下がる。その下がり方の速さは σ² によって決まる。」**

指数部分 $\exp\left(-\frac{1}{2\sigma^2}(x-\mu)^2\right)$ の複雑な式がベル形状を作り出します。重要な洞察:
- x = μ（平均値）のとき、指数は 0 となり exp{0} = 1（最大の高さ）
- x が μ から離れるにつれ、$(x-\mu)^2$ が大きくなり、指数が負に向かう
- 負の指数は 0 に近づき、裾野（テール）を形成する

---

## 68-95-99.7 規則

ガウス分布の最も有用な性質の一つ:

**値の 68% が平均値から1標準偏差以内に収まる**
- つまり、μ - σ から μ + σ の間

**値の 95% が2標準偏差以内に収まる**
- μ - 2σ から μ + 2σ の間

**値の 99.7% が3標準偏差以内に収まる**
- μ - 3σ から μ + 3σ の間

### なぜ重要なのか

Chibany のとんかつ弁当が N(500, 4)（平均 500g、分散 4g²）に従うとすると:
- 標準偏差 σ = √4 = 2g
- 弁当の 68% が 498g から 502g の間（500 ± 2）
- 95% が 496g から 504g の間（500 ± 4）
- 99.7% が 494g から 506g の間（500 ± 6）

494g より軽い、または 506g より重い弁当は異常です（確率 0.3% 未満）。

---

## GenJAX でのガウス分布

GenJAX を使って、とんかつ弁当の重量をガウス分布でモデル化してみましょう。

<!-- validate: tol=0.1 -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, normal
import jax.random as random

@gen
def tonkatsu_weight():
    """Model: tonkatsu bentos ~ N(500, 4)"""
    # Mean = 500g, Standard deviation = 2g (so variance = 4)
    mu = 500.0
    sigma = 2.0

    weight = normal(mu, sigma) @ "weight"
    return weight

# Simulate 10,000 bentos
key = random.PRNGKey(42)
weights = []

for _ in range(10000):
    key, subkey = random.split(key)
    # Run the model once: model.simulate(key, args) returns a trace; args is () here.
    trace = tonkatsu_weight.simulate(subkey, ())
    weights.append(trace.get_retval())

weights = jnp.array(weights)

print(f"Simulated mean: {jnp.mean(weights):.2f}g")
print(f"Simulated std dev: {jnp.std(weights):.2f}g")
print(f"Theoretical mean: 500.00g")
print(f"Theoretical std dev: 2.00g")
```

**出力:**
```
Simulated mean: 499.98g
Simulated std dev: 2.01g
Theoretical mean: 500.00g
Theoretical std dev: 2.00g
```

ぴったり一致しています！大数の法則がここでも働いています。

### 68-95-99.7 規則の検証

<!-- validate: tol=1.0 -->
```python
# Count how many fall within each range
import jax.numpy as jnp

within_1_sigma = jnp.sum((weights >= 498) & (weights <= 502)) / len(weights)
within_2_sigma = jnp.sum((weights >= 496) & (weights <= 504)) / len(weights)
within_3_sigma = jnp.sum((weights >= 494) & (weights <= 506)) / len(weights)

print(f"Within 1σ (498-502g): {within_1_sigma:.1%} (expect 68%)")
print(f"Within 2σ (496-504g): {within_2_sigma:.1%} (expect 95%)")
print(f"Within 3σ (494-506g): {within_3_sigma:.1%} (expect 99.7%)")
```

**出力:**
```
Within 1σ (498-502g): 68.2% (expect 68%)
Within 2σ (496-504g): 95.4% (expect 95%)
Within 3σ (494-506g): 99.7% (expect 99.7%)
```

経験則が成り立っています！

---

## さまざまなガウス分布の可視化

μ と σ が形状にどう影響するかを見てみましょう。

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Create a range of x values
x = jnp.linspace(490, 510, 1000)

# Define the Gaussian PDF function
```

<details>
<summary>可視化コードを表示する</summary>

```python
import jax.numpy as jnp

def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * jnp.sqrt(2 * jnp.pi))) * \
           jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Plot different means (same variance)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Different means
for mu in [495, 500, 505]:
    y = gaussian_pdf(x, mu, 2.0)
    ax1.plot(x, y, label=f'μ={mu}, σ=2')
ax1.set_xlabel('Weight (g)')
ax1.set_ylabel('Probability Density')
ax1.set_title('Different Means (μ), Same Standard Deviation')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Different standard deviations
for sigma in [1, 2, 3]:
    y = gaussian_pdf(x, 500, sigma)
    ax2.plot(x, y, label=f'μ=500, σ={sigma}')
ax2.set_xlabel('Weight (g)')
ax2.set_ylabel('Probability Density')
ax2.set_title('Same Mean, Different Standard Deviations (σ)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gaussian_variations.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![ガウス分布のバリエーション](../../images/intro2/gaussian_variations.png)

**主な観察点:**
- **左のグラフ**: μ を変えると曲線が水平方向にシフトする（位置の変化）
- **右のグラフ**: σ を変えると広がりが変わる（σ が小さい = 高く/狭く、σ が大きい = 低く/広く）

---

## Chibany の弁当に戻る

第1章の謎を覚えていますか？これで、より現実的にモデル化できます。

**とんかつ弁当**: N(500, 4)（平均 500g、標準偏差 2g）
**ハンバーガー弁当**: N(350, 4)（平均 350g、標準偏差 2g）

{{% notice style="info" title="注記：説明用コード" %}}
以下のコードは混合モデルの**概念**を示しています。JAX の関数型設計のため、実際に動作する実装には高度なテクニックが必要です（完全な動作バージョンについては [インタラクティブ Colab ノートブック](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/gaussian_bayesian_interactive_exploration.ipynb) を参照してください）。

学習目的として、この簡略版ではモデリングのロジックを示しています。
{{% /notice %}}

<!-- validate: tol=6.0 -->
```python
from genjax import gen, bernoulli, normal
import jax.numpy as jnp
import jax.random as random

@gen
def realistic_bento():
    """A more realistic bento mixture model (conceptual)"""
    # 70% tonkatsu, 30% hamburger
    is_tonkatsu = bernoulli(0.7) @ "type"

    # Each type has Gaussian weight distribution
    # Use jnp.where to select mean based on type (JAX compatible)
    mean_weight = jnp.where(is_tonkatsu, 500.0, 350.0)
    weight = normal(mean_weight, 2.0) @ "weight"

    return weight

# Simulate 10,000 bentos
key = random.PRNGKey(42)
weights = []

for _ in range(10000):
    key, subkey = random.split(key)
    # model.simulate(key, args) runs the model once and returns a trace; args is () here.
    trace = realistic_bento.simulate(subkey, ())
    weights.append(trace.get_retval())

weights = jnp.array(weights)

print(f"Average weight: {jnp.mean(weights):.1f}g")
print(f"Expected value: {0.7 * 500 + 0.3 * 350:.1f}g")
```

**出力:**
```
Average weight: 455.1g
Expected value: 455.0g
```

次に、この混合モデルを可視化しましょう。

```python
import matplotlib.pyplot as plt
```

<details>
<summary>可視化コードを表示する</summary>

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(weights, bins=100, density=True, alpha=0.7, edgecolor='black')
plt.axvline(jnp.mean(weights), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {jnp.mean(weights):.1f}g')
plt.xlabel('Weight (g)')
plt.ylabel('Probability Density')
plt.title('Realistic Bento Mixture: Two Gaussians')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('realistic_bento_mixture.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![2つのガウス成分を持つ現実的な弁当混合モデル](../../images/intro2/realistic_bento_mixture.png)

これで、2つのピークに自然なばらつきが生まれました（500g と 350g の完全なスパイクではなく）。しかし、平均値は依然として個々の弁当が存在しない谷間に落ちています！

---

## ガウス分布が特別な理由

### 1. 中心極限定理

ガウス分布があらゆるところに現れる理由の一つ: **中心極限定理**によると、多くの独立した確率変数を合計すると、個々の変数がどのような分布であっても、結果はガウス分布に近づきます。

**例**: 弁当の重量は次のような要素によって決まるかもしれません:
- ご飯の量（ランダムに変動する）
- メインのタンパク質の量（ランダムに変動する）
- 野菜の量（ランダムに変動する）
- ソースの量（ランダムに変動する）
- 容器のばらつき（ランダムに変動する）

各成分がガウス分布でなくても、それらの**和**（合計重量）はガウス分布に近づく傾向があります！

### 2. 最大エントロピー分布

平均値と分散だけが与えられた場合、ガウス分布は最大エントロピーを持ちます（追加の仮定を最も少なくします）。これにより、ガウス分布は「最も謙虚な」分布になります。

### 3. 共役事前分布（もうすぐ登場！）

第4章では、ガウス分布がベイズ推論を扱いやすくする特別な数学的性質を持つことを学びます。ガウス分布のデータを観測し、ガウス事前分布を使うと、事後分布もガウス分布になります。この「共役性」により、計算がエレガントになります。

### 4. 加法性

X ~ N(μ₁, σ₁²) と Y ~ N(μ₂, σ₂²) が独立であれば:
- X + Y ~ N(μ₁ + μ₂, σ₁² + σ₂²)

平均値は足し合わさり、分散も足し合わさります。美しい！

---

## ガウス分布の累積分布関数を使った確率の計算

一様分布と同様に、累積分布関数（CDF）を使って確率を計算できます。

**問い**: とんかつ弁当の重量が 503g を超える確率は？

```python
from scipy.stats import norm

# Parameters: mean=500, std dev=2
mu, sigma = 500.0, 2.0

# P(X > 503) = 1 - P(X ≤ 503) = 1 - CDF(503)
prob_over_503 = 1 - norm.cdf(503, mu, sigma)
print(f"P(weight > 503g) = {prob_over_503:.4f}")
```

**出力:**
```
P(weight > 503g) = 0.0668
```

弁当の約 6.68% が 503g を超えます。

**シミュレーションで検証する:**
<!-- validate: tol=0.015 -->
```python
# Using our GenJAX simulation from earlier. Note: `weights` is the MIXTURE
# (70% tonkatsu near 500g, 30% hamburger near 350g). Only the tonkatsu
# component can clear 503g, so the simulated fraction is about 0.7 × 0.0668.
import jax.numpy as jnp

simulated_prob = jnp.mean(weights > 503)
print(f"Simulated P(weight > 503g) = {simulated_prob:.4f}")
```

**出力:**
```
Simulated P(weight > 503g) = 0.0424
```

これはおよそ $0.7 \times 0.0668 \approx 0.047$ です。解析的な $0.0668$ はとんかつクラスター*内*での確率であり、弁当の約70%しかとんかつではないため、全体の割合は低くなります。（すべての弁当がとんかつであれば、両者は完全に一致するでしょう。）

---

## 標準正規分布

特殊なケース: **標準正規分布**は μ = 0、σ² = 1 を持ち、N(0, 1) と表記されます。

任意のガウス分布 X ~ N(μ, σ²) は**標準化**できます:

$$Z = \frac{X - \mu}{\sigma}$$

このとき Z ~ N(0, 1) となります。この「Z スコア」は、X が平均値から何標準偏差離れているかを示します。

**例**: 504g のとんかつ弁当:
```python
x = 504
z = (x - mu) / sigma
print(f"Z-score: {z}")  # Z-score: 2.0
```

この弁当は平均値からちょうど2標準偏差上にあります。68-95-99.7 規則を使うと、これは第95パーセンタイルの範囲内にあることがわかります（珍しいですが、極めてまれではありません）。

---

## 練習問題

### 問題 1: 学生のテストの点数

テストの点数は N(75, 100)（平均 75、分散 100、標準偏差 = 10）に従います。

**a)** 65 から 85 の間に入る学生の割合は？

**b)** 第90パーセンタイルに相当する点数は？

**c)** 1000 人の学生をシミュレーションして、答えを検証してください。

<details>
<summary>解答を表示する</summary>

<!-- validate: tol=3.0 -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, normal
import jax.random as random
from scipy.stats import norm

mu, sigma = 75, 10

# Part a: P(65 < X < 85)
# This is μ ± 1σ, so we expect 68%
prob_between = norm.cdf(85, mu, sigma) - norm.cdf(65, mu, sigma)
print(f"a) P(65 < score < 85) = {prob_between:.1%}")

# Part b: 90th percentile
score_90th = norm.ppf(0.90, mu, sigma)
print(f"b) 90th percentile score: {score_90th:.1f}")

# Part c: Simulate
@gen
def student_score():
    score = normal(75.0, 10.0) @ "score"
    return score

key = random.PRNGKey(42)
scores = []

for _ in range(1000):
    key, subkey = random.split(key)
    trace = student_score.simulate(subkey, ())
    scores.append(trace.get_retval())

scores = jnp.array(scores)

sim_prob = jnp.mean((scores >= 65) & (scores <= 85))
sim_90th = jnp.percentile(scores, 90)

print(f"c) Simulated P(65-85): {sim_prob:.1%}")
print(f"   Simulated 90th percentile: {sim_90th:.1f}")
```

**出力:**
```
a) P(65 < score < 85) = 68.3%
b) 90th percentile score: 87.8
c) Simulated P(65-85): 68.1%
   Simulated 90th percentile: 87.4
```
</details>

---

### 問題 2: 品質管理

ある工場でボルトを製造しており、その長さは N(50, 0.25) mm（平均 50mm、標準偏差 0.5mm）に従います。49mm から 51mm の範囲外のボルトは不良品として却下されます。

**a)** 却下されるボルトの割合は？

**b)** 工場は不良率を 1% 未満に抑えたいと考えています。標準偏差はいくつにする必要がありますか？

<details>
<summary>解答を表示する</summary>

```python
from scipy.stats import norm

mu, sigma = 50, 0.5

# Part a: P(X < 49 or X > 51) = 1 - P(49 ≤ X ≤ 51)
prob_good = norm.cdf(51, mu, sigma) - norm.cdf(49, mu, sigma)
prob_reject = 1 - prob_good
print(f"a) Rejection rate: {prob_reject:.1%}")

# Part b: We need P(49 ≤ X ≤ 51) ≥ 0.99
# This means P(X ≤ 51) - P(X ≤ 49) ≥ 0.99
# With symmetry, P(X ≤ 49) ≈ 0.005 and P(X ≤ 51) ≈ 0.995
# So 49 must be at the 0.5th percentile, meaning (49-50)/σ = norm.ppf(0.005)
z_005 = norm.ppf(0.005)
new_sigma = (49 - 50) / z_005
print(f"b) Required std dev: {new_sigma:.3f}mm")

# Verify
prob_good_new = norm.cdf(51, 50, new_sigma) - norm.cdf(49, 50, new_sigma)
prob_reject_new = 1 - prob_good_new
print(f"   New rejection rate: {prob_reject_new:.2%}")
```

**出力:**
```
a) Rejection rate: 4.6%
b) Required std dev: 0.388mm
   New rejection rate: 1.00%
```
</details>

---

## 次のステップ

これで以下のことを理解しました:
- ガウス分布とそのパラメータ
- 68-95-99.7 規則
- GenJAX でのガウス分布の扱い方
- ガウス分布があらゆるところに現れる理由

しかし、ここで疑問が生じます: **μ と σ² がわからない場合はどうすればよいのか？**

第4章では、**ベイズ学習**を学びます。データからこれらのパラメータを推定する方法として、事前分布の信念から始め、弁当の重量を観測するにつれてそれを更新していきます。ここで確率的プログラミングが真価を発揮します！

---

{{% notice style="tip" title="重要なポイント" %}}
1. **ガウス分布**: 平均（μ）と分散（σ²）によって記述される「ベルカーブ」
2. **68-95-99.7 規則**: データの約 68%/95%/99.7% が 1/2/3 標準偏差以内に収まる
3. **普遍性**: 中心極限定理によりガウス分布はあらゆるところに現れる
4. **GenJAX**: `normal(mu, sigma)` が N(μ, σ²) からサンプリングする
5. **シミュレーション**: モンテカルロ検証が理論的確率と一致する
{{% /notice %}}

---

**次の章**: [ガウス分布によるベイズ学習 →](./04_bayesian_learning.md)
