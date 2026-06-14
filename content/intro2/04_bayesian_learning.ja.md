+++
date = "2026-06-14"
title = "ガウス分布によるベイズ学習"
weight = 4
+++

## 学習問題

Chibany には新たな課題があります。新しいサプライヤーから弁当が届くようになりましたが、そのとんかつ弁当の平均重量がわかりません。いつものサプライヤーと同様に 500g を目標にしていると思っていますが、確信はありません。もしかすると 495g を目指しているのかも？それとも 505g？

**問い**：観測データから真の平均重量をどうやって学習すればよいか？

これが **ベイズ学習** です。事前信念から始まり、データを観測し、事後信念へと更新していきます。

---

## 設定：平均未知、分散既知

まずシンプルなケースから始めましょう。次のように仮定します。
- 個々の弁当重量は X ~ N(μ, σ²) に従う
- 分散 σ² = 4（標準偏差 = 2g）は **既知** [一定の精度]
- 平均 μ は **未知** [学習したいもの]

**事前信念**：データを見る前に、Chibany は μ ~ N(500, 25) と考えています。
- 最良の推測：500g（事前分布の平均）
- 不確実性：標準偏差 5g（したがって分散 = 25）

これは「平均は 500g 前後だと思うが、±5g 程度の不確かさがある」ということを表しています。

---

## データの観測

Chibany は新しいサプライヤーの最初の弁当を計量しました：**x₁ = 497g**

**重要な洞察**：この単一の観測は μ に関する情報を含んでいます！

- μ が 500g なら、497g を観測することはそれなりに起こりうる（1.5σ 以内）
- μ が 510g なら、497g を観測することはかなり起こりにくい（6.5σ も離れている！）
- μ が 495g なら、497g を観測することは非常に起こりやすい（わずか 1σ）

観測は、データをより尤もらしくする値の方向に μ に関する信念を **シフト** させます。

---

## ベイズ更新：数式

未知パラメータに対する **ベイズの定理**：

$$p(\mu | x_1, ..., x_n) = \frac{p(x_1, ..., x_n | \mu) \cdot p(\mu)}{p(x_1, ..., x_n)}$$

**言葉で表すと**：
- **事後分布** ∝ **尤度** × **事前分布**
- データを観測した後の信念 ∝ （データがどれだけ起こりやすいか）× （観測前の信念）

{{% notice style="info" title="📘 基礎概念：ベイズの定理の拡張" %}}
**チュートリアル 1 第 5 章で学んだ** ように、ベイズの定理はエビデンスによって信念を更新するものです：

$$P(H | E) = \frac{P(E | H) \cdot P(H)}{P(E)}$$

「タクシーは青かったか？」を「Chibany が青と言った」というエビデンスのもとで答えるような **離散事象** に使いました。

**今はこれを連続パラメータへ拡張しています！**

**構造はまったく同じです：**

| チュートリアル 1（離散） | チュートリアル 3（連続） |
|----------------------|------------------------|
| $P(H \mid E)$ — 仮説に関する事後信念 | $p(\mu \mid x_1, ..., x_n)$ — パラメータに関する事後信念 |
| $P(E \mid H)$ — 仮説のもとでのエビデンスの尤度 | $p(x_1, ..., x_n \mid \mu)$ — パラメータのもとでのデータの尤度 |
| $P(H)$ — 仮説に関する事前信念 | $p(\mu)$ — パラメータに関する事前信念 |
| $P(E)$ — エビデンスの全確率 | $p(x_1, ..., x_n)$ — データの全確率 |

**ロジックは変わっていません：**
- 事前信念から始める（データを見る前）
- エビデンスで更新する（観測の尤度）
- 事後信念を得る（データを見た後）

**新しい点：** 離散確率（0.15、0.85）ではなく、連続密度（ガウス分布）を扱っています。しかし **信念更新の原理** はまったく同じです！

[← チュートリアル 1 第 5 章でベイズの定理を復習する](../../intro/05_bayes/)
{{% /notice %}}

### ガウス-ガウス共役事前分布

ここが魔法のポイントです：**事前分布がガウス分布で尤度もガウス分布の場合、事後分布もガウス分布になります！**

これを **共役性** と呼び、計算を非常にエレガントにします。

**事前分布**：μ ~ N(μ₀, σ₀²)
**尤度**：X | μ ~ N(μ, σ²) [σ² は既知]

**x₁, x₂, ..., xₙ を観測した後：**

$$\mu | x_1, ..., x_n \sim N(\mu_n, \sigma_n^2)$$

ここで：

$$\mu_n = \frac{\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}$$

$$\frac{1}{\sigma_n^2} = \frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}$$

**慌てないでください！** GenJAX がこれを処理します。ただし直感を理解しておきましょう。

---

## 直感：精度加重平均

事後平均 μₙ は以下の **加重平均** です：
- 事前平均 μ₀
- 標本平均 $\bar{x}$

重みは **精度**（分散の逆数）に依存します：
- 事前精度：$\frac{1}{\sigma_0^2}$
- データ精度：$\frac{n}{\sigma^2}$（データが多い = 精度が高い）

### わかりやすく言うと

「更新された信念は、事前に思っていたこと（事前分布）とデータが示すこと（標本平均）の妥協点です。最初の確信が強かった（σ₀² が小さい）場合や手元のデータが少ない（n が小さい）場合ほど、事前分布に近い値を維持します。データが多い（n が大きい）場合や最初の不確実性が高かった（σ₀² が大きい）場合ほど、データを信頼します。」

---

## 例題を解く

**事前分布**：μ ~ N(500, 25) [μ₀ = 500, σ₀² = 25]
**データ分散**：σ² = 4
**観測値**：x₁ = 497g、したがって $\bar{x}$ = 497、n = 1

**事後分散**：
$$\frac{1}{\sigma_1^2} = \frac{1}{25} + \frac{1}{4} = 0.04 + 0.25 = 0.29$$
$$\sigma_1^2 = \frac{1}{0.29} \approx 3.45$$
$$\sigma_1 \approx 1.86$$

**事後平均**：
$$\mu_1 = \frac{\frac{500}{25} + \frac{1 \cdot 497}{4}}{\frac{1}{25} + \frac{1}{4}} = \frac{20 + 124.25}{0.29} = \frac{144.25}{0.29} \approx 497.4$$

**結果**：497g を観測した後、Chibany の信念は μ ~ N(497.4, 3.45) に更新されます。

**解釈**：
- 最良の推測は 500g から 497.4g にシフトした（データの方向へ移動）
- 不確実性は σ₀ = 5g から σ₁ ≈ 1.86g に減少した（より確信が持てるようになった）

---

## GenJAX での実装

ベイズ学習モデルを構築しましょう：

<!-- validate: tol=0.1 -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, normal, ChoiceMap
import jax.random as random

# Known parameters
DATA_STD = 2.0   # observation noise: N(mu, 4) means std 2

@gen
def generative_model():
    """Full model: prior on the mean, then one observation given that mean."""
    # Prior belief about the mean
    mu = normal(500.0, 5.0) @ "mu"           # Prior: N(500, 25)
    # Generate the observation from N(mu, 4)
    weight = normal(mu, DATA_STD) @ "weight_0"
    return mu

# Observe one bento: 497g. We CONDITION on it with a ChoiceMap (the modern way to
# pin an addressed random choice to an observed value), then importance-sample.
observed = ChoiceMap.d({"weight_0": 497.0})

# Run importance sampling to approximate the posterior over mu
key = random.PRNGKey(42)
num_samples = 10000

# model.importance(key, constraints, args) returns (trace, log_weight): the trace is
# a sample with weight_0 pinned to 497, and log_weight scores how well its mu explains
# the data. A weighted average of mu over many samples approximates the posterior.
mus = []
log_weights = []
for _ in range(num_samples):
    key, subkey = random.split(key)
    trace, log_w = generative_model.importance(subkey, observed, ())
    mus.append(trace.get_choices()["mu"])
    log_weights.append(log_w)

mus = jnp.array(mus)
log_weights = jnp.array(log_weights)

# Normalize the importance weights (log-space, like the GenJAX tutorial), then take
# the weighted mean and std of mu.
w = jnp.exp(log_weights - log_weights.max())
w = w / w.sum()
post_mean = jnp.sum(w * mus)
post_std = jnp.sqrt(jnp.sum(w * (mus - post_mean) ** 2))

print(f"Posterior mean: {post_mean:.2f}g")
print(f"Posterior std dev: {post_std:.2f}g")
print(f"Theoretical posterior mean: 497.4g")
print(f"Theoretical posterior std dev: 1.86g")
```

**出力：**
```
Posterior mean: 497.41g
Posterior std dev: 1.85g
Theoretical posterior mean: 497.4g
Theoretical posterior std dev: 1.86g
```

**注意**：上記は概念的な構造を示しています。実際には GenJAX の重点サンプリングは重みの正規化が必要な場合があります。直接解析的に更新することでシンプルにしましょう：

```python
import jax.numpy as jnp

def gaussian_gaussian_update(prior_mu, prior_var, data, data_var):
    """
    Analytical Bayesian update for Gaussian-Gaussian conjugate prior

    Args:
        prior_mu: Prior mean
        prior_var: Prior variance
        data: List of observations
        data_var: Known data variance

    Returns:
        posterior_mu, posterior_var
    """
    n = len(data)
    sample_mean = jnp.mean(jnp.array(data))

    # Precision-weighted update
    prior_precision = 1.0 / prior_var
    data_precision = n / data_var

    posterior_precision = prior_precision + data_precision
    posterior_var = 1.0 / posterior_precision

    posterior_mu = posterior_var * (prior_precision * prior_mu +
                                     data_precision * sample_mean)

    return posterior_mu, posterior_var

# Apply to our example
prior_mu, prior_var = 500.0, 25.0
data = [497.0]
data_var = 4.0

post_mu, post_var = gaussian_gaussian_update(prior_mu, prior_var, data, data_var)
post_std = jnp.sqrt(post_var)

print(f"After 1 observation:")
print(f"  Posterior mean: {post_mu:.2f}g")
print(f"  Posterior std dev: {post_std:.2f}g")
```

**出力：**
```
After 1 observation:
  Posterior mean: 497.41g
  Posterior std dev: 1.86g
```

手計算との完全一致！

---

## 逐次学習：データを増やす

今度は Chibany が同じサプライヤーからさらに 9 個の弁当を計量します：

```python
# Additional observations
import jax.numpy as jnp

all_data = [497.0, 498.5, 496.0, 499.0, 497.5, 498.0, 496.5, 497.0, 498.5, 497.5]

# Start with prior
mu, var = 500.0, 25.0

print(f"Prior: N({mu:.2f}, {var:.2f})")
print(f"  Mean: {mu:.2f}g, Std dev: {jnp.sqrt(var):.2f}g\n")

# Update with each observation sequentially
for i, obs in enumerate(all_data, 1):
    mu, var = gaussian_gaussian_update(mu, var, [obs], data_var)
    std = jnp.sqrt(var)
    print(f"After observation {i} (x={obs}g):")
    print(f"  Posterior: N({mu:.2f}, {var:.2f})")
    print(f"  Mean: {mu:.2f}g, Std dev: {std:.2f}g")
```

**出力：**
```
Prior: N(500.00, 25.00)
  Mean: 500.00g, Std dev: 5.00g

After observation 1 (x=497.0g):
  Posterior: N(497.41, 3.45)
  Mean: 497.41g, Std dev: 1.86g
After observation 2 (x=498.5g):
  Posterior: N(497.92, 1.85)
  Mean: 497.92g, Std dev: 1.36g
After observation 3 (x=496.0g):
  Posterior: N(497.31, 1.27)
  Mean: 497.31g, Std dev: 1.13g
After observation 4 (x=499.0g):
  Posterior: N(497.72, 0.96)
  Mean: 497.72g, Std dev: 0.98g
After observation 5 (x=497.5g):
  Posterior: N(497.67, 0.78)
  Mean: 497.67g, Std dev: 0.88g
After observation 6 (x=498.0g):
  Posterior: N(497.73, 0.65)
  Mean: 497.73g, Std dev: 0.81g
After observation 7 (x=496.5g):
  Posterior: N(497.56, 0.56)
  Mean: 497.56g, Std dev: 0.75g
After observation 8 (x=497.0g):
  Posterior: N(497.49, 0.49)
  Mean: 497.49g, Std dev: 0.70g
After observation 9 (x=498.5g):
  Posterior: N(497.60, 0.44)
  Mean: 497.60g, Std dev: 0.66g
After observation 10 (x=497.5g):
  Posterior: N(497.59, 0.39)
  Mean: 497.59g, Std dev: 0.63g
```

**主な観察点**：
1. 平均はデータの平均値（約 497.6g）に向かってシフトする
2. 観測のたびに不確実性が減少する
3. 10 回の観測後、σ は 5.0g から 0.61g に低下した（はるかに確信が高まった！）

---

## 学習過程の可視化

```python
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm

# Prior
import jax.numpy as jnp

x_range = jnp.linspace(490, 505, 1000)
prior_pdf = scipy_norm.pdf(x_range, 500, 5)

# After 1, 5, and 10 observations
results = []
mu, var = 500.0, 25.0
for i, obs in enumerate(all_data):
    mu, var = gaussian_gaussian_update(mu, var, [obs], data_var)
    if i + 1 in [1, 5, 10]:
        results.append((i + 1, mu, jnp.sqrt(var)))

# Plot
```

<details>
<summary>可視化コードを表示</summary>

```python
import jax.numpy as jnp

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(x_range, prior_pdf, 'k--', linewidth=2, label='Prior: N(500, 25)')

colors = ['blue', 'green', 'red']
for (n_obs, mu, std), color in zip(results, colors):
    post_pdf = scipy_norm.pdf(x_range, mu, std)
    ax.plot(x_range, post_pdf, color=color, linewidth=2,
            label=f'After {n_obs} obs: N({mu:.1f}, {std**2:.2f})')

# Mark the true sample mean
sample_mean = jnp.mean(jnp.array(all_data))
ax.axvline(sample_mean, color='purple', linestyle=':', linewidth=2,
           label=f'Sample mean: {sample_mean:.2f}g')

ax.set_xlabel('Mean weight μ (g)')
ax.set_ylabel('Probability Density')
ax.set_title('Bayesian Learning: Posterior Distribution Updates')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('bayesian_learning_posterior.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![ベイズ学習：事後分布の更新](../../images/intro2/posterior_updates.png)

**プロットが示すストーリー**：
- **黒の破線**：事前信念（幅広く、500g を中心）
- **青**：1 回の観測後（データに向かってシフト、より狭くなった）
- **緑**：5 回の観測後（標本平均に近づき、かなり狭くなった）
- **赤**：10 回の観測後（標本平均にほぼ一致、非常に狭い）
- **紫の点線**：真の標本平均（497.65g）

データが蓄積されるにつれて、事後分布は真の値に収束します！

---

## 🔬 探索演習：パラメータが学習に与える影響

ベイズ更新のメカニズムを理解したところで、主要なパラメータが学習過程にどのような影響を与えるかを体系的に探りましょう：

1. **尤度分散**（σ²_x）：測定の精度はどれくらいか？
2. **観測数**（N）：どれだけのデータがあるか？

### インタラクティブな探索

**📓 インタラクティブノートブックを開く**：[Colab で開く：`gaussian_bayesian_interactive_exploration.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/gaussian_bayesian_interactive_exploration.ipynb)

このノートブックでは次のことができます：
- **インタラクティブスライダー**でパラメータをリアルタイムに調整
- 事前分布 → 事後分布 → 予測分布の変化の **視覚的比較**
- 収束を示す **逐次学習の可視化**
- 実践的な体験のための **GenJAX 実装**

**探索すべき重要な問い**：
- σ²_x が非常に小さい（精密な測定）場合と非常に大きい（ノイズの多い測定）場合では何が起きるか？
- N が 1 から 10 の観測に増えるにつれ、事後分布はどう変わるか？
- データが事前分布を「圧倒する」のはいつか？
- 予測分布が常に事後分布より広いのはなぜか？

### 課題問題

**📝 詳細な解答を確認する**：[Colab で開く：`solution_1_gaussian_bayesian_update.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/solution_1_gaussian_bayesian_update.ipynb)

この課題では以下を体系的に探求します：
- **パート (a)**：事前分布の可視化
- **パート (b)**：尤度分散の影響（σ²_x = 0.25 対 4）
- **パート (c)**：観測数の影響（N=1 対 N=5）
- **GenJAX による検証**：解析的な公式とシミュレーションの比較

**学習目標**：
- 精度加重平均への直感を養う
- 分散とサンプルサイズのトレードオフを理解する
- 大数の法則を実際に確認する（N → ∞ で事後分布 → 標本平均）
- 数式をコードに変換する練習をする

{{% notice style="success" title="💡 なぜこれが重要か" %}}
これらのパラメータの影響を理解することは以下のために不可欠です：
- **実験計画**：何サンプル必要か？
- **センサーのキャリブレーション**：測定ノイズは推論にどう影響するか？
- **事前分布の選択**：事前分布が優勢になるのはいつか、圧倒されるのはいつか？
- **不確実性の定量化**：推定値にどれくらい確信を持てるか？

これらのノートブックで、現実世界のあらゆるベイズ応用に現れる概念を実践的に体験できます！
{{% /notice %}}

---

## 予測分布

Chibany は今、こう問います：**「次の弁当の重量はどれくらいになると予測すべきか？」**

これには **事後予測分布** が必要です：

$$p(x_{new} | x_1, ..., x_n)$$

「これまでに学習したことを踏まえて、新しい観測の確率分布はどうなるか？」

### 数式

μ に関する不確実性を積分で消去します：

$$p(x_{new} | data) = \int p(x_{new} | \mu) \cdot p(\mu | data)   d\mu$$

ガウス-ガウスモデルでは、これもガウス分布になります！

$$X_{new} | data \sim N(\mu_n, \sigma^2 + \sigma_n^2)$$

**重要な洞察**：予測分散は次の 2 つを合算します：
- データ分散 σ²（弁当固有のばらつき）
- 事後分散 σₙ²（μ に関する残余の不確実性）

### 例

10 回の観測後、事後分布 N(497.65, 0.37) が得られています：

```python
# Posterior from before
import jax.numpy as jnp

post_mu = 497.65
post_var = 0.37

# Predictive distribution
pred_mu = post_mu  # Same mean
pred_var = data_var + post_var  # 4.0 + 0.37 = 4.37
pred_std = jnp.sqrt(pred_var)

print(f"Posterior for μ: N({post_mu:.2f}, {post_var:.2f})")
print(f"Predictive for next X: N({pred_mu:.2f}, {pred_var:.2f})")
print(f"  Predictive std dev: {pred_std:.2f}g")
```

**出力：**
```
Posterior for μ: N(497.65, 0.37)
Predictive for next X: N(497.65, 4.37)
  Predictive std dev: 2.09g
```

**解釈**：次の弁当の重量は約 497.65g ± 2.09g になると予測されます。

---

## GenJAX での予測分布の実装

<!-- validate: tol=0.1 -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, normal
import jax.random as random

@gen
def posterior_predictive(post_mu, post_var, data_var):
    """
    Sample from posterior predictive distribution
    """
    # First, sample a μ from the posterior
    mu = normal(post_mu, jnp.sqrt(post_var)) @ "mu"

    # Then, sample a new observation given that μ
    x_new = normal(mu, jnp.sqrt(data_var)) @ "x_new"

    return x_new

# Posterior from before (after observing 497g)
post_mu = 497.65
post_var = 0.37
data_var = 4.0

# Theoretical posterior-predictive: same mean, variance = posterior var + data var
pred_mu = post_mu
pred_std = jnp.sqrt(post_var + data_var)

# Simulate 10,000 predictions
key = random.PRNGKey(42)
predictions = []

for _ in range(10000):
    key, subkey = random.split(key)
    # model.simulate(key, args) runs the model once; args is the tuple of arguments.
    trace = posterior_predictive.simulate(subkey, (post_mu, post_var, data_var))
    predictions.append(trace.get_retval())

predictions = jnp.array(predictions)

print(f"Simulated predictive mean: {jnp.mean(predictions):.2f}g")
print(f"Simulated predictive std: {jnp.std(predictions):.2f}g")
print(f"Theoretical predictive mean: {pred_mu:.2f}g")
print(f"Theoretical predictive std: {pred_std:.2f}g")
```

**出力：**
```
Simulated predictive mean: 497.65g
Simulated predictive std: 2.10g
Theoretical predictive mean: 497.65g
Theoretical predictive std: 2.09g
```

完全一致！

---

## 共役性が重要な理由

ガウス-ガウスの設定は **共役** です。つまり：
- 事前分布はガウス分布
- 尤度はガウス分布
- **事後分布もガウス分布**

これには大きな利点があります：
1. **閉形式の更新**：複雑な推論アルゴリズムが不要
2. **逐次学習**：1 回の観測ごとに更新可能
3. **解釈しやすい**：精度加重平均は明確な意味を持つ
4. **計算効率が良い**：2 つのパラメータ（μₙ、σₙ²）を更新するだけでよい

すべての事前-尤度のペアが共役というわけではありません。共役でない場合は近似手法が必要になります（後のチュートリアルで見ていきます）。

---

## 全体像：パラメータと観測値

以下を区別することが重要です：

**パラメータ**（未知、学習するもの）：
- μ（サプライヤーが目標とする平均重量）
- データを観測した後に事後分布で記述される

**観測値**（既知、収集するもの）：
- x₁, x₂, ..., xₙ（実際に測定した弁当の重量）
- パラメータが与えられたもとでの尤度分布で記述される

**ベイズ的アプローチ**：未知のパラメータを分布を持つ確率変数として扱い、データでその分布を更新します。

---

## 練習問題

### 問題 1：新しいコーヒーショップ

新しいコーヒーショップはエスプレッソショットの平均が 30ml だと主張しています。あなたはそれを信じていますが不確かです。事前分布：μ ~ N(30, 9)（標準偏差 = 3ml）。

5 ショットを計量しました：[28.5, 29.0, 31.0, 29.5, 30.5] ml。

既知：各ショットの分散は 4（標準偏差 = 2ml）。

**a)** 5 回の観測後の μ の事後分布は？

**b)** μ の 95% 信用区間は？

**c)** 次のショットの予測分布は？

<details>
<summary>解答を表示</summary>

```python
# Prior
import jax.numpy as jnp

prior_mu, prior_var = 30.0, 9.0

# Data
data = jnp.array([28.5, 29.0, 31.0, 29.5, 30.5])
data_var = 4.0
n = len(data)

# Posterior calculation
post_mu, post_var = gaussian_gaussian_update(prior_mu, prior_var, data, data_var)
post_std = jnp.sqrt(post_var)

# Part a
print(f"a) Posterior: N({post_mu:.2f}, {post_var:.2f})")
print(f"   Mean: {post_mu:.2f}ml, Std dev: {post_std:.2f}ml")

# Part b: 95% credible interval (±1.96σ)
lower = post_mu - 1.96 * post_std
upper = post_mu + 1.96 * post_std
print(f"b) 95% credible interval: [{lower:.2f}, {upper:.2f}] ml")

# Part c: Predictive distribution
pred_var = data_var + post_var
pred_std = jnp.sqrt(pred_var)
print(f"c) Predictive: N({post_mu:.2f}, {pred_var:.2f})")
print(f"   Mean: {post_mu:.2f}ml, Std dev: {pred_std:.2f}ml")
```

**出力：**
```
a) Posterior: N(29.72, 0.73)
   Mean: 29.72ml, Std dev: 0.86ml
b) 95% credible interval: [28.04, 31.40] ml
c) Predictive: N(29.72, 4.73)
   Mean: 29.72ml, Std dev: 2.18ml
```
</details>

---

### 問題 2：矛盾するデータからの学習

強い事前信念があるとします：μ ~ N(500, 1)（500g に非常に確信がある）。

3 個の弁当を観測しました：[490, 491, 489]（いずれもずっと軽い！）。

データ分散：4。

**a)** 事後分布は？

**b)** 事後分布がなぜ 490g にもっと大きくシフトしなかったのか？

**c)** 事前分布よりデータを信頼するためには何回の観測が必要か？

<details>
<summary>解答を表示</summary>

```python
# Strong prior
import jax.numpy as jnp

prior_mu, prior_var = 500.0, 1.0  # Very confident!

# Contradictory data
data = jnp.array([490.0, 491.0, 489.0])
data_var = 4.0
n = len(data)
sample_mean = jnp.mean(data)

# Posterior
post_mu, post_var = gaussian_gaussian_update(prior_mu, prior_var, data, data_var)
post_std = jnp.sqrt(post_var)

# Part a
print(f"a) Prior: N({prior_mu:.0f}, {prior_var:.2f}) [very confident]")
print(f"   Sample mean: {sample_mean:.1f}g")
print(f"   Posterior: N({post_mu:.2f}, {post_var:.2f})")
print(f"   Mean: {post_mu:.2f}g, Std dev: {post_std:.2f}g")

# Part b
print(f"\nb) Prior precision: {1/prior_var:.2f}")
print(f"   Data precision (n=3): {n/data_var:.2f}")
print(f"   Prior precision is stronger, so posterior stays near 500g")

# Part c: When would data dominate?
# We want data precision > prior precision
# n/data_var > 1/prior_var
# n > data_var/prior_var
n_needed = jnp.ceil(data_var / prior_var).astype(int)
print(f"\nc) Need n > {n_needed} observations for data to dominate")

# Verify with n=5
data_more = jnp.array([490.0, 491.0, 489.0, 490.5, 489.5])
post_mu_more, post_var_more = gaussian_gaussian_update(
    prior_mu, prior_var, data_more, data_var
)
print(f"   With n=5: Posterior mean = {post_mu_more:.2f}g (shifted more)")
```

**出力：**
```
a) Prior: N(500, 1.00) [very confident]
   Sample mean: 490.0g
   Posterior: N(495.71, 0.57)
   Mean: 495.71g, Std dev: 0.76g

b) Prior precision: 1.00
   Data precision (n=3): 0.75
   Prior precision is stronger, so posterior stays near 500g

c) Need n > 4 observations for data to dominate
   With n=5: Posterior mean = 494.44g (shifted more)
```
</details>

---

## 🎯 プレビュー：ガウス混合による分類

**単一のガウス分布** に関する信念の更新方法を学びました。しかし、第 1 章で登場した Chibany の謎の弁当を覚えていますか？それらは **2 種類の混合**（とんかつとハンバーグ）から来ていました。

### 分類問題

Chibany が不透明な 425g の弁当を受け取ったとします。どちらの種類でしょうか？
- **とんかつ弁当**：重量 ~ N(500, 100)
- **ハンバーグ弁当**：重量 ~ N(350, 100)
- **事前信念**：とんかつ 70%、ハンバーグ 30%

これは **混合モデル** の問題で、以下が必要です：
1. 観測した重量からカテゴリを **推論**：P(カテゴリ | 重量)
2. 連続的な尤度で **ベイズの定理を使用**
3. **決定境界を理解**：P(とんかつ | x) = 0.5 になるのはどこか？

### インタラクティブな探索

**📓 混合モデルを探索する**：[Colab で開く：`gaussian_bayesian_interactive_exploration.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/gaussian_bayesian_interactive_exploration.ipynb)（パート 2）

このセクションでは：
- **インタラクティブな分類**：P(c=1|x) が x に応じてどう変わるかを確認
- **事前分布の影響**：θ（事前確率）が決定境界をどうシフトさせるか？
- **分散の影響**：カテゴリが異なるばらつきを持つ場合は何が起きるか？
- **周辺分布**：重み付き混合分布 p(x) を可視化

**📝 詳細な解答**：[Colab で開く：`solution_2_gaussian_clusters.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/solution_2_gaussian_clusters.ipynb)

この課題の内容：
- **パート (a)**：ベイズの定理を使って P(c=1|x) を導出
- **パート (b)**：事前分布と分散が分類に与える影響
- **パート (c)**：周辺分布 p(x) の導出
- **パート (d)**：二峰性対単峰性の混合分布の理解

{{% notice style="info" title="🔗 概念の接続" %}}
**第 1 章より**：離散混合（70% × 500g + 30% × 350g）の E[X] を計算しました。

**今は**：連続混合に対して P(カテゴリ | 観測値) を計算しています！

**進行の流れ**：
1. **第 1 章**：離散混合、既知のカテゴリ → 期待値を計算
2. **第 4 章（この章）**：単一ガウス分布 → データからパラメータを学習
3. **プレビュー（問題 2）**：既知の混合パラメータ → 隠れたカテゴリを推論
4. **第 5 章**（次回）：未知の混合パラメータ → すべてを学習！

このプレビュー問題は、単一成分の学習から完全な混合モデルの推論へと橋渡しします。
{{% /notice %}}

### なぜこれが重要か

混合モデルはあらゆる場所に登場します：
- **生物学**：測定値から細胞の種類を分類
- **金融**：市場のレジーム（強気 vs. 弱気）を識別
- **コンピュータービジョン**：色のクラスターで画像をセグメント化
- **自然言語処理**：文書のトピックモデリング

ガウス分布を用いた分類の理解は、クラスタリング、分類、教師なし学習の基礎となります！

---

## 次は何を学ぶか？

これまでに理解したこと：
- 共役事前分布を用いたベイズ学習
- データが届くたびに信念を更新する方法
- 事後予測分布
- 共役性が計算をエレガントにする理由
- **プレビュー**：混合モデルで観測値を分類する方法

しかし、私たちは **1 つの成分**（単一のガウス分布）または **既知の混合パラメータ** しか扱っていません。**パラメータが未知の複数の成分** がある場合はどうすればよいでしょうか？

第 5 章では完全な問題に取り組みます：どの弁当がとんかつでどれがハンバーグかを学習しながら、同時に各種類の平均重量も学習します！

---

{{% notice style="tip" title="重要なまとめ" %}}
1. **ベイズ学習**：事前分布から始まり → データを観測 → 事後分布へ更新
2. **共役性**：ガウス事前分布 + ガウス尤度 = ガウス事後分布
3. **精度加重**：事後分布は事前分布とデータの加重平均
4. **逐次学習**：1 回の観測ごとに更新可能
5. **予測分布**：事後の不確実性 + データ分散を合算
6. **GenJAX**：解析的な更新または重点サンプリングで実装
{{% /notice %}}

---

**次の章**：[ガウス混合モデル →](./05_mixture_models.md)
