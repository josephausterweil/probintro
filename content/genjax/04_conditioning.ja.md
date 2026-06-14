+++
title = "条件付けと推論"
weight = 5
+++

## シミュレーションから推論へ

これまで、GenJAX を使って結果を**生成**してきました――起こりうることをシミュレートしてきたのです。

次は**推論**を学びます――観測結果から原因へと逆方向に論理を進めることです。

これが確率的プログラミングの核心です！

![Chibany が考えている](images/chibanylayingdown.png)

{{% notice style="tip" title="📓 インタラクティブノートブックあり" %}}
**実践的な学習を好む方へ？** この章には、Bayesian 推論をインタラクティブに体験できるサンプルコード・可視化・演習付きの**[Jupyter ノートブック（Colab で開く）](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/bayesian_learning.ipynb)**が用意されています。先にノートブックを進めてからここに戻ってきても、両方を並行して使っても構いません！
{{% /notice %}}

---

## 復習：条件付き確率

確率チュートリアルで学んだ**条件付き確率**を思い出しましょう：

> **「$B$ が観測されたとき、$A$ の確率はいくらか？」**

**記法：** $P(A \mid B)$

**意味：** 結果空間を $B$ に含まれる結果のみに制限し、その制限された空間内での $A$ の確率を計算する。

**公式：** $P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{|A \cap B|}{|B|}$

{{% notice style="info" title="📘 基礎概念：制限としての条件付け" %}}
**チュートリアル 1・第 4 章**から、条件付き確率とは**結果空間を制限すること**であることを思い出してください：

$$P(A \mid B) = \frac{|A \cap B|}{|B|}$$

**核心的なアイデア：** $B$ が起こらなかった結果を除外し、残った空間で確率を計算する。

**チュートリアル 1 の例：**「最初の食事がとんかつだったとき、少なくとも 1 食はとんかつ」
- 元の空間：{HH, HT, TH, TT}
- 条件：最初の食事が T → {TH, TT} に制限
- 事象：少なくとも 1 食が T → 制限空間内：{TH, TT}
- 確率：2/2 = 1（残った結果はすべてとんかつ！）

**GenJAX がすること：**
- チュートリアル 1：手動で結果を除外してカウント
- チュートリアル 2：コードでシミュレーションをフィルタリングするか、`ChoiceMap` で制限

**ロジックは同じです** ―― 条件付け＝観測に合わせて可能性を制限すること！

[← チュートリアル 1・第 4 章で条件付き確率を復習](../../intro/04_conditional/)
{{% /notice %}}

---

## タクシー問題：実際の推論課題

これらのアイデアをチュートリアルの実際の問題に適用しましょう。

**シナリオ：** Chibany は夜間にひき逃げを目撃します。タクシーは青かったと言います。しかし：
- タクシーの 85% は緑、15% は青
- Chibany は 80% の確率で色を正しく識別できる

**問い：** 実際に青いタクシーだった確率は？

### なぜ驚くべき結果になるのか

多くの人の直感は「Chibany は 80% 正確だから、おそらく 80% の確率で青だろう」と言います。

**しかし答えはわずか約 41% です！**

なぜでしょうか？**ほとんどのタクシーは緑だから**です。80% の精度があっても、実際に青いタクシーが正しく識別される数より、緑のタクシーが青と誤識別される数のほうが多いのです。

これが**ベースレート無視**――集団の中でどれほど一般的かを無視すること――です。

GenJAX でこれを解いてみましょう！

---

## 生成モデル

まず、タクシー問題を GenJAX の生成関数として表現します：

```python
import jax
import jax.numpy as jnp
from genjax import gen, flip

@gen
def taxicab_model(base_rate_blue=0.15, accuracy=0.80):
    """Generate the taxi color and what Chibany says.

    Args:
        base_rate_blue: Probability a taxi is blue (default 0.15)
        accuracy: Probability Chibany identifies correctly (default 0.80)

    Returns:
        True if taxi is blue, False if green
    """

    # True taxi color
    is_blue = flip(base_rate_blue) @ "is_blue"

    # Model: What does Chibany say?
    # Probability of saying "blue" depends on the true color:
    # - If blue: says "blue" with probability = accuracy (correct identification)
    # - If green: says "blue" with probability = (1 - accuracy) (mistake)
    says_blue_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    says_blue = flip(says_blue_prob) @ "says_blue"

    return is_blue
```

**このモデルがエンコードするもの：**

1. **事前分布：** タクシーは 15% の確率で青（ベースレート）
2. **尤度：** 観測（「青と言った」）が真の色にどう依存するか
   - 青の場合：80% の確率で「青」と言う（正解）
   - 緑の場合：20% の確率で「青」と言う（間違い）
3. **完全なモデル：** 真の色と観測に関する同時分布

{{% notice style="info" title="📘 基礎概念：コードとしてのベイズの定理" %}}
**チュートリアル 1・第 5 章**から、ベイズの定理が証拠によって信念を更新することを思い出してください：

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

**タクシー問題では：**
- **仮説 (H)：** タクシーは青
- **証拠 (E)：** Chibany が「青」と言う
- **問い：** $P(\text{青} \mid \text{青と言った})$ = ?

**チュートリアル 1 のアプローチ（手計算）：**
1. $P(\text{青と言った} \mid \text{青}) \cdot P(\text{青}) = 0.80 \times 0.15 = 0.12$ を計算
2. $P(\text{青と言った} \mid \text{緑}) \cdot P(\text{緑}) = 0.20 \times 0.85 = 0.17$ を計算
3. $P(\text{青と言った}) = 0.12 + 0.17 = 0.29$ を計算
4. ベイズ適用：$P(\text{青} \mid \text{青と言った}) = \frac{0.12}{0.29} \approx 0.41$

**チュートリアル 2 のアプローチ（GenJAX）：**
1. **生成モデルを定義**（事前分布 + 尤度）
2. **観測を指定**（青と言った）
3. **GenJAX に事後分布を自動計算させる！**

**構造は同じです：**
- `is_blue = flip(0.15)` → 事前分布：$P(\text{青})$
- `says_blue_prob = jnp.where(is_blue, accuracy, 1 - accuracy)` → 尤度：$P(\text{青と言った} \mid \text{青})$
- GenJAX の条件付け → 事後分布を計算：$P(\text{青} \mid \text{青と言った})$

**重要な洞察：** GenJAX がベイズの定理の代数計算をすべてやってくれます！生成ストーリー（事前分布 + 尤度）を書くだけで、条件付けが事後分布を与えてくれます。

[← チュートリアル 1・第 5 章でベイズの定理を復習](../../intro/05_bayes/)
{{% /notice %}}

---

## 推論への 3 つのアプローチ

GenJAX は条件付き確率を計算する 3 つの方法を提供します：

### アプローチ 1：フィルタリング（棄却サンプリング）

多くのトレースを生成し、観測に一致するものだけを残す。

**疑似コード：**
```
1. Generate many traces
2. Keep only traces where observation is true
3. Among those, count how many satisfy the query
4. Calculate the ratio
```

これは**モンテカルロ条件付き確率**――集合を使って手作業で行ったこととまったく同じです！

### アプローチ 2：`generate()` を使った条件付け

GenJAX には観測を指定するための組み込みサポートがあります。観測値を含む**チョイスマップ**を提供すると、GenJAX がその観測と一致するトレースを生成します。

### アプローチ 3：完全な推論（重要度サンプリング、MCMC）

より高度な手法（このチュートリアルの範囲外）。観測が稀な場合により効率的です。

**この章ではアプローチ 1 と 2 に焦点を当てます** ―― 最も直感的な手法です。

{{% notice style="success" title="📐→💻 数学からコードへの変換" %}}
**ベイズ推論が GenJAX にどう変換されるか：**

| 数学的概念 | 数学的記法 | GenJAX コード |
|--------------|----------------------|-------------|
| **事前分布** | $P(H)$ | `flip(0.15) @ "is_blue"` |
| **尤度** | $P(E \mid H)$ | `jnp.where(is_blue, accuracy, 1-accuracy)` |
| **証拠** | $P(E)$ | GenJAX が自動計算 |
| **事後分布** | $P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)}$ | 条件付けの結果 |
| **観測** | $E$ = "青と言った" | `ChoiceMap.d({"says_blue": True})` |
| **推論クエリ** | $P(\text{is\_blue} \mid \text{says\_blue})$ | `mean(posterior_samples)` |

**3 つの等価な推論アプローチ：**

| アプローチ | 数学的アイデア | GenJAX の実装 |
|----------|------------------|----------------------|
| **1. フィルタリング** | 同時分布からサンプリング、$E$ に一致するものを残す | `says_blue == 1` のトレースをフィルタ |
| **2. generate()** | $P(H \mid E)$ から直接サンプリング | `model.generate(key, observation, args)` |
| **3. importance()** | 重み付きサンプリング | `target.importance(key, n_particles)` |

**重要な洞察：**
- **生成モデル＝事前分布＋尤度** ―― @gen 関数が両方をエンコード
- **条件付け＝事後分布の計算** ―― GenJAX がベイズの定理の計算を実行
- **3 つの手法はすべて同じものを計算する** ―― 効率性が異なるだけ
- **ベースレートが重要！** ―― 事前分布 P(H) が事後分布 P(H|E) に強く影響
{{% /notice %}}

---

## アプローチ 1：フィルタリング（棄却サンプリング）

多くのシナリオを生成し、観測にフィルタリングしてタクシー問題を解いてみましょう。

### ステップ 1：多くのシナリオを生成する

```python
# Generate 100,000 scenarios
import jax.numpy as jnp

key = jax.random.key(42)
keys = jax.random.split(key, 100000)

def run_scenario_vec(k):
    trace = taxicab_model.simulate(k, (0.15, 0.80))
    choices = trace.get_choices()
    return jnp.array([choices['is_blue'], choices['says_blue']])

scenarios = jax.vmap(run_scenario_vec)(keys)
is_blue = scenarios[:, 0]
says_blue = scenarios[:, 1]
```

### ステップ 2：観測でフィルタリングする

**観測：** Chibany が「青」と言った

```python
# Keep only scenarios where Chibany says "blue"
import jax.numpy as jnp

observation_satisfied = says_blue == 1

n_says_blue = jnp.sum(observation_satisfied)
print(f"Scenarios where Chibany says blue: {n_says_blue} / {len(scenarios)}")
```

**出力（例）：**
```
Scenarios where Chibany says blue: 29017 / 100000
```

**なぜ約 29% なのか？**
- $P(\text{青と言った}) = P(\text{青}) \cdot P(\text{青と言った} \mid \text{青}) + P(\text{緑}) \cdot P(\text{青と言った} \mid \text{緑})$
- $= 0.15 \times 0.80 + 0.85 \times 0.20 = 0.12 + 0.17 = 0.29$

### ステップ 3：真陽性をカウントする

「青」と言ったシナリオの中で、実際に青いものはいくつか？

```python
# Both says blue AND is blue
import jax.numpy as jnp

both_blue = observation_satisfied & (is_blue == 1)

n_actually_blue = jnp.sum(both_blue)
print(f"Scenarios where taxi IS blue: {n_actually_blue} / {n_says_blue}")
```

**出力（例）：**
```
Scenarios where taxi IS blue: 12038 / 29017
```

### ステップ 4：事後確率を計算する

```python
prob_blue_given_says_blue = n_actually_blue / n_says_blue
print(f"\nP(Blue | says Blue) ≈ {prob_blue_given_says_blue:.3f}")
```

**出力：**
```
P(Blue | says Blue) ≈ 0.415
```

**わずか 41.5%！** Chibany が 80% 正確でも、タクシーが本当に青だった確率は 50% 未満です！

{{% notice style="warning" title="ベースレートの影響！" %}}
**なぜそんなに低いのか？**

Chibany が 80% 正確でも、**ほとんどのタクシーは緑**（85%）です。そのため、緑のタクシーに対する 20% の誤識別率でも、実際の青いタクシーよりも**青と誤識別された緑のタクシーのほうが多い**のです！

**数字で見ると：**
- 正しく識別された青いタクシー：$0.15 \times 0.80 = 0.12$（12%）
- 誤って識別された緑のタクシー：$0.85 \times 0.20 = 0.17$（17%）

**偽陽性が真陽性より多い！**

だから事後確率はわずか 41.5% ≈ 12/(12+17) となります。
{{% /notice %}}

{{% notice style="success" title="フィルタリングのパターン" %}}
**フィルタリングによる条件付き確率：**

1. 多くのトレースを**生成**する
2. 観測に**フィルタリング**する（一致するトレースのみ残す）
3. フィルタリング後のトレース中でクエリを**カウント**する
4. 条件付き確率を得るために**割る**

これが**棄却サンプリング**――最もシンプルな推論の形式です！
{{% /notice %}}

---

## アプローチ 2：観測と共に `generate()` を使う

次は GenJAX の組み込み条件付けを使いましょう。通常こちらのほうが便利です！

### チョイスマップの作成と条件付きトレースの生成

**チョイスマップ**は名前付き確率的選択に値を指定する辞書です。これを使ってモデルを観測に条件付けます：

<!-- validate: tol=0.02 -->
```python
from genjax import ChoiceMap

# Specify that Chibany says "blue"
# Note: flip() returns boolean values, so we use True/False
import jax.numpy as jnp

observation = ChoiceMap.d({"says_blue": True})

# Generate 10,000 traces conditional on observation
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_conditional(k):
    trace, weight = taxicab_model.generate(k, observation, (0.15, 0.80))
    return trace.get_retval(), weight  # Return both value and weight

results = jax.vmap(run_conditional)(keys)
is_blue_samples = results[0]  # The sampled values
weights = results[1]  # The importance weights (log probabilities)

# IMPORTANT: Use importance sampling with weights!
# Normalize weights (convert from log space and normalize)
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)

# Calculate weighted posterior probability
prob_blue_posterior = jnp.sum(is_blue_samples * normalized_weights)
print(f"P(Blue | says Blue) ≈ {prob_blue_posterior:.3f}")
```

**出力：**
```
P(Blue | says Blue) ≈ 0.414
```

**同じ答えです！** どちらの手法も機能します――`generate()` はより便利なだけです。

{{% notice style="warning" title="⚠️ 重要：必ず重みを使うこと！" %}}
条件付けと共に `generate()` を使う場合、返される重要度重みを**必ず**使用してください！

**なぜか？** GenJAX が観測に条件付けたトレースを生成する際、異なるトレースは異なる確率を持ちます。`weight` は各トレースがどれくらい尤もらしいかを教えてくれます。重みなしで単純に平均を取ると、**事後分布**（証拠を見た後の信念）ではなく**事前分布**（証拠を見る前の信念）が得られます。

**正しいアプローチ：**
<!-- validate: skip -->
```python
# Return BOTH value and weight
trace, weight = model.generate(key, observation, args)
# Then use weighted average with normalized weights
```

**間違ったアプローチ**（誤った答えを与える）：
<!-- validate: skip -->
```python
# Only returning value, ignoring weight
trace, weight = model.generate(key, observation, args)
return trace.get_retval()  # ❌ Discarding weight!
# Then simple average → gives prior, not posterior!
```

これが**重要度サンプリング**の本質です――確率的プログラミングにおける基本的な推論技術です。
{{% /notice %}}

{{% notice style="info" title="generate() vs simulate()" %}}
**`simulate(key, args)`：**
- すべての選択がランダムなトレースを生成
- 観測は指定しない
- トレースだけを返す

**`generate(key, observations, args)`：**
- 観測と一致するトレースを生成
- 指定された選択は与えられた値を取る
- 指定されていない選択はランダム
- `(trace, weight)` を返す（weight = 観測の対数確率）

**どちらを使うべきか：**
- **前向きシミュレーション**（観測なし）：`simulate()` を使う
- **条件付きサンプリング**（一部観測あり）：`generate()` を使う
{{% /notice %}}

---

## 理論的検証

ベイズの定理の厳密な計算に対してシミュレーションを検証しましょう：

```python
# Prior
P_blue = 0.15
P_green = 0.85

# Likelihood
P_says_blue_given_blue = 0.80
P_says_blue_given_green = 0.20

# Evidence (total probability of saying blue)
P_says_blue = (P_blue * P_says_blue_given_blue +
               P_green * P_says_blue_given_green)

# Posterior (Bayes' theorem)
P_blue_given_says_blue = (P_says_blue_given_blue * P_blue) / P_says_blue

print(f"=== Bayes' Theorem Calculation ===")
print(f"P(Blue) = {P_blue}")
print(f"P(says Blue | Blue) = {P_says_blue_given_blue}")
print(f"P(says Blue | Green) = {P_says_blue_given_green}")
print(f"P(says Blue) = {P_says_blue}")
print(f"\nP(Blue | says Blue) = {P_blue_given_says_blue:.3f}")
```

**出力：**
```
=== Bayes' Theorem Calculation ===
P(Blue) = 0.15
P(says Blue | Blue) = 0.8
P(says Blue | Green) = 0.2
P(says Blue) = 0.29

P(Blue | says Blue) = 0.414
```

**完全一致！** GenJAX シミュレーション ≈ 0.415、ベイズの定理の厳密値 = 0.414

---

## 事前分布と事後分布の可視化

証拠がどのように信念を変えるかを可視化しましょう：

```python
import matplotlib.pyplot as plt

# Prior: before observation
prior_blue = 0.15
prior_green = 0.85

# Posterior: after observation
posterior_blue = prob_blue_posterior  # From simulation
posterior_green = 1 - posterior_blue

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

categories = ['Green', 'Blue']
colors = ['#4ecdc4', '#6c5ce7']

# Prior
ax1.bar(categories, [prior_green, prior_blue], color=colors)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title('Prior: Before Chibany Speaks', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

for i, prob in enumerate([prior_green, prior_blue]):
    ax1.text(i, prob + 0.05, f'{prob:.1%}', ha='center', fontsize=14, fontweight='bold')

# Posterior
ax2.bar(categories, [posterior_green, posterior_blue], color=colors)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('Posterior: After Chibany Says "Blue"', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

for i, prob in enumerate([posterior_green, posterior_blue]):
    ax2.text(i, prob + 0.05, f'{prob:.1%}', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n📊 Belief Update:")
print(f"   Before: P(Blue) = {prior_blue:.1%}")
print(f"   After:  P(Blue | says Blue) = {posterior_blue:.1%}")
print(f"   Change: +{(posterior_blue - prior_blue):.1%}")
```

**重要な洞察：** 証拠によって青への信念は 15% から 41% に増加しましたが、ベースレートが非常に強いため**まだ 50% にも達しません**！

![事前分布と事後分布の確率分布](../../images/genjax/taxicab_prior_posterior.png)

---

## ベースレートの影響を調べる

ベースレートを変えると答えがどう変わるかを見てみましょう。

### シナリオ 1：タクシーが均等（青 50%、緑 50%）

<!-- validate: tol=0.02 -->
```python
import jax
import jax.numpy as jnp
from genjax import ChoiceMap

observation = ChoiceMap.d({"says_blue": True})

def run_equal_base(k):
    trace, weight = taxicab_model.generate(k, observation, (0.50, 0.80))
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 10000)
results_equal = jax.vmap(run_equal_base)(keys)
weights_equal = jnp.exp(results_equal[1] - jnp.max(results_equal[1]))
weights_equal = weights_equal / jnp.sum(weights_equal)
prob_equal = jnp.sum(results_equal[0] * weights_equal)

print(f"If 50% blue: P(Blue | says Blue) = {prob_equal:.3f}")
```

**出力：**
```
If 50% blue: P(Blue | says Blue) = 0.800
```

**80% になりました！** ベースレートが均等なとき、精度が支配的になります。

### シナリオ 2：ほとんどが青（青 85%、緑 15%）

<!-- validate: tol=0.03 -->
```python
import jax.numpy as jnp

def run_mostly_blue(k):
    trace, weight = taxicab_model.generate(k, observation, (0.85, 0.80))
    return trace.get_retval(), weight

results_mostly_blue = jax.vmap(run_mostly_blue)(keys)
weights_mostly_blue = jnp.exp(results_mostly_blue[1] - jnp.max(results_mostly_blue[1]))
weights_mostly_blue = weights_mostly_blue / jnp.sum(weights_mostly_blue)
prob_mostly_blue = jnp.sum(results_mostly_blue[0] * weights_mostly_blue)

print(f"If 85% blue: P(Blue | says Blue) = {prob_mostly_blue:.3f}")
```

**出力：**
```
If 85% blue: P(Blue | says Blue) = 0.971
```

**97% になりました！** ほとんどのタクシーが青いとき、「青」という発言は強い証拠になります。

### 効果の可視化

```python
import jax
import jax.numpy as jnp
from genjax import ChoiceMap

observation = ChoiceMap.d({"says_blue": True})

# Test different base rates
base_rates = jnp.linspace(0.01, 0.99, 50)
posteriors = []
key = jax.random.key(42)

for rate in base_rates:
    def run_with_rate(k):
        trace, weight = taxicab_model.generate(k, observation, (float(rate), 0.80))
        return trace.get_retval(), weight

    keys = jax.random.split(key, 1000)
    results = jax.vmap(run_with_rate)(keys)
    samples, weights = results[0], results[1]

    # Use importance sampling
    normalized_weights = jnp.exp(weights - jnp.max(weights))
    normalized_weights = normalized_weights / jnp.sum(normalized_weights)
    posteriors.append(jnp.sum(samples * normalized_weights))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(base_rates, posteriors, linewidth=2, color='#6c5ce7')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
plt.axvline(x=0.15, color='red', linestyle='--', alpha=0.7, label='Original problem (15%)')
plt.scatter([0.15], [0.414], color='red', s=100, zorder=5)

plt.xlabel('Base Rate: P(Blue)', fontsize=12)
plt.ylabel('Posterior: P(Blue | says Blue)', fontsize=12)
plt.title('How Base Rates Affect Inference\n(Chibany 80% accurate)', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

**グラフはシグモイド（S 字）曲線を示します：**
- **低いベースレート**（例：1%）：陽性の証人がいても事後確率は低いまま（約 4%）
- **中程度のベースレート**（例：50%）：急上昇――証拠の影響が最大（約 80%）
- **高いベースレート**（例：99%）：事後確率が確実性に近づく（約 99.7%）

このグラフが示すこと：**80% の精度でも、事後確率はベースレートに強く依存します！**

![ベースレートが推論に与える影響](../../images/genjax/taxicab_base_rate_effect.png)

{{% notice style="success" title="教訓" %}}
**ベースレートは実世界の推論において非常に重要です！**

医療診断、不正検出、証人証言――すべてにおいて以下を考慮する必要があります：
1. テスト・証人はどれほど正確か？（尤度）
2. その状態・犯罪はどれほど一般的か？（事前分布／ベースレート）

**ベースレートを無視すると間違った結論につながります。**

これが**ベースレート無視**――よくある認知バイアスです。
{{% /notice %}}

---

## 重要な概念

### 事前分布
**データを見る前の信念。**

- `simulate()` で生成（観測なし）
- 初期の不確かさを表す

### 事後分布
**データを見た後の信念。**

- `generate()` と観測で生成
- 証拠を取り込んだ更新後の信念を表す

### 実際のベイズの定理

GenJAX が自動的に数学を処理します：

$$P(\text{hypothesis} \mid \text{data}) = \frac{P(\text{data} \mid \text{hypothesis}) \cdot P(\text{hypothesis})}{P(\text{data})}$$

**あなたがすることは：**
1. 生成モデルを定義する（$P(\text{data} \mid \text{hypothesis})$ と $P(\text{hypothesis})$ をエンコード）
2. 観測を指定する（データ）
3. 条件付きトレースを生成する（GenJAX が事後分布を計算）

**手動でのベイズの定理計算は不要です！**

{{% notice style="success" title="生成モデルの力" %}}
生成関数を書くとき、あなたは次のことを指定しています：
- **事前分布：** 観測前の確率的選択の分布
- **尤度：** 観測が隠れた変数にどう依存するか
- **同時分布：** 完全な確率モデル

GenJAX が推論を自動的に処理します！
{{% /notice %}}

---

## 完全なサンプルコード

以下がすべてまとめたもので、簡単にコピーできます：

```python
import jax
import jax.numpy as jnp
from genjax import gen, flip, ChoiceMap
import matplotlib.pyplot as plt

@gen
def taxicab_model(base_rate_blue=0.15, accuracy=0.80):
    """Taxicab problem generative model."""
    is_blue = flip(base_rate_blue) @ "is_blue"

    # What Chibany says depends on true color
    says_blue_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    says_blue = flip(says_blue_prob) @ "says_blue"

    return is_blue

# Observation: Chibany says "blue"
observation = ChoiceMap.d({"says_blue": True})

# Generate posterior samples
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_inference(k):
    trace, weight = taxicab_model.generate(k, observation, (0.15, 0.80))
    return trace.get_retval(), weight

results = jax.vmap(run_inference)(keys)
is_blue_samples = results[0]
weights = results[1]

# Use importance sampling with weights
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
prob_blue = jnp.sum(is_blue_samples * normalized_weights)

print(f"=== TAXICAB INFERENCE ===")
print(f"Base rate: 15% blue")
print(f"Accuracy: 80%")
print(f"Observation: Says 'blue'")
print(f"\nP(Blue | says Blue) ≈ {prob_blue:.3f}")
```

---

## インタラクティブな探求

{{% notice style="tip" title="📓 インタラクティブノートブック：ベイズ学習" %}}
インタラクティブな例でベイズ学習を深く探求したいですか？**[ベイズ学習ノートブック（Colab で開く）](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/bayesian_learning.ipynb)**をチェックしてください。以下をカバーしています：

- 可視化付きの完全なタクシー問題
- 複数の観測によるシーケンシャルなベイズ更新
- 異なるベースレートと精度を探求するインタラクティブスライダー
- ベースレートが事後分布の信念に与える影響

このノートブックで今学んだ概念を実験できます！
{{% /notice %}}

---

## 演習

### 演習 1：高い精度

Chibany が 80% でなく 95% の精度を持っていたら？

**課題：** `accuracy=0.95` を使うようにコードを修正し、事後確率を計算してください。

{{% expand "解答" %}}
```python
import jax.numpy as jnp

def run_high_accuracy(k):
    trace, weight = taxicab_model.generate(k, observation, (0.15, 0.95))
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 10000)
results_high_acc = jax.vmap(run_high_accuracy)(keys)
weights_high_acc = jnp.exp(results_high_acc[1] - jnp.max(results_high_acc[1]))
weights_high_acc = weights_high_acc / jnp.sum(weights_high_acc)
prob_high_acc = jnp.sum(results_high_acc[0] * weights_high_acc)

print(f"With 95% accuracy: P(Blue | says Blue) = {prob_high_acc:.3f}")
```

**期待値：** ≈ 0.77 (77%)

**大幅に高い！** 精度は重要ですが、95% でもベースレートが 100% 未満に引き下げます。

**理論値：**
$$P = \frac{0.95 \times 0.15}{0.95 \times 0.15 + 0.05 \times 0.85} = \frac{0.1425}{0.1850} \approx 0.770$$
{{% /expand %}}

---

### 演習 2：反対の観測

Chibany が「青」ではなく「緑」と言ったら？

**課題：** $P(\text{青} \mid \text{緑と言った})$ を計算してください。

{{% expand "解答" %}}
```python
# Observation: says "green"
import jax.numpy as jnp

observation_green = ChoiceMap.d({"says_blue": False})

def run_says_green(k):
    trace, weight = taxicab_model.generate(k, observation_green, (0.15, 0.80))
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 10000)
results_green = jax.vmap(run_says_green)(keys)
weights_green = jnp.exp(results_green[1] - jnp.max(results_green[1]))
weights_green = weights_green / jnp.sum(weights_green)
prob_blue_given_green = jnp.sum(results_green[0] * weights_green)

print(f"P(Blue | says Green) = {prob_blue_given_green:.3f}")
```

**期待値：** ≈ 0.041 (4.1%)

**非常に低い！** Chibany（80% 精度）が「緑」と言うなら、非常に高い確率で緑です。

**理論値：**
$$P = \frac{0.20 \times 0.15}{0.20 \times 0.15 + 0.80 \times 0.85} = \frac{0.03}{0.71} \approx 0.042$$
{{% /expand %}}

---

### 演習 3：2 人の証人

**独立した 2 人の証人**が両方とも「青」と言ったら？

**課題：** 2 人の証人（どちらも 80% の精度）を含むようにモデルを拡張し、事後確率を計算してください。

{{% expand "解答" %}}
```python
import jax.numpy as jnp

@gen
def taxicab_two_witnesses(base_rate_blue=0.15, accuracy=0.80):
    """Two independent witnesses."""
    is_blue = flip(base_rate_blue) @ "is_blue"

    # Witness 1
    witness1_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    witness1 = flip(witness1_prob) @ "witness1"

    # Witness 2 (independent)
    witness2_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    witness2 = flip(witness2_prob) @ "witness2"

    return is_blue

# Both say "blue"
observation_two = ChoiceMap.d({"witness1": True, "witness2": True})

def run_two_witnesses(k):
    trace, weight = taxicab_two_witnesses.generate(k, observation_two, (0.15, 0.80))
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 10000)
results_two = jax.vmap(run_two_witnesses)(keys)
weights_two = jnp.exp(results_two[1] - jnp.max(results_two[1]))
weights_two = weights_two / jnp.sum(weights_two)
prob_two = jnp.sum(results_two[0] * weights_two)

print(f"P(Blue | both say Blue) = {prob_two:.3f}")
```

**期待値：** ≈ 0.73 (73%)

**大幅に高い！** 独立した 2 つの証拠ははるかに強力です。

**理論値：**
$$P(\text{両方が青と言った} \mid \text{青}) = 0.80^2 = 0.64$$
$$P(\text{両方が青と言った} \mid \text{緑}) = 0.20^2 = 0.04$$
$$P = \frac{0.64 \times 0.15}{0.64 \times 0.15 + 0.04 \times 0.85} = \frac{0.096}{0.130} \approx 0.738$$

**2 人の証人によって、低いベースレートにもかかわらず 50% を超えました！**
{{% /expand %}}

---

## 学んだこと

この章では以下を学びました：

✅ **条件付き確率** ―― 観測への制限

✅ **フィルタリングアプローチ** ―― 推論のための棄却サンプリング

✅ **`generate()` 関数** ―― チョイスマップによる条件付け

✅ **事前分布と事後分布** ―― データの前後の信念

✅ **実際のベイズの定理** ―― 自動的なベイズ更新

✅ **ベースレートの影響** ―― 事前分布が非常に重要な理由

✅ **実際の推論問題** ―― タクシーシナリオ

**重要な洞察：** 確率的プログラミングにより、手動でベイズの定理を計算することなく、**仮定をエンコード**（生成モデル）し**質問をする**（条件付け）ことができます！

---

## なぜこれが重要か

**実世界のアプリケーション：**

1. **医療診断：** 検査精度 + 疾患の有病率 → 疾患の確率
2. **不正検出：** 取引パターン + 不正のベースレート → 不正の確率
3. **スパムフィルタリング：** メールの特徴 + スパムのベースレート → スパムの確率
4. **刑事司法：** 証人の精度 + 犯罪のベースレート → 有罪の確率

**すべて同じパターンに従います：**
- 生成モデルを定義する（データがどのように生じるか）
- データを観測する
- 隠れた原因を推論する

**GenJAX はこれを体系的かつスケーラブルにします。**

---

## 次のステップ

これであなたは以下を知っています：
- 生成モデルの構築方法
- 観測を使った推論の実行方法
- 事後確率の解釈方法
- なぜベースレートが重要か

**次は：** 第 6 章でゼロからモデルを構築する方法を学びます！

---

|[← 前：トレースを理解する](./03_traces.md) | [次：自分のモデルを構築する →](./06_building_models.md)|
| :--- | ---: |
