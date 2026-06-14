+++
date = "2026-06-14"
title = "自分でモデルを作る"
weight = 6
+++

## レシピを使うことから、自分で作ることへ

GenJAX を例題を通して学んできました。いよいよ **自分の** 確率モデルを構築する時間です！

この章では、生成モデルの作り方——現実の問題をコードに変換する方法——の **考え方** を学びます。

![作る準備ができた Chibany](images/chibanyplain.png)

---

## モデル構築のプロセス

### ステップ 1: 問題を理解する

**コードを書く前に、次の問いに答えてください：**

1. **何を予測・理解しようとしているか？** （問い）
2. **何を観測しているか？** （データ／証拠）
3. **何が隠れているか？** （未知の変数）
4. **それらはどのように関係しているか？** （因果構造）

**例：** スパムメール検出
1. **問い：** このメールはスパムか？
2. **観測：** メールの内容、送信者、時刻
3. **隠れた変数：** 真のスパム状態
4. **関係：** スパムメールには特定の単語パターンがある

---

### ステップ 2: 生成ストーリーをスケッチする

**データを生成するプロセスを書き出します：**

「最初に、自然界は…を選ぶ。次にそれに基づいて…を生成し、…を生み出す。」

**例：** コイン投げ
1. まず、コインには（隠れた）バイアスパラメータがある
2. そのバイアスに基づいて、各投げは表か裏になる
3. 投げた結果の系列を観測する

**この物語がそのままコードになります！**

---

### ステップ 3: 分布を選ぶ

**各確率的な選択について、分布を選んでください：**

| 変数の種類 | よく使う分布 |
|------------------|---------------------|
| 二値（はい/いいえ） | `flip(p)` |
| カテゴリカル（A/B/C） | `categorical(probs)` |
| カウント（0, 1, 2, ...） | `poisson(rate)` |
| 連続値 | `normal(mean, std)`, `uniform(low, high)` |

**シンプルに始めましょう！** ほとんどの二値の選択には `flip` を使ってください。

---

### ステップ 4: コードを書く

**パターン：**

<!-- validate: skip -->
```python
@gen
def my_model(parameters):
    # Hidden variables (causes)
    hidden = distribution(...) @ "hidden"

    # Observed variables (effects)
    # Usually depend on hidden variables
    if hidden:
        observed = distribution_A(...) @ "observed"
    else:
        observed = distribution_B(...) @ "observed"

    return hidden  # Or whatever you want to predict
```

**重要なポイント：**
- `@gen` デコレータを使う
- すべての確率的な選択を `@ "名前"` で命名する
- 推論したいものを返す
- `if` 文を使って依存関係をモデル化する

---

### ステップ 5: テストと検証

1. **サンプルを生成する** — 出力は妥当に見えるか？
2. **極端なケースを確認する** — パラメータが 0 や 1 の場合は？
3. **推論を検証する** — 事後分布の結果は直感的に納得できるか？

{{% notice style="success" title="📐→💻 数学からコードへの変換" %}}
**モデル構築の概念が GenJAX でどのように変換されるか：**

| 数学的概念 | 数学的表記 | GenJAX パターン |
|--------------|----------------------|----------------|
| **同時分布** | $P(X, Y)$ | @gen 関数内の複数の `flip()` 呼び出し |
| **条件付き分布** | $P(Y \mid X)$ | `if X: Y = flip(p1)` |
| **独立性** | $P(X, Y) = P(X) \cdot P(Y)$ | 独立した確率的選択（if 文なし） |
| **依存性** | $P(Y \mid X) \neq P(Y)$ | Y の分布が X を if 文で使用 |
| **階層モデル** | $\theta \sim \text{Prior}, X \mid \theta$ | パラメータを確率変数として：`theta = uniform() @ "theta"` |
| **混合モデル** | $\sum_k P(Z=k) P(X \mid Z=k)$ | `if category == k: X = distribution_k()` |
| **系列モデル** | $P(X_t \mid X_{t-1})$ | 前の状態への依存を持つループ |

**よくあるモデリングパターン：**

| パターン | 確率的構造 | コード構造 |
|---------|---------------------|----------------|
| **独立した観測** | $P(X_1, \ldots, X_n) = \prod P(X_i)$ | `for i: X_i = flip()` |
| **階層的** | $P(\theta) P(X \mid \theta)$ | `theta = uniform(); X = flip(theta)` |
| **条件付き** | $P(Y \mid X)$ が X に依存 | `if X: Y = flip(p1) else: Y = flip(p2)` |
| **時系列** | $P(X_t \mid X_{t-1})$ | `for t: X[t] = flip(f(X[t-1]))` |
| **混合** | $\sum_k \pi_k P(X \mid k)$ | `k = categorical(pi); if k==0: ... else: ...` |

**重要な洞察：**
- **@gen 関数 = 同時分布** — P（全変数）を定義する
- **if 文 = 条件付き依存** — Y が X に依存する
- **for ループ = 繰り返し構造** — 複数の観測や時間ステップ
- **確率変数としてのパラメータ = 階層的** — 複数の階層での不確実性
- **生成ストーリー = 数学** — データがどのように生成されるかを説明できれば、コードに変換できる

**例：医療診断**
```
Math: P(Disease, Fever, Cough) = P(Disease) × P(Fever|Disease) × P(Cough|Disease)
Code: has_disease = flip(0.01) @ "disease"
      fever_prob = jnp.where(has_disease, 0.9, 0.1)
      cough_prob = jnp.where(has_disease, 0.8, 0.2)
      fever = flip(fever_prob) @ "fever"
      cough = flip(cough_prob) @ "cough"
```
{{% /notice %}}

---

## よくあるパターン

### パターン 1: 独立した観測

**シナリオ：** 複数の独立した測定

**例：** コイン投げ

```python
import jax
import jax.numpy as jnp
from genjax import gen, flip, uniform, categorical, ChoiceMap

# The number of flips is a Python constant captured by the @gen, NOT a model
# argument. JAX traces model arguments into abstract values, so a Python
# `for i in range(n)` loop can't use a count that arrives as a traced argument
# (it raises TracerIntegerConversionError). Fixing the count as a module-level
# constant keeps the teaching loop readable and runnable.
N_FLIPS = 10

@gen
def coin_flips(bias=0.5):
    """Generate N_FLIPS independent coin flips."""

    results = []
    for i in range(N_FLIPS):
        # Each flip is independent
        result = flip(bias) @ f"flip_{i}"
        results.append(result)

    return jnp.array(results).astype(int)
```

{{% notice style="note" title="なぜカウントは引数ではなく定数なのか？" %}}
JAX はモデルの引数を実行前に *抽象的な* 値にトレースするため、`for i in range(n)` のような Python ループは、モデル引数として渡された `n` を使用できません——`TracerIntegerConversionError` が発生します。解決策は、カウントを `@gen` がクローズオーバーする Python 定数にすること（ここでは `N_FLIPS`）か、カウントを受け取って `@gen` を返すファクトリ関数でモデルをラップすることです（チュートリアル 3 の DPMM 章が使うパターン）。どちらの方法でもループ自体は変わりません。
{{% /notice %}}

**使い方：**

<!-- validate: skip-output -->
```python
key = jax.random.key(42)
trace = coin_flips.simulate(key, (0.7,))
flips = trace.get_retval()
print(f"Flips: {flips}")
```

**出力（例）：**
```
Flips: [0 1 1 0 1 1 1 1 1 1]
```

---

### パターン 2: 階層的構造

**シナリオ：** パラメータ自体が分布を持つ

**例：** 投げ結果からコインのバイアスを学習する

```python
@gen
def coin_with_unknown_bias():
    """Coin with unknown bias — infer it from N_FLIPS flips."""

    # Hidden: the coin's true bias (uniform between 0 and 1)
    bias = uniform(0.0, 1.0) @ "bias"

    # Observations: flip outcomes (N_FLIPS is the module constant from above)
    flips = []
    for i in range(N_FLIPS):
        result = flip(bias) @ f"flip_{i}"
        flips.append(result)

    return bias  # Want to infer this!
```

**推論：**

<!-- validate: tol=0.1 -->
```python
# Observe 7 heads out of 10 flips
observations = ChoiceMap.d({
    "flip_0": 1, "flip_1": 1, "flip_2": 0,
    "flip_3": 1, "flip_4": 1, "flip_5": 0,
    "flip_6": 1, "flip_7": 1, "flip_8": 0,
    "flip_9": 1
})

# Infer bias
key = jax.random.key(42)
keys = jax.random.split(key, 1000)

def infer_bias(k):
    # generate(key, CONSTRAINTS, ARGS) — the model takes no args, so ()
    trace, weight = coin_with_unknown_bias.generate(k, observations, ())
    return trace.get_retval(), weight

results = jax.vmap(infer_bias)(keys)
posterior_bias = results[0]
weights = results[1]

# Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
mean_bias = jnp.sum(posterior_bias * normalized_weights)

print(f"Estimated bias: {mean_bias:.2f}")
# Should be around 0.70 (7 heads / 10 flips)
```

**出力：**
```
Estimated bias: 0.66
```

---

### パターン 3: 条件付き依存

**シナリオ：** 観測が隠れた状態に依存する

**例：** 天気が気分に影響する

```python
import jax.numpy as jnp

@gen
def mood_model():
    """Weather affects Chibany's mood."""

    # Hidden: today's weather
    is_sunny = flip(0.7) @ "is_sunny"  # 70% sunny days

    # Observable: Chibany's mood depends on weather
    # Sunny → happy 90% of the time, Rainy → happy only 30% of the time
    happy_prob = jnp.where(is_sunny, 0.9, 0.3)
    is_happy = flip(happy_prob) @ "is_happy"

    return is_sunny
```

**問い：** 「Chibany は幸せそうです。晴れている確率はどのくらいですか？」

<!-- validate: tol=0.02 -->
```python
observation = ChoiceMap.d({"is_happy": 1})

def infer_weather(k):
    # generate(key, CONSTRAINTS, ARGS)
    trace, weight = mood_model.generate(k, observation, ())
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 10000)
results = jax.vmap(infer_weather)(keys)
posterior_sunny = results[0]
weights = results[1]

# Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
prob_sunny = jnp.sum(posterior_sunny * normalized_weights)

print(f"P(Sunny | Happy) ≈ {prob_sunny:.3f}")
```

**出力：**
```
P(Sunny | Happy) ≈ 0.873
```

{{% expand "理論的な答え" %}}
ベイズの定理を使うと：

$$P(\text{Sunny} \mid \text{Happy}) = \frac{P(\text{Happy} \mid \text{Sunny}) \cdot P(\text{Sunny})}{P(\text{Happy})}$$

- $P(\text{Sunny}) = 0.7$
- $P(\text{Happy} \mid \text{Sunny}) = 0.9$
- $P(\text{Happy} \mid \text{Rainy}) = 0.3$
- $P(\text{Happy}) = 0.7 \times 0.9 + 0.3 \times 0.3 = 0.63 + 0.09 = 0.72$

$$P = \frac{0.9 \times 0.7}{0.72} = \frac{0.63}{0.72} \approx 0.875$$

**期待値：** ≈ 87.5%
{{% /expand %}}

---

### パターン 4: 系列と時系列

**シナリオ：** 出来事が時間とともに展開する

**例：** Chibany の一週間の食事

<!-- validate: skip-output -->
```python
# Like N_FLIPS above, the number of days is a Python constant, not a model arg.
DAYS = 7

@gen
def weekly_meals():
    """Model a week of meals with memory."""

    meals = []

    # First day is random
    prev_meal = flip(0.5) @ "day_0"
    meals.append(prev_meal)

    # Each subsequent day depends on the previous day. prev_meal is a traced
    # value, so we pick the probability with jnp.where (not a Python if):
    #   tonkatsu yesterday (1) → want variety → 0.3; hamburger (0) → craving → 0.8
    for day in range(1, DAYS):
        current_prob = jnp.where(prev_meal == 1, 0.3, 0.8)
        current_meal = flip(current_prob) @ f"day_{day}"
        meals.append(current_meal)
        prev_meal = current_meal

    return jnp.array(meals).astype(int)

# Simulate one week
meals = weekly_meals.simulate(jax.random.key(0), ()).get_retval()
print(f"Week of meals (1=tonkatsu, 0=hamburger): {meals}")
```

**出力（例）：**
```
Week of meals (1=tonkatsu, 0=hamburger): [1 0 1 1 0 0 1]
```

**これが時間を通じた依存性をモデル化しています！**

---

### パターン 5: 混合モデル

**シナリオ：** データが複数のソースから来るが、どのソースかは観測されない

**例：** 二種類の日（平日 vs 週末）。Chibany は何曜日か知りません。週末のお弁当にはとんかつが入る可能性がずっと高いです。

```python
import jax.numpy as jnp

@gen
def mixed_days():
    """Different behavior on weekends vs weekdays."""

    # Hidden: is it a weekend?
    is_weekend = flip(2/7) @ "is_weekend"  # 2 out of 7 days

    # Weekend: high chance of tonkatsu (relaxed), Weekday: lower chance (busy)
    tonkatsu_prob = jnp.where(is_weekend, 0.9, 0.3)
    lunch = flip(tonkatsu_prob) @ "lunch"

    return is_weekend
```

**推論：** 「Chibany がとんかつを食べたとします。週末である確率は？」

---

## 完全なモデルを構築する：医療診断

ゼロから現実的な例を構築しましょう。

**シナリオ：** 症状に基づく疾患の診断

**設定：**
- 疾患の有病率：1%（稀）
- 症状 1（発熱）：罹患時 90%、健常時 10%
- 症状 2（咳）：罹患時 80%、健常時 20%

**問い：** 患者が発熱と咳を持っています。疾患の確率は？

### ステップ 1: 問題を理解する

- **問い：** 患者は疾患を持っているか？
- **観測：** 発熱と咳
- **隠れた変数：** 真の疾患状態
- **関係：** 罹患時に症状が出やすい

### ステップ 2: 生成ストーリー

1. まず、患者は疾患を持っている（1%）か持っていない（99%）
2. 罹患している場合、発熱は非常に起こりやすい（90%）
3. 罹患している場合、咳は非常に起こりやすい（80%）
4. 健常の場合、どちらの症状も稀（10%、20%）

### ステップ 3: モデルを書く

```python
import jax.numpy as jnp

@gen
def disease_model(prevalence=0.01, fever_if_disease=0.9, cough_if_disease=0.8,
                  fever_if_healthy=0.1, cough_if_healthy=0.2):
    """Medical diagnosis model."""

    # Hidden: disease status
    has_disease = flip(prevalence) @ "has_disease"

    # Symptoms depend on disease status
    fever_prob = jnp.where(has_disease, fever_if_disease, fever_if_healthy)
    cough_prob = jnp.where(has_disease, cough_if_disease, cough_if_healthy)
    fever = flip(fever_prob) @ "fever"
    cough = flip(cough_prob) @ "cough"

    return has_disease
```

### ステップ 4: 推論を実行する

<!-- validate: tol=0.05 -->
```python
# Patient has both symptoms
observation = ChoiceMap.d({"fever": 1, "cough": 1})

def infer_disease(k):
    # generate(key, CONSTRAINTS, ARGS)
    trace, weight = disease_model.generate(k, observation, ())
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 10000)
results = jax.vmap(infer_disease)(keys)
posterior = results[0]
weights = results[1]

# Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
prob_disease = jnp.sum(posterior * normalized_weights)

print(f"=== MEDICAL DIAGNOSIS ===")
print(f"Prevalence: 1%")
print(f"Symptoms: Fever + Cough")
print(f"P(Disease | Symptoms) ≈ {prob_disease:.3f}")
```

**出力：**
```
=== MEDICAL DIAGNOSIS ===
Prevalence: 1%
Symptoms: Fever + Cough
P(Disease | Symptoms) ≈ 0.269
```

**期待値：** ≈ 0.265（26.5%）

**解釈：** 両方の症状があっても、疾患が非常に稀なため、疾患の確率はわずか 26.5% です！

{{% notice style="warning" title="医療における基準率の無視！" %}}
**これが医療検査で偽陽性が問題になる理由です。**

稀な疾患では、精度の高い検査でも多くの偽陽性が発生します。なぜなら：
- 真陽性：$0.01 \times 0.9 \times 0.8 = 0.0072$（0.72%）
- 偽陽性：$0.99 \times 0.1 \times 0.2 = 0.0198$（1.98%）

**真陽性より偽陽性の方が多い！**

これが医師が症状だけで診断しない理由です——確認検査が必要か、患者の病歴を考慮する（事前分布を更新する）必要があります。
{{% /notice %}}

---

## ベストプラクティス

### ✅ やること

#### 1. すべてを明確に命名する

<!-- validate: skip -->
```python
# Good
is_diseased = flip(0.01) @ "is_diseased"

# Bad
x = flip(0.01) @ "x"
```

#### 2. 意味のあるパラメータを使う

<!-- validate: skip -->
```python
# Good
@gen
def model(disease_prevalence=0.01, test_accuracy=0.95):
    ...

# Bad
@gen
def model(p1=0.01, p2=0.95):
    ...
```

#### 3. モデルをドキュメント化する

<!-- validate: skip -->
```python
@gen
def weather_mood(sunny_prior=0.7):
    """Model how weather affects mood.

    Args:
        sunny_prior: Base rate of sunny days (default 0.7)

    Returns:
        is_sunny: Whether it's sunny today
    """
```

#### 4. シンプルに始めて、複雑さを追加する

- まず最もシンプルなモデルを構築する
- 動作を確認する
- 機能を段階的に追加する

#### 5. エッジケースをテストする

- パラメータが 0 や 1 の場合は？
- すべての観測が同じ場合は？
- 事後分布は直感的に納得できるか？

---

### ❌ やってはいけないこと

#### 1. 確率的な選択に名前を付けるのを忘れない

<!-- validate: skip -->
```python
# Bad — can't condition on this!
x = flip(0.5)

# Good
x = flip(0.5) @ "x"
```

#### 2. 同じ名前を 2 回使わない

<!-- validate: skip -->
```python
# Bad — name collision!
flip1 = flip(0.5) @ "flip"
flip2 = flip(0.5) @ "flip"  # ERROR!

# Good — unique names
flip1 = flip(0.5) @ "flip_1"
flip2 = flip(0.5) @ "flip_2"
```

#### 3. 分布を考えすぎない

- `flip` はほとんどの二値のケースをカバーする
- `normal` は連続値に
- `categorical` は複数の選択肢に
- 始めるのに特殊な分布は必要ありません！

#### 4. 検証をスキップしない

- 常に最初にサンプルを生成する
- 出力が合理的かどうかを確認する
- 極端なパラメータ値を検証する

---

## 演習

### 演習 1: メールスパムフィルター

シンプルなスパムフィルターモデルを構築してください。

**シナリオ：**
- メールの 30% はスパムです
- スパムメールには「FREE」が 80% の確率で含まれます
- 正規のメールには「FREE」が 10% の確率で含まれます

**タスク：** $P(\text{Spam} \mid \text{contains "FREE"})$ を計算してください

{{% expand "解答" %}}
<!-- validate: tol=0.05 -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, flip, ChoiceMap

@gen
def spam_filter(spam_rate=0.30):
    """Simple spam filter based on keyword."""

    # Hidden: is it spam?
    is_spam = flip(spam_rate) @ "is_spam"

    # Observation: contains "FREE"?
    contains_free_prob = jnp.where(is_spam, 0.80, 0.10)
    contains_free = flip(contains_free_prob) @ "contains_free"

    return is_spam

# Email contains "FREE"
observation = ChoiceMap.d({"contains_free": 1})

def infer_spam(k):
    # generate(key, CONSTRAINTS, ARGS)
    trace, weight = spam_filter.generate(k, observation, ())
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 10000)
results = jax.vmap(infer_spam)(keys)
posterior = results[0]
weights = results[1]

# Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
prob_spam = jnp.sum(posterior * normalized_weights)

print(f"P(Spam | contains 'FREE') ≈ {prob_spam:.3f}")
```

**出力：**
```
P(Spam | contains 'FREE') ≈ 0.777
```

**期待値：** ≈ 0.774（77.4%）

**理論値：**
$$P = \frac{0.80 \times 0.30}{0.80 \times 0.30 + 0.10 \times 0.70} = \frac{0.24}{0.31} \approx 0.774$$
{{% /expand %}}

---

### 演習 2: 複数の観測から学習する

コイン投げモデルを拡張して、複数の観測からバイアスを推論してください。

**タスク：** 20 回の投げの系列（例：`[1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1]`）が与えられた場合、コインのバイアスを推論してください。

{{% expand "解答" %}}
<!-- validate: tol=0.1 -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, flip, uniform, ChoiceMap

# Fixed count as a module constant (see the note at the top of the chapter).
N_OBSERVED = 20

@gen
def coin_model():
    """Infer coin bias from N_OBSERVED observed flips."""

    # Hidden: coin's true bias
    bias = uniform(0.0, 1.0) @ "bias"

    # Observations: flips
    for i in range(N_OBSERVED):
        result = flip(bias) @ f"flip_{i}"

    return bias

# Observed flips: 16 heads out of 20
observed_flips = [1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1]
observations = ChoiceMap.d({f"flip_{i}": observed_flips[i] for i in range(N_OBSERVED)})

def infer_bias(k):
    # generate(key, CONSTRAINTS, ARGS)
    trace, weight = coin_model.generate(k, observations, ())
    return trace.get_retval(), weight

key = jax.random.key(42)
keys = jax.random.split(key, 1000)
results = jax.vmap(infer_bias)(keys)
posterior_bias = results[0]
weights = results[1]

# Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
mean_bias = jnp.sum(posterior_bias * normalized_weights)
# For standard deviation with weighted samples
variance = jnp.sum(normalized_weights * (posterior_bias - mean_bias)**2)
std_bias = jnp.sqrt(variance)

print(f"Estimated bias: {mean_bias:.2f} ± {std_bias:.2f}")
# Should be around 0.80 (16/20)
```

**出力：**
```
Estimated bias: 0.77 ± 0.09
```

**期待値：** 平均 ≈ 0.80、不確実性あり

**事後分布をプロットする：**
```python
import matplotlib.pyplot as plt

plt.hist(posterior_bias, bins=50, density=True, alpha=0.7, color='#4ecdc4')
plt.axvline(mean_bias, color='red', linestyle='--', label=f'Mean = {mean_bias:.2f}')
plt.xlabel('Coin Bias')
plt.ylabel('Posterior Density')
plt.title('Posterior Distribution of Coin Bias\n(16 heads in 20 flips)')
plt.legend()
plt.show()
```
{{% /expand %}}

---

### 演習 3: 複数症状の診断

疾患モデルを拡張して、3 つの症状（発熱、咳、倦怠感）を含めてください。

**パラメータ：**
- 疾患：有病率 2%
- 罹患時：発熱 90%、咳 80%、倦怠感 95%
- 健常時：発熱 10%、咳 20%、倦怠感 30%

**タスク：** 次の場合の事後分布を計算してください：
1. 発熱のみ
2. 発熱 + 咳
3. 3 つすべての症状

{{% expand "解答" %}}
<!-- validate: tol=0.05 -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, flip, ChoiceMap

@gen
def disease_three_symptoms(prevalence=0.02):
    """Disease model with three symptoms."""

    has_disease = flip(prevalence) @ "has_disease"

    # Symptoms depend on disease status
    fever_prob = jnp.where(has_disease, 0.90, 0.10)
    cough_prob = jnp.where(has_disease, 0.80, 0.20)
    fatigue_prob = jnp.where(has_disease, 0.95, 0.30)
    fever = flip(fever_prob) @ "fever"
    cough = flip(cough_prob) @ "cough"
    fatigue = flip(fatigue_prob) @ "fatigue"

    return has_disease

key = jax.random.key(42)

# Scenario 1: Fever only
obs1 = ChoiceMap.d({"fever": 1})

# Scenario 2: Fever + cough
obs2 = ChoiceMap.d({"fever": 1, "cough": 1})

# Scenario 3: All three
obs3 = ChoiceMap.d({"fever": 1, "cough": 1, "fatigue": 1})

for i, obs in enumerate([obs1, obs2, obs3], 1):
    def infer(k, obs=obs):
        # generate(key, CONSTRAINTS, ARGS); obs bound per-iteration
        trace, weight = disease_three_symptoms.generate(k, obs, ())
        return trace.get_retval(), weight

    keys = jax.random.split(key, 10000)
    results = jax.vmap(infer)(keys)
    posterior = results[0]
    weights = results[1]

    # Use importance sampling
    normalized_weights = jnp.exp(weights - jnp.max(weights))
    normalized_weights = normalized_weights / jnp.sum(normalized_weights)
    prob = jnp.sum(posterior * normalized_weights)

    print(f"Scenario {i}: P(Disease) ≈ {prob:.3f}")
```

**出力：**
```
Scenario 1: P(Disease) ≈ 0.164
Scenario 2: P(Disease) ≈ 0.439
Scenario 3: P(Disease) ≈ 0.713
```

**洞察：** 証拠が増えるほど事後確率が高くなります！（発熱のみ → 発熱 + 咳 → 3 つすべての症状）
{{% /expand %}}

---

## 学んだこと

この章では次のことを学びました：

✅ **モデル構築のプロセス** — 問題からコードへ
✅ **よくあるパターン** — 独立的、階層的、条件付き、系列的、混合
✅ **ベストプラクティス** — 命名、ドキュメント化、テスト
✅ **完全な例** — 医療診断、スパムフィルタリング、コイン投げ
✅ **生成的に考える方法** — 「何がデータを生成するのか？」

**重要な洞察：** モデルを構築することは、世界がどのように機能するかについての **仮定をコードに埋め込む** ことであり、そのあとは GenJAX が推論を行います！

---

## 次のステップ

### 構築する準備ができました！

これで次のことができるためのすべてのツールが揃っています：
- 自分の問題のための生成モデルを構築する
- ベイズ推論を自動的に実行する
- 予測の不確実性を理解する

**これからどこへ行くか：**

### 1. より多くの分布を探索する

GenJAX は `flip` 以外にも多くの分布をサポートしています：

- `normal(mean, std)` — 連続値（身長、体重、気温）
- `categorical(probs)` — 複数の離散的な選択（A, B, C, D）
- `poisson(rate)` — カウントデータ（イベントの数）
- `gamma`, `beta`, `exponential` — 特殊な連続分布

**完全なリファレンスは GenJAX のドキュメントをご覧ください。**

### 2. 高度な推論を学ぶ

このチュートリアルでカバーしたもの：
- フィルタリング／棄却サンプリング
- `generate()` による条件付け

**次のレベル：**
- 重点サンプリング（稀なイベントに対してより効率的）
- マルコフ連鎖モンテカルロ（MCMC）（複雑なモデルに）
- 変分推論（近似的だが高速）

**参照：** GenJAX 上級チュートリアル

### 3. 現実世界への応用

学んだことをこれらに応用してください：
- **科学：** 実験のモデル化、データ分析
- **医療：** 診断、治療の最適化
- **工学：** 故障検出、品質管理
- **社会科学：** 人間の行動の理解
- **AI/ML：** 不確実性を持つより良いモデルの構築

---

## 旅路

**最初は：** 集合、カウント、基本的な確率

**今では：** 確率的プログラムを構築し、ベイズ推論を実行し、不確実性の下で推論できます

**それは大きな成果です！**

---

## 最後に

確率的プログラミングは **スーパーパワー** です：

1. **不確実性を表現する** — 世界は不確実であり、モデルはそれを反映すべきです
2. **推論を自動化する** — コンピュータが難しい数学を担当します
3. **知識とデータを組み合わせる** — ドメイン専門知識（事前分布）と観測（データ）の両方を使用します
4. **より良い意思決定をする** — リスクと確率を理解します

**構築し続けて、学び続けて、問い続けましょう！**

---

## 章が完了しました！

一から自分の確率モデルを構築する方法を学びました。これが GenJAX プログラミングチュートリアルの最後の章です。

**このチュートリアルで達成したこと：**
- GenJAX 環境をセットアップした
- 確率的プログラミングのための Python の基礎を学んだ
- `@gen` デコレータを使った生成モデルを構築した
- トレースと GenJAX が実行を記録する方法を理解した
- 観測に基づいてモデルを条件付けした
- 確率的な問いに答えるための推論を実行した
- 現実世界の問題のための完全なモデルを作成した

**次のステップへの準備ができました！**

---

## 次は：連続確率とベイズ学習

これまで、**離散的な** 確率変数（コイン投げ、カテゴリ、はい/いいえの結果）を扱ってきました。しかし、多くの現実世界の量は **連続的** です——身長、気温、待ち時間。

**チュートリアル 3：連続確率とベイズ学習** では：

- 連続分布（正規分布、指数分布など）を扱います
- 連続パラメータを使ったベイズ更新を学びます
- クラスタリングのための混合モデルを構築します
- 無限混合のためのディリクレ過程を探索します

**ここで学んだ確率的プログラミングのスキルは直接応用できます！**

[**チュートリアル 3：連続確率へ進む →**](/probintro/intro2/)

---

|[← 前へ：行動する推論](./05_inference.md) | [チュートリアル 3：連続確率 →](/probintro/intro2/)|
| :--- | ---: |
