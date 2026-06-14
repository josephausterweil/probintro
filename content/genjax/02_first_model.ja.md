+++
date = "2026-06-14"
title = "はじめてのGenJAXモデル"
weight = 3
+++

## 集合からシミュレーションへ

Chibanyの日々の食事を覚えていますか？ 結果空間 $\Omega = \\{HH, HT, TH, TT\\}$ を列挙して可能性を数えました。

今度は、コンピュータにそれらの結果を**生成**させてみましょう！

![Chibany laying down](images/chibanylayingdown.png)

---

## 生成的プロセス

毎日：
1. **昼食が届く** — ランダムにH（ハンバーガー）またはT（とんかつ）（等確率）
2. **夕食が届く** — ランダムにH（ハンバーガー）またはT（とんかつ）（等確率）
3. **その日を記録する** — 食事のペア

GenJAXでは、これを**生成関数**として表現します。

---

## はじめての生成関数

GenJAXによるChibanyの食事は次のとおりです：

```python
import jax
from genjax import gen, flip

@gen
def chibany_day():
    """Generate one day of Chibany's meals."""

    # Lunch: flip a coin (0=Hamburger, 1=Tonkatsu)
    lunch_is_tonkatsu = flip(0.5) @ "lunch"

    # Dinner: flip another coin
    dinner_is_tonkatsu = flip(0.5) @ "dinner"

    # Return the pair
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)
```

{{% notice style="warning" title="重要：flip()を使うこと（bernoulli()ではなく）" %}}
GenJAXにはベルヌーイ分布のための関数が二つあります：`flip(p)` と `bernoulli(p)` です。**常に `flip(p)` を使ってください** — 直感どおりに正しく動作します！

`bernoulli(p)` 関数はパラメータとしてコインフリップのロジット値を期待するため、私たちの目的には直感的ではありません。`flip(p)` 関数は期待どおりに動作し、GenJAXの公式サンプルでも使われています。

**バグの例：**
- `bernoulli(0.9)` は約71%を返す（90%ではない） ❌
- `flip(0.9)` は期待どおり約90%を返す ✅

このチュートリアルでは正しい動作を保証するために、全体を通じて `flip()` を使います。
{{% /notice %}}

{{% notice style="success" title="📐→💻 数学からコードへの対応" %}}
**数学的概念のGenJAXへの対応：**

| 数学の概念 | 数学的記法 | GenJAXのコード |
|--------------|----------------------|-------------|
| **結果空間** | $\Omega = \\{HH, HT, TH, TT\\}$ | `@gen def chibany_day(): ...` |
| **確率変数** | $X \sim \text{Bernoulli}(0.5)$ | `flip(0.5) @ "lunch"` |
| **確率** | $P(A) = \frac{\|A\|}{\|\Omega\|}$ | `jnp.mean(condition_satisfied)` |
| **事象** | $A = \\{HT, TH, TT\\}$ | `has_tonkatsu = (days[:, 0] == 1) \| (days[:, 1] == 1)` |

**重要なポイント：**
- **@gen関数** = Ωを定義する生成的プロセス
- **flip(p)** = 確率pを持つ確率変数（ベルヌーイ分布）
- **@ "name"** = 確率的選択にラベルを付ける（後の推論のため）
- **シミュレーション＋カウント** = 確率を計算すること
{{% /notice %}}

### 内訳の解説

**1行目：`@gen`**
- GenJAXに「これは生成関数である」と伝える
- GenJAXはすべての確率的選択を追跡する

**2〜3行目：関数定義**
- `def chibany_day():` で関数を定義する
- ドキュメント文字列がその機能を説明する

**6行目：最初の確率的選択**
```python
lunch_is_tonkatsu = flip(0.5) @ "lunch"
```
- `flip(0.5)` — 公平なコインを投げる（1が出る確率50%、0が出る確率50%）
- `@ "lunch"` — この確率的選択に「lunch」という**名前**を付ける
- 結果を `lunch_is_tonkatsu` に格納する

**9行目：二番目の確率的選択**
```python
dinner_is_tonkatsu = flip(0.5) @ "dinner"
```
- もう一度コインを投げる。「dinner」という名前を付ける

**12行目：戻り値**
<!-- validate: skip -->
```python
return (lunch_is_tonkatsu, dinner_is_tonkatsu)
```
- 二つの値の**タプル**（ペア）を返す
- これは $\Omega$ からの一つの結果に相当します！

---

## 関数を実行する

### 1日分を生成する

```python
# Create a random key (JAX requirement for randomness)
key = jax.random.key(42)

# Generate one day
trace = chibany_day.simulate(key, ())

# What happened?
meals = trace.get_retval()
print(f"Today's meals: {meals}")
```

**出力（例）：**
```
Today's meals: (0, 1)
```

{{% notice style="warning" title="実際に表示される内容" %}}
このコードを実行すると、次のような出力が表示されます：
```
Today's meals: (Array(0, dtype=int32), Array(1, dtype=int32))
```

**慌てないでください！** GenJAXはPythonの通常の数値ではなくJAX配列を返すためです。

<details>
<summary><em>なぜ違うのか？（クリックして展開）</em></summary>

**表示されるもの**：`Array(0, dtype=int32)` または `Array(1, dtype=int32)`

**その意味**：
- `Array(0, dtype=int32)` = 0 = ハンバーガー
- `Array(1, dtype=int32)` = 1 = とんかつ

**JAXがこうする理由**：JAXはGPU上で高速計算を可能にするためにすべてに配列を使います。これはCPUとGPUの両方で効率的に実行できる数値を表すJAXの方法です。

**シンプルな数値を得るには**、次のように変換できます：
```python
meals_simple = (int(meals[0]), int(meals[1]))
print(f"Today's meals: {meals_simple}")
# Output: (0, 1)
```

**このチュートリアルでは**：`Array(0, dtype=int32)` は単に0を表す別の表現であり、`Array(1, dtype=int32)` は1を意味することを覚えておいてください。

</details>
{{% /notice %}}

これは、昼食がハンバーガー（0）、夕食がとんかつ（1）を意味します — つまり私たちの記法では $HT$ です！

{{% notice style="info" title="「key」とは何ですか？" %}}
JAXは乱数性を制御するために**ランダムキー**を使います。これはシードのようなものです — 同じキーは常に同じ「ランダムな」結果を生成するため、再現性の確保に役立ちます。

**詳細は気にしないでください！** 知っておくことは次の2点だけです：
- `jax.random.key(some_number)` でキーを作成する
- `jax.random.split(key, n)` で複数回使用するために分割する
{{% /notice %}}

### 確率的選択にアクセスする

```python
# Get all the random choices made
choices = trace.get_choices()

print(f"Lunch was tonkatsu: {choices['lunch']}")
print(f"Dinner was tonkatsu: {choices['dinner']}")
```

**出力（上記のトレースに対して）：**
```
Lunch was tonkatsu: 0
Dinner was tonkatsu: 1
```

{{% notice style="tip" title="期待される出力" %}}
実際には次のように表示されます：
```
Lunch was tonkatsu: 0
Dinner was tonkatsu: 1
```

**良いニュース**：`choices['lunch']` のように個別の選択にアクセスする場合、GenJAXはラップされた `Array(...)` 形式ではなく、通常の数値（0または1）を返します。これにより扱いやすくなります。
{{% /notice %}}

---

## 多くの日をシミュレートする

今度は10,000日分を生成しましょう！

```python
# Generate 10,000 random keys
import jax.numpy as jnp

keys = jax.random.split(key, 10000)

# Run the generative function for each key
def run_one_day(k):
    trace = chibany_day.simulate(k, ())
    return trace.get_retval()

# Use JAX's vmap for parallel execution
days_tuples = jax.vmap(run_one_day)(keys)

# Convert from tuples to a 2D array for easier analysis
# Each row is one day, columns are [lunch, dinner]
days = jnp.stack(days_tuples, axis=1)
```

{{% notice style="tip" title="vmapとは何ですか？" %}}
`vmap` は「ベクトル化されたmap」を意味します — 関数を並列で何度も実行するため、**非常に高速**です！

「これを10,000回実行するが、一度に全部行う（一つずつではなく）」というイメージです。
{{% /notice %}}

### 結果を数える

これで10,000日分のデータがあります。少なくとも一食がとんかつの日を数えてみましょう：

```python
import jax.numpy as jnp

# Check if either meal is tonkatsu (1)
has_tonkatsu = jnp.logical_or(days[:, 0], days[:, 1])

# Count how many days have tonkatsu
count_with_tonkatsu = jnp.sum(has_tonkatsu)

# Calculate probability
prob_tonkatsu = jnp.mean(has_tonkatsu)

print(f"Days with tonkatsu: {count_with_tonkatsu} out of 10000")
print(f"P(at least one tonkatsu) ≈ {prob_tonkatsu:.3f}")
```

**出力（例）：**
```
Days with tonkatsu: 7489 out of 10000
P(at least one tonkatsu) ≈ 0.749
```

**確率チュートリアルから：** 正確な答えは $3/4 = 0.75$ です！

10,000回のシミュレーションで非常に近い値が得られました：$0.749 \approx 0.75$

{{% notice style="info" title="📘 基礎概念：シミュレーション対カウント" %}}
**チュートリアル1、第3章を思い出してください**。確率はカウントです：

$$P(A) = \frac{|A|}{|\Omega|} = \frac{\text{events in event}}{\text{total outcomes}}$$

私たちは手計算で $P(\text{少なくとも一食がとんかつ}) = \frac{|\{HT, TH, TT\}|}{|\{HH, HT, TH, TT\}|} = \frac{3}{4} = 0.75$ を計算しました。

**GenJAXでは、列挙の代わりにシミュレートします：**

| チュートリアル1（手計算） | チュートリアル2（GenJAX） |
|---------------------|---------------------|
| すべての結果を**列挙**：{HH, HT, TH, TT} | 10,000サンプルを**生成** |
| 好ましい結果を**カウント**：4つ中3つ | 好ましい結果を**カウント**：10,000のうち約7,500 |
| **除算**：3/4 = 0.75 | **除算**：7,500/10,000 ≈ 0.75 |

**なぜシミュレートするのか？**
- チュートリアル1のアプローチは複雑なモデルでは機能しなくなる（列挙するには結果が多すぎる）
- シミュレーションはスケールする：Ωが4つの結果であろうと40億であろうと同じコードが機能する
- シミュレーション数が増えるにつれて（10K → 100K → 1M）、正確な答えに近づく

**原理は同一です** — 好ましい結果を数えて合計で割るだけです。しかし、シミュレーションによって手では列挙不可能なモデルも扱えるようになります！

[← チュートリアル1、第3章でカウントとしての確率を復習する](../../intro/03_prob_count/)
{{% /notice %}}

---

## 結果を可視化する

四つの結果すべてを示す棒グラフを作成しましょう：

```python
import matplotlib.pyplot as plt

# Count each outcome
import jax.numpy as jnp

HH = jnp.sum((days[:, 0] == 0) & (days[:, 1] == 0))
HT = jnp.sum((days[:, 0] == 0) & (days[:, 1] == 1))
TH = jnp.sum((days[:, 0] == 1) & (days[:, 1] == 0))
TT = jnp.sum((days[:, 0] == 1) & (days[:, 1] == 1))

# Create bar chart
outcomes = ['HH', 'HT', 'TH', 'TT']
counts = [HH, HT, TH, TT]

plt.figure(figsize=(8, 5))
plt.bar(outcomes, counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7dc6f'])
plt.xlabel('Outcome')
plt.ylabel('Count (out of 10,000)')
plt.title("Chibany's Meals: 10,000 Simulated Days")
plt.axhline(y=2500, color='gray', linestyle='--', label='Expected (2500 each)')
plt.legend()
plt.show()
```

**表示される内容：** 四つの棒がほぼ同じ高さ（それぞれ約2500）で表示され、各結果の理論的期待値 $1/4$ に一致します！

![10,000回のシミュレーションによる結果の分布](../../images/genjax/first_model_outcome_distribution.png)

---

## インタラクティブな探索

{{% notice style="tip" title="📓 インタラクティブノートブック" %}}
ライブスライダーと可視化機能を備えた**[インタラクティブノートブック - Colabで開く](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/02_first_genjax_model.ipynb)**をお試しください！この章のすべてのコードに加え、パラメータを変えて結果がどう変わるかを探索できるインタラクティブウィジェットが含まれています。
{{% /notice %}}

付属のノートブックには**インタラクティブウィジェット**があり、次のことができます：

### スライダー1：昼食にとんかつが出る確率
- スライダーを0.0から1.0まで動かす
- 分布がどのように変わるかを確認する！

### スライダー2：夕食にとんかつが出る確率
- 夕食を昼食から独立させる
- あるいは特定の食事でとんかつが出やすく/出にくくする

### スライダー3：シミュレーション回数
- 100、1,000、10,000、あるいは100,000回のシミュレーションを試す
- シミュレーション回数が増えるにつれて推定値がより正確になることを確認する

**スライダーを動かすとグラフが自動的に更新されます！**

{{% notice style="success" title="試してみよう！" %}}
Colabノートブックで：
1. 昼食の確率を0.8（80%がとんかつ）に設定する
2. 夕食の確率を0.2（20%がとんかつ）に設定する
3. 10,000回のシミュレーションを実行する
4. 分布について何に気づきますか？

**答え：** 昼食がとんかつの結果（TH、TT）が、そうでない結果（HH、HT）よりもはるかに多くなります！
{{% /notice %}}

---

## 集合ベースの確率との関連

これを学んだことに結びつけてみましょう：

| 集合ベースの概念 | GenJAXの対応 |
|-------------------|-------------------|
| 結果空間 $\Omega$ | `simulate()` を何度も実行すること |
| 一つの結果 $\omega$ | `simulate()` を一度呼び出すこと |
| 事象 $A \subseteq \Omega$ | シミュレーションのフィルタリング |
| $\|A\|$（要素を数える） | `jnp.sum(condition)` |
| $P(A) = \|A\|/\|\Omega\|$ | `jnp.mean(condition)` |

**例：**

**集合ベース：**
- 事象：「少なくとも一食がとんかつ」= $\\{HT, TH, TT\\}$
- 確率：$|\\{HT, TH, TT\\}| / |\\{HH, HT, TH, TT\\}| = 3/4$

**GenJAX：**
```python
import jax.numpy as jnp

has_tonkatsu = (days[:, 0] == 1) | (days[:, 1] == 1)
prob = jnp.mean(has_tonkatsu)  # ≈ 0.75
```

**同じ概念です！** ただし、手でカウントする代わりにコンピュータで計算します。

---

## トレースを理解する

`chibany_day.simulate(key, ())` を実行すると、GenJAXは次の内容を記録した**トレース**を作成します：

1. **引数** — 提供された入力（この場合はなし）
2. **確率的選択** — 行われたすべての確率的決定とその名前
3. **戻り値** — 最終的な結果

```python
trace = chibany_day.simulate(key, ())

# Access different parts
print(f"Return value: {trace.get_retval()}")
print(f"Choices: {trace.get_choices()}")
print(f"Log probability: {trace.get_score()}")
```

**出力（例）：**
```
Return value: (Array(False, dtype=bool), Array(False, dtype=bool))
Choices: {'lunch': Array(False, dtype=bool), 'dinner': Array(False, dtype=bool)}
Log probability: -1.3862943611198906
```

**これが意味すること：**
- **戻り値**：食事のペア（両方がFalse = 両方がハンバーガー = HH）
- **選択**：すべての名前付き確率的選択とその値の辞書
- **対数確率**：この特定の結果の対数尤度（$\log(0.5 \times 0.5) = \log(0.25) \approx -1.386$）

{{% notice style="info" title="なぜすべてを追跡するのか？" %}}
すべての確率的選択を追跡することは**推論**に不可欠です — 「これを観測したとき、何が確からしいか？」と問いたい場合です。

これは第4章で実際に見てみましょう！
{{% /notice %}}

---

## 練習問題

Colabノートブックでこれらを試してみてください：

### 練習問題1：異なる確率

次のようにコードを変更してください：
- 昼食がとんかつである確率を70%にする
- 夕食がとんかつである確率を30%にする

**ヒント：** `flip(0.5)` の値を変えてみましょう！

{{% expand "解答" %}}
```python
@gen
def chibany_day_weighted():
    lunch_is_tonkatsu = flip(0.7) @ "lunch"
    dinner_is_tonkatsu = flip(0.3) @ "dinner"
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)
```
{{% /expand %}}

### 練習問題2：とんかつのカウント

シミュレートされたすべての日を通じてChibanyが**何食のとんかつを食べるか**を数えるコードを書いてください（とんかつがある日を特定するだけでなく、合計数を求める）。

**ヒント：** `days[:, 0] + days[:, 1]` を合計してみましょう

{{% expand "解答" %}}
```python
import jax.numpy as jnp

total_tonkatsu = jnp.sum(days[:, 0]) + jnp.sum(days[:, 1])
avg_per_day = total_tonkatsu / len(days)

print(f"Total tonkatsu: {total_tonkatsu}")
print(f"Average per day: {avg_per_day:.2f}")
```

**等しい確率（それぞれ0.5）の場合、1日平均約1.0食のとんかつになるはずです！**
{{% /expand %}}

### 練習問題3：三食に拡張？

モデルを朝食も含めるように拡張してください！これでChibanyは1日3食になります。

{{% expand "解答" %}}
```python
@gen
def chibany_three_meals():
    breakfast_is_tonkatsu = flip(0.5) @ "breakfast"
    lunch_is_tonkatsu = flip(0.5) @ "lunch"
    dinner_is_tonkatsu = flip(0.5) @ "dinner"
    return (breakfast_is_tonkatsu, lunch_is_tonkatsu, dinner_is_tonkatsu)
```

**これで結果空間には $2^3 = 8$ 通りの可能な結果があります！**
{{% /expand %}}

---

## 学んだこと

この章では：

✅ はじめての生成関数を書いた

✅ 何千もの確率的結果をシミュレートした

✅ カウントによって確率を計算した

✅ 分布を可視化した

✅ 集合とシミュレーションの関係を理解した

✅ トレースと確率的選択について学んだ

**重要な洞察：** 生成関数を使うことで、コンピュータは集合を使って手でやっていたことを実行できます — しかも今や何百万もの可能性を扱えます！

---

## 次のステップ

結果を生成できるようになったので、次の疑問は：

**もし何かを観測したら？どのように信念を更新するか？**

それが**推論**であり、GenJAXが真の力を発揮する領域です！

---

|[← 前：Pythonの基礎](./01_python_basics.md) | [次：トレースを理解する →](./03_traces.md)|
| :--- | ---: |
