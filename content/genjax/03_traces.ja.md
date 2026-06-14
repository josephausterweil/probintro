+++
date = "2026-06-14"
title = "トレースを理解する"
weight = 4
+++

## コードが実行されるとき何が記録されるか？

通常の Python 関数を実行すると、処理を行って値を返します。それで終わり — 内部で何が起きたかの記録は残りません。

**GenJAX は異なります。** 生成的関数を実行すると、GenJAX は **トレース** を生成します。これは以下の完全な記録です：
1. どのようなランダムな選択が行われたか
2. それらがどのような値を取ったか
3. 関数が何を返したか
4. この実行がどのくらい確率的だったか

まるで実験のあらゆる詳細を自動的に記録する実験ノートのようなものです！

![Chibany investigating](images/chibanylayingdown.png)

---

## トレースが重要な理由

**端的に言えば：** トレースは **推論** を可能にします — 「もしこれを観測したらどうなるか？」という問いに答えること。

**シナリオ例：**
- `chibany_day()` を実行すると `(0, 1)` が返ってきた — 昼食はハンバーガー、夕食はとんかつ
- トレースには「昼食に 0、夕食に 1 を選んだ」と記録される
- 後で次のように問うことができる：「夕食がとんかつだったとき、昼食もとんかつだった確率は？」

**トレースにより、観測から原因への逆方向の推論が可能になります！**

これについては第 4 章で詳しく探ります。まず、トレースに何が含まれるかを理解しましょう。

---

## トレースの構造

生成的関数を思い出してください：

```python
import jax
from genjax import gen, flip

@gen
def chibany_day():
    lunch_is_tonkatsu = flip(0.5) @ "lunch"
    dinner_is_tonkatsu = flip(0.5) @ "dinner"
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)
```

実行すると：

```python
key = jax.random.key(42)
trace = chibany_day.simulate(key, ())
```

GenJAX は **3 つの主要要素** を含むトレースオブジェクトを生成します：

### 1. 戻り値

**関数が返した値：**

```python
meals = trace.get_retval()
print(meals)  # Output: (0, 1)
```

これが最終結果 — 観測可能な結果です。

### 2. ランダムな選択

**行われた全てのランダムな決定と、その名前：**

```python
choices = trace.get_choices()
print(choices)
# Output: {'lunch': 0, 'dinner': 1}
```

これが **選択マップ** — アドレス（名前）を値にマッピングする辞書です。

{{% notice style="info" title="名前が重要な理由" %}}
`flip(0.5) @ "lunch"` において、`@ "lunch"` の部分がこのランダムな選択に **名前**（またはアドレス）を与えます。

GenJAX はこれらの名前を以下の目的に使います：
- どの選択がどれかを追跡する
- 観測値を指定できるようにする（詳しくは第 4 章！）
- 推論アルゴリズムを可能にする

**化学実験室で試験管にラベルを貼るようなものだと考えてください。** どれがどれかを知る必要があります！
{{% /notice %}}

### 3. 対数確率（スコア）

**この実行はどのくらい確率的だったか？**

```python
score = trace.get_score()
print(score)  # Output: -1.3862943611198906
```

これはこの特定の実行の **対数確率** です。

{{% notice style="note" title="数学表記：対数確率" %}}
この例では：
- 昼食 = 0 の確率は 0.5
- 夕食 = 1 の確率は 0.5
- 同時確率：$P(\text{lunch}=0, \text{dinner}=1) = 0.5 \times 0.5 = 0.25$

対数確率：$\log(0.25) = -1.386...$

**なぜ対数を使うのか？**
- 数値アンダーフロー（非常に小さい確率）を防ぐ
- 乗算を加算に変換する（計算が簡単！）
- 確率的プログラミングの標準

**対数確率を直接扱う必要はありません** — GenJAX がこれを処理してくれます。「この結果がどのくらいあり得るか」を測定するものだと知っていれば十分です。
{{% /notice %}}

{{% notice style="success" title="📐→💻 数学からコードへの対応" %}}
**トレースと確率論の接続：**

| 数学的概念 | 数学的表記 | GenJAX トレース要素 |
|--------------|----------------------|------------------------|
| **結果** | $\omega \in \Omega$ | 1 つのトレース（1 回の実行） |
| **結果空間** | $\Omega = \\{HH, HT, TH, TT\\}$ | 全ての可能なトレース |
| **確率変数** | $X(\omega)$ | 選択マップ中の選択 |
| **確率** | $P(\omega)$ | `jnp.exp(trace.get_score())` |
| **対数確率** | $\log P(\omega)$ | `trace.get_score()` |
| **同時分布** | $P(X_1, X_2)$ | トレース上の分布 |

**重要な洞察：**
- **トレースは結果そのもの** — ランダムなプロセスが展開する 1 つの完全な方法を表す
- **選択マップ = 確率変数** — `"lunch"` や `"dinner"` のような名前付きランダム選択
- **get_retval() = 観測可能な結果** — 直接観測できるもの
- **get_score() = 対数確率** — この特定のトレースがどのくらいあり得るか
- **複数のトレース = 複数の結果** — `simulate()` を繰り返し実行すると Ω からサンプリングされる

**対応の例：**
```
Math: ω = HT  (outcome from Ω)
Code: trace with choices = {'lunch': 0, 'dinner': 1}
They're the same thing, just different representations!
```
{{% /notice %}}

---

## トレース全体の図

トレースに何が含まれるかを可視化しましょう：

```
┌─────────────────────────────────────────┐
│           TRACE OBJECT                  │
├─────────────────────────────────────────┤
│                                         │
│  1. Arguments: ()                       │
│     (what was passed to the function)   │
│                                         │
│  2. Random Choices (Choice Map):        │
│     {'lunch': 0, 'dinner': 1}           │
│     (all random decisions made)         │
│                                         │
│  3. Return Value:                       │
│     (0, 1)                              │
│     (what the function returned)        │
│                                         │
│  4. Log Probability (Score):            │
│     -1.386                              │
│     (how probable was this trace)       │
│                                         │
└─────────────────────────────────────────┘
```

**`simulate()` を呼び出すたびに、（潜在的に）異なるランダムな選択を持つ新しいトレースが得られます。**

---

## トレース要素へのアクセス

トレース情報にアクセスする 3 つの方法を示す完全な例です：

```python
import jax
from genjax import gen, flip

@gen
def chibany_day():
    lunch_is_tonkatsu = flip(0.5) @ "lunch"
    dinner_is_tonkatsu = flip(0.5) @ "dinner"
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)

# Generate one trace
key = jax.random.key(42)
trace = chibany_day.simulate(key, ())

# Access different parts
print("=== TRACE CONTENTS ===")
print(f"Return value: {trace.get_retval()}")
print(f"Random choices: {trace.get_choices()}")
print(f"Log probability: {trace.get_score()}")

# Decode to outcome notation
outcome_map = {(0, 0): "HH", (0, 1): "HT", (1, 0): "TH", (1, 1): "TT"}
retval = trace.get_retval()
outcome = outcome_map[(int(retval[0]), int(retval[1]))]
print(f"Outcome: {outcome}")
```

**出力（例）：**
```
=== TRACE CONTENTS ===
Return value: (0, 1)
Random choices: {'lunch': 0, 'dinner': 1}
Log probability: -1.3862943611198906
Outcome: HT
```

{{% notice style="tip" title="実際に表示されるもの" %}}
このコードを実行すると、「Random choices」の出力はより複雑に見えます：
```
Random choices: Static({'lunch': Choice(v=<jax.Array(False, dtype=bool)>), 'dinner': Choice(v=<jax.Array(False, dtype=bool)>)})
```

**心配しないでください！** これは GenJAX の内部表現です。重要な部分は：
- `'lunch': Choice(v=<jax.Array(False, ...)>)` は昼食 = 0（False = ハンバーガー）を意味する
- `'dinner': Choice(v=<jax.Array(False, ...)>)` は夕食 = 0（False = ハンバーガー）を意味する

<details>
<summary><em>なぜ違いがあるのか？（クリックして展開）</em></summary>

GenJAX はランダムな選択に関するメタデータを追跡するために値を `Choice` オブジェクトでラップします。`choices['lunch']` で個々の選択にアクセスすると、実際の値が得られます。

上に示した簡略化された出力（`{'lunch': 0, 'dinner': 1}`）は、技術的な実装の詳細ではなく、選択が実際に何であるかという **論理的内容** を表しています。

</details>
{{% /notice %}}

---

## 複数のトレース、複数の履歴

各トレースは生成的関数の **1 回の可能な実行** を表します。

5 回実行すると、5 つの異なるトレースが得られます：

```python
key = jax.random.key(42)

for i in range(5):
    # Split key for each run (JAX requirement)
    key, subkey = jax.random.split(key)

    trace = chibany_day.simulate(subkey, ())
    retval = trace.get_retval()
    outcome = outcome_map[(int(retval[0]), int(retval[1]))]
    choices = trace.get_choices()

    print(f"Day {i+1}: {outcome} — lunch={choices['lunch']}, dinner={choices['dinner']}")
```

**出力（例）：**
```
Day 1: HT — lunch=0, dinner=1
Day 2: TH — lunch=1, dinner=0
Day 3: HH — lunch=0, dinner=0
Day 4: TT — lunch=1, dinner=1
Day 5: HT — lunch=0, dinner=1
```

各トレースは **異なる履歴** — ランダムなプロセスが展開し得た異なる方法です。

{{% notice style="tip" title="JAX ランダムキー" %}}
各実行に新しいキーを生成するために `jax.random.split(key)` を使っていることに気づきましたか？

**なぜか？** JAX は再現性のために明示的なランダムキーを使います。同じキーは常に同じ結果を与えます。

**パターン：**
<!-- validate: skip -->
```python
key, subkey = jax.random.split(key)  # Create new key
trace = model.simulate(subkey, ...)   # Use the subkey
```

これにより、再現性を保ちながら毎回異なるランダムな結果が保証されます。
{{% /notice %}}

---

## トレースと戻り値

**重要な区別：**

| `simulate()` の戻り値 | `get_retval()` の戻り値 |
|---------------------|----------------------|
| **トレースオブジェクト** | **実際の値** |
| 選択、スコア、戻り値を含む | 戻り値のみ |
| 推論に使用 | 結果に使用 |

**例：**

```python
# This is a trace object
trace = chibany_day.simulate(key, ())

# This is the return value (a tuple)
meals = trace.get_retval()

# These are different!
print(type(trace))   # <class 'genjax.generative_functions.static.trace.StaticTrace'>
print(type(meals))   # <class 'tuple'>
```

**どちらをいつ使うか：**
- **結果だけが必要な場合？** `trace.get_retval()` を使う
- **ランダムな選択を検査したい場合？** `trace.get_choices()` を使う
- **推論を行う場合？** トレースオブジェクト全体を使う

---

## 確率論との接続

トレースを集合ベースの確率に結びつけましょう：

| 確率の概念 | トレースの等価物 |
|---------------------|------------------|
| 結果 $\omega \in \Omega$ | 1 つのトレース（1 回の実行） |
| 結果空間 $\Omega$ | 全ての可能なトレース |
| $P(\omega)$ | `exp(trace.get_score())` |
| 確率変数 $X(\omega)$ | 選択マップ中の選択 |
| 同時分布 | トレース上の分布 |

**重要な洞察：** トレースは結果そのものです！トレースはランダムなプロセスが展開し得る 1 つの完全な方法を表します。

**例：**
- **集合ベース：** $\omega = HT$（$\Omega = \{HH, HT, TH, TT\}$ からの 1 つの結果）
- **トレースベース：** `choices = {'lunch': 0, 'dinner': 1}` を持つトレース

**これらは同じものです！** ただし異なる表現です。

---

## 推論においてこれが重要な理由

次の問いを考えてみましょう：

> **「Chibany が夕食にとんかつを食べたとき、昼食にもとんかつを食べた確率は？」**

**集合ベースのアプローチ：**
1. 事象 $D$ = 「夕食がとんかつ」= $\{HT, TT\}$ を定義する
2. 事象 $L$ = 「昼食がとんかつ」= $\{TH, TT\}$ を定義する
3. $P(L \mid D) = \frac{|L \cap D|}{|D|} = \frac{1}{2}$ を計算する

**トレースベースのアプローチ：**
1. 多くのトレースを生成する
2. `choices['dinner'] == 1` のトレースをフィルタリングする
3. その中で `choices['lunch'] == 1` の数を数える
4. 比率を計算する

**トレース構造がこのフィルタリングを可能にします！** GenJAX が全てのランダムな選択を記録しているため、内部を見て何が起きたかを確認できます。

これは第 4 章で実装します！

---

## 実践例：トレースの検査

10 個のトレースを生成して検査しましょう：

```python
import jax
import jax.numpy as jnp
from genjax import gen, flip

@gen
def chibany_day():
    lunch_is_tonkatsu = flip(0.5) @ "lunch"
    dinner_is_tonkatsu = flip(0.5) @ "dinner"
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)

# Generate 10 traces
key = jax.random.key(42)
outcome_map = {(0, 0): "HH", (0, 1): "HT", (1, 0): "TH", (1, 1): "TT"}

print("Day | Outcome | Lunch | Dinner | Log Prob")
print("----|---------|-------|--------|----------")

for i in range(10):
    key, subkey = jax.random.split(key)
    trace = chibany_day.simulate(subkey, ())

    retval = trace.get_retval()
    outcome = outcome_map[(int(retval[0]), int(retval[1]))]
    choices = trace.get_choices()
    score = trace.get_score()

    print(f" {i+1:2d} |   {outcome}    |   {choices['lunch']}   |   {choices['dinner']}    | {score:.2f}")
```

**出力（例）：**
```
Day | Outcome | Lunch | Dinner | Log Prob
----|---------|-------|--------|----------
  1 |   HT    |   0   |   1    | -1.39
  2 |   TH    |   1   |   0    | -1.39
  3 |   HH    |   0   |   0    | -1.39
  4 |   TT    |   1   |   1    | -1.39
  5 |   HT    |   0   |   1    | -1.39
  6 |   HH    |   0   |   0    | -1.39
  7 |   TT    |   1   |   1    | -1.39
  8 |   HT    |   0   |   1    | -1.39
  9 |   TH    |   1   |   0    | -1.39
 10 |   HH    |   0   |   0    | -1.39
```

**注意：** 全ての結果が等しく確率的であるため、全ての対数確率は同じです（-1.39 ≈ log(0.25)）！

---

## 練習問題

### 練習問題 1：トレースの探索

このコードを実行して質問に答えてください：

```python
key = jax.random.key(123)
trace = chibany_day.simulate(key, ())

print(f"Return value: {trace.get_retval()}")
print(f"Choices: {trace.get_choices()}")
print(f"Score: {trace.get_score()}")
```

**質問：**
1. どの結果が得られましたか？（HH、HT、TH、または TT）
2. 選択マップには何がありますか？
3. 対数確率は前の例と同じですか？

{{% expand "解答" %}}
**回答：**
1. 結果はランダムシード（123）によって異なる
2. 選択マップには `{'lunch': 0 or 1, 'dinner': 0 or 1}` が含まれる
3. はい！全ての結果は等しい確率（0.25）を持つため、対数確率は常に -1.386... になる

**重要な洞察：** 異なるランダムキー → 異なるトレース、しかし同じ確率（この対称的な例の場合）
{{% /expand %}}

---

### 練習問題 2：不均等な確率

`chibany_day` を不均等な確率に変更してください：

```python
@gen
def chibany_day_biased():
    lunch_is_tonkatsu = flip(0.8) @ "lunch"  # 80% Tonkatsu
    dinner_is_tonkatsu = flip(0.2) @ "dinner"  # 20% Tonkatsu
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)
```

5 つのトレースを生成して対数確率を比較してください。

**質問：** 全ての対数確率は同じですか？なぜそうなのか、またはなぜそうでないのか？

{{% expand "解答" %}}
```python
key = jax.random.key(42)

for i in range(5):
    key, subkey = jax.random.split(key)
    trace = chibany_day_biased.simulate(subkey, ())

    retval = trace.get_retval()
    outcome = outcome_map[(int(retval[0]), int(retval[1]))]
    score = trace.get_score()

    print(f"Day {i+1}: {outcome} — Log prob: {score:.3f}")
```

**回答：** いいえ！結果が異なる確率を持つため、対数確率は異なります：
- TT: $P = 0.8 \times 0.2 = 0.16$, $\log(0.16) = -1.83$
- TH: $P = 0.8 \times 0.8 = 0.64$, $\log(0.64) = -0.45$
- HT: $P = 0.2 \times 0.2 = 0.04$, $\log(0.04) = -3.22$
- HH: $P = 0.2 \times 0.8 = 0.16$, $\log(0.16) = -1.83$

**TH が最も確率が高い**（最も高い確率 = 最も負でない対数確率）！
{{% /expand %}}

---

### 練習問題 3：条件付きカウント

`chibany_day()` から 1000 個のトレースを生成して次に答えてください：

**「夕食がとんかつである日の中で、昼食もとんかつである割合は？」**

**ヒント：** `choices['dinner'] == 1` のトレースをフィルタリングして、`choices['lunch'] == 1` の数を数えてください。

{{% expand "解答" %}}
```python
import jax
import jax.numpy as jnp

key = jax.random.key(42)
keys = jax.random.split(key, 1000)

# Generate all traces
def run_one_day(k):
    trace = chibany_day.simulate(k, ())
    return trace.get_retval()

days = jax.vmap(run_one_day)(keys)

# get_retval() returns a tuple (lunch, dinner), so vmap gives us a
# tuple of two arrays — unpack them rather than indexing columns.
lunch, dinner = days

# Filter: dinner is Tonkatsu (dinner == 1)
dinner_is_tonkatsu = dinner == 1

# Among those, count lunch is Tonkatsu
both_tonkatsu = (lunch == 1) & (dinner == 1)

# Calculate conditional probability
n_dinner_tonkatsu = jnp.sum(dinner_is_tonkatsu)
n_both = jnp.sum(both_tonkatsu)

prob_lunch_given_dinner = n_both / n_dinner_tonkatsu

print(f"Days with dinner = Tonkatsu: {n_dinner_tonkatsu}")
print(f"Days with both = Tonkatsu: {n_both}")
print(f"P(lunch=T | dinner=T) ≈ {prob_lunch_given_dinner:.3f}")
```

**出力：**
```
Days with dinner = Tonkatsu: 481
Days with both = Tonkatsu: 243
P(lunch=T | dinner=T) ≈ 0.505
```

**期待される結果：** ≈ 0.5（50%）

**なぜか？** 昼食と夕食は独立性を持ちます！夕食を知っても昼食の確率は変わりません。

**これはフィルタリングによる条件付き確率です！**（詳しくは第 4 章）
{{% /expand %}}

---

## 学んだこと

この章では以下を学びました：

✅ **トレースとは何か** — ランダムな実行の完全な記録

✅ **3 つの主要要素** — 戻り値、選択マップ、対数確率

✅ **名前が重要な理由** — `@ "address"` により追跡と推論が可能になる

✅ **トレース要素へのアクセス方法** — `get_retval()`、`get_choices()`、`get_score()`

✅ **結果としてのトレース** — 確率論との接続

✅ **推論の予告** — 条件付き問いに答えるためのトレースのフィルタリング

**重要な洞察：** トレースは単なる記録ではありません — 生成的コードと確率的推論の橋渡しです！

---

## 次のステップ

トレースを理解したので、GenJAX の最も強力な機能に進む準備ができました：

**第 4 章：条件付けと観測** — 「もしこれを観測したらどうなるか？」という問いに答え、証拠に基づいて信念を更新する方法！

これは GenJAX が通常のシミュレーションと比べて真に輝く部分です。

---

|[← 前の章：最初の GenJAX モデル](./02_first_model.md) | [次の章：条件付けと観測 →](./04_conditioning.md)|
| :--- | ---: |
