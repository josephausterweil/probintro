+++
date = "2026-06-26"
title = "用語集 - 全チュートリアル"
weight = 100
+++

## この用語集の使い方

この用語集は、GenJAXによる確率論シリーズの3つのチュートリアル全体をカバーしています。各用語には、どのチュートリアルで導入されるかを示すタグが付いています：

- 📘 **チュートリアル1**（離散確率）- 集合と数え上げのアプローチ
- 💻 **チュートリアル2**（GenJAXプログラミング）- 確率的プログラミングの基礎
- 📊 **チュートリアル3**（連続確率）- 発展的トピックとベイズ学習

各用語をクリックすると、例とコードを含む定義が展開されます。

---

## 基本概念（チュートリアル1）

### ベイズの定理 📘
{{% expand "ベイズの定理" %}}
*ベイズの定理*（またはベイズの規則）は、変数が条件付けられる順序を逆転させるための公式です。すなわち、$P(A \mid B)$ から $P(B \mid A)$ を求める方法です。

**公式:** $P(H \mid D) = \frac{P(D \mid H) P(H)}{P(D)}$

**構成要素:**
- $P(H \mid D)$ = 事後分布（データを見た後の更新された確信）
- $P(D \mid H)$ = 尤度（データが仮説にどの程度合致するか）
- $P(H)$ = 事前分布（データを見る前の確信）
- $P(D)$ = エビデンス（データの全確率）

**応用:** 新しい情報による確信の更新

**関連項目**: 事前分布, 事後分布, 尤度
{{% /expand %}}

### 濃度（基数） 📘
{{% expand "濃度（基数）" %}}
集合の*濃度*または*大きさ*とは、その集合が含む要素の個数です。$A = \{H, T\}$ のとき、$A$ の濃度は $|A|=2$ です。

**記法:** $|A|$ は「集合 $A$ の大きさ」を意味します

**プログラミングでは**: Python の `len(A)` や配列要素の数え上げに相当します
{{% /expand %}}

### 条件付き確率 📘
{{% expand "条件付き確率" %}}
*条件付き確率*とは、別の事象についての知識を条件として与えたときの、ある事象の確率です。ある事象を条件とするとは、その事象における可能な結果が可能性の集合（結果空間）を形成することを意味します。その後、その*制限された*結果空間の中で通常通りに確率を計算します。

**形式的定義:** $P(A \mid B) = \frac{|A \cap B|}{|B|}$、ここで $\mid$ の左側は確率を知りたい対象であり、$\mid$ の右側は真であることが知られている事柄です。

**別の公式:** $P(A \mid B) = \frac{P(A,B)}{P(B)}$（$P(B) > 0$ を仮定）

**GenJAXでは 💻**: `ChoiceMap` を使って観測値を指定することで条件付けを行います
{{% /expand %}}

### 依存性 📘
{{% expand "依存性" %}}
一方の確率変数や事象の結果を知ることが、もう一方の確率に影響を与えるとき、それらの変数や事象は*従属*であると言います。これは $A \not\perp B$ と表記されます。

互いに影響し合わない場合は、*独立*であると言います。これは $A \perp B$ と表記されます。

**独立性の形式的定義:** $P(A \mid B) = P(A)$、あるいは同値に $P(A, B) = P(A) \times P(B)$

**例**: コイン投げは独立です（一方の結果がもう一方に影響しません）。カードを戻さずに引くときは従属です（最初の引きが2枚目に影響します）。
{{% /expand %}}

### 事象 📘
{{% expand "事象" %}}
*事象*とは、可能な結果を0個、いくつか、またはすべて含む集合です。言い換えると、事象は結果空間 $\Omega$ の任意の部分集合です。

**例:** 「少なくとも一皿のとんかつ」という事象は $\{HT, TH, TT\} \subseteq \Omega$ です。

**プログラミングでは**: 事象は条件を満たすサンプルのフィルタリング・集計に対応します
{{% /expand %}}

### 生成過程 📘💻
{{% expand "生成過程" %}}
*生成過程*とは、ランダムな選択を持つアルゴリズムに従って、可能な結果の確率を定義するものです。結果を生み出すためのレシピと考えてください。

**例:** 「2枚のコインを投げる：1枚目はランチ用（H か T）、2枚目はディナー用（H か T）。ペアを記録する。」

**GenJAXでは 💻**: 生成過程は `@gen` デコレータを付けた関数として記述します

```python
@gen
def chibany_day():
    lunch = flip(0.5) @ "lunch"
    dinner = flip(0.5) @ "dinner"
    return (lunch, dinner)
```

これにより、確率論的な考え方が実際に実行可能なコードと結びつきます！
{{% /expand %}}

### 同時確率 📘
{{% expand "同時確率" %}}
*同時確率*とは、複数の事象がすべて起こる確率です。これは事象の積集合（すべての事象に属する結果）に対応します。

**記法:** $P(A, B)$ または $P(A \cap B)$

**直感:** 「$A$ と $B$ の両方が起こる確率はどれくらいか？」

**例**: $P(\text{lunch}=T, \text{dinner}=T) = P(TT)$
{{% /expand %}}

### 周辺確率 📘
{{% expand "周辺確率" %}}
*周辺確率*とは、1つ以上の他の確率変数の可能な値について和をとることで計算された確率変数の確率です。

**公式:** $P(A) = \sum_{b} P(A, B=b)$

**直感:** 「$B$ が何であるかに関わらず、$A$ の確率はどれくらいか？」

**例**: $P(\text{lunch}=T) = P(TH) + P(TT)$（ディナーに関わらず、ランチがとんかつである確率）
{{% /expand %}}

### マルコフ同値類 📘
{{% expand "マルコフ同値類" %}}
*マルコフ同値類*とは、**まったく同じ条件付き独立性の集合**をエンコードするすべての有向非巡回グラフ（DAG）の集合です。同じクラスに属する2つのグラフは*マルコフ同値*と呼ばれます：それらは同時分布に同一の制約を課するため、**どれだけ多くの観測データでも区別することはできません**。データはクラス内のすべてのグラフと同等に適合します。

**直感:** いくつかの辺の向きを逆にしても、グラフの*統計的な*内容が完全に変わらないことがあります。2変数の場合、$T \to C$ と $C \to T$ はマルコフ同値です。両者は同じ同時分布 $P(T,C)$ に分解され、「$T$ と $C$ は従属である」とだけ主張します。それらを区別するには、観測ではなく*介入*（do演算子）が必要です。

**同値を破る例外:** *コライダー* $A \to B \leftarrow C$ は、その逆向きの変形が持たない独立性（$A \perp C$）を主張するため、コライダーは一般に対応する連鎖やフォークと同値ではありません。（同じスケルトン、異なる「v構造」⇒異なるクラス。）

**登場箇所:** [チュートリアル3、第9章：条件付き独立性](../intro2/09_conditional_independence/)（連鎖・フォーク・コライダー、d分離）と[第10章：因果ベイズネット](../intro2/10_causal_bayes_nets/#the-same-statistics-three-different-stories)（do演算子、介入）。
{{% /expand %}}

### 結果空間 📘
{{% expand "結果空間" %}}
*結果空間*（ギリシャ文字オメガ $\Omega$ で表記）とは、ランダムな過程における可能なすべての結果の集合です。確率を計算する基盤となります。

**例:** Chibany の1日2食について、$\Omega = \{HH, HT, TH, TT\}$。

**GenJAXでは 💻**: `simulate()` を何度も実行することで、結果空間から結果を生成します
{{% /expand %}}

### 確率 📘
{{% expand "確率" %}}
結果空間 $\Omega$ に対する事象 $A$ の*確率*は、それらの大きさの比です：$P(A) = \frac{|A|}{|\Omega|}$。

結果に重みがある（等確率でない）場合は、数え上げる代わりに重みを合計します。

**解釈:** 「可能な結果のうち、事象 $A$ に含まれるものはどれくらいの割合か？」

**コードでは**: これをシミュレーションで近似します：過程を何度も実行し、事象が起こった実行の割合を計算します。
{{% /expand %}}

### 確率変数 📘
{{% expand "確率変数" %}}
*確率変数*とは、可能な結果の集合から何らかの集合や空間へと写像する関数です。関数の出力または値域は、結果の集合そのものでも、結果に基づく整数（例：とんかつの数を数える）でも、より複雑なものでも構いません。

技術的には出力が*可測*でなければなりません。確率変数の出力が非常に大きく（連続値のように）なる場合を除き、その区別を気にする必要はありません。連続確率変数上の確率については、チュートリアル3📊でさらに説明します。

**重要な洞察:** 「ランダム」と呼ばれるのは、その値がどの結果が起こるかに依存するからですが、本質的にはただの関数です！

**例**: $X(\omega)$ = 結果 $\omega$ におけるとんかつの食事の数
{{% /expand %}}

### 集合 📘
{{% expand "集合" %}}
*集合*とは、要素またはメンバーの集まりです。集合は、含むまたは含まない要素によって定義されます。要素はコンマで区切って列挙され、「$\{$」が集合の始まりを、「$\}$」が終わりを表します。集合の要素は一意であることに注意してください。

**例:** $\{H, T\}$ は2つの要素 H と T を含む集合です。

**プログラミングでは**: Python の集合 `{0, 1}` や一意な要素のリストのようなものです
{{% /expand %}}

---

## GenJAXプログラミング（チュートリアル2）

### @gen デコレータ 💻
{{% expand "@gen デコレータ" %}}
GenJAX の `@gen` デコレータは、Python の関数をアドレス付きランダム選択を行い確率的推論に使用できる*生成関数*としてマークします。

**使い方**:
```python
@gen
def my_model():
    x = bernoulli(0.5) @ "x"  # Random choice at address "x"
    return x
```

**動作内容**:
- 行われたすべてのランダム選択を追跡する
- 観測値への条件付けを可能にする
- 推論（重点サンプリング、MCMCなど）を有効化する

**関連項目**: 生成関数, トレース, ChoiceMap
{{% /expand %}}

### ベルヌーイ分布 💻
{{% expand "ベルヌーイ分布" %}}
1回の二値試行（成功/失敗、1/0、真/偽）を表す確率分布。数学者ヤコブ・ベルヌーイにちなんで命名されました。

**パラメータ**: $p$ = 成功（1を返す）の確率

**意味**: 単一のyes/noの結果。確率 $p$ で表が出て確率 $1-p$ で裏が出る偏ったコイン投げと考えてください。

**GenJAXでは**:
```python
@gen
def coin_flip():
    is_heads = flip(0.5) @ "coin"  # 50% chance of 1 (heads)
    return is_heads
```

**注意**: GenJAXでは `bernoulli(p)` の代わりに `flip(p)` を使います——名前はコイン投げのメタファーを反映しています！

**返り値**: `True`/`1`（成功）または `False`/`0`（失敗）

**使用例**: コイン投げ、yes/no問題、オン/オフ状態、二値決定

**関連項目**: flip(), カテゴリカル分布（複数の結果への一般化）
{{% /expand %}}

### flip() 💻
{{% expand "flip()" %}}
ベルヌーイ分布からサンプリングするためのGenJAXの関数。名前はコイン投げのメタファーを反映しています。

**シグネチャ**: `flip(p)`

**パラメータ**:
- `p` - `True`/`1` を返す確率（表が出るようなもの）

**返り値**: `True` または `False`（JAX配列では `1` または `0` として表現）

**GenJAXでは**:
```python
@gen
def coin_flip():
    result = flip(0.7) @ "coin"  # 70% chance of True (heads)
    return result
```

**よく使う値**:
- `flip(0.5)` - 公平なコイン投げ（50/50）
- `flip(0.8)` - True寄り（80%の確率）
- `flip(0.2)` - False寄り（Falseが80%の確率）

**`bernoulli` ではなく `flip` を使う理由は？** GenJAXは両方の関数を持っていますが、引数が異なります：
- `flip(p)` - **確率**（0から1）を受け取る——ほとんどのユーザーにとって直感的
- `bernoulli(logit)` - **ロジット**（対数オッズ、$-\infty$ から $+\infty$）を受け取る——TensorFlowの規約を引き継いだもの

ほとんどのユーザーは `flip()` を使うべきです。確率論の期待通りに動作します（70%の確率でtrueなら0.7を渡す）。

**関連項目**: ベルヌーイ分布
{{% /expand %}}

### カテゴリカル分布 💻📊
{{% expand "カテゴリカル分布" %}}
指定された確率を持つ離散的な結果に対する確率分布。

**パラメータ**: 合計が1.0になる確率の配列

**GenJAXでは**:
```python
@gen
def roll_die(probs):
    outcome = categorical(probs) @ "roll"  # Returns 0,1,2,3,4, or 5
    return outcome
```

**例**: 公平な4面サイコロには `categorical([0.25, 0.25, 0.25, 0.25])`

**返り値**: 整数インデックス（0, 1, 2, ..., k-1）

**チュートリアル1📘との接続**: 集合を使って学んだ離散的な結果空間を一般化したもの

**📊での使用**: 混合モデルおよびDPMMにおけるクラスタ割り当て
{{% /expand %}}

### ChoiceMap 💻
{{% expand "ChoiceMap" %}}
ランダム選択の観測値を指定するためのGenJAXの方法。アドレス（名前）を値にマッピングする辞書のような構造です。

**用途**:
- （トレースから）どのランダム選択が行われたかを記録する
- 推論のための観測値を指定する
- ランダム選択を制約する

**コードでは**:
```python
from genjax import ChoiceMap

# Observe x=2.5
observations = ChoiceMap.d({"x": 2.5})

# Or use builder pattern
cm = ChoiceMap.empty()
cm = cm.set("x", 2.5)
cm = cm.set("y", 1.0)
```

**考え方**: すべてのランダムな決定に名前を付けて追跡する方法

**関連項目**: トレース, Target
{{% /expand %}}

### 生成関数 💻
{{% expand "生成関数" %}}
GenJAXにおける生成関数とは、`@gen` デコレータを付けたPythonの関数であり、アドレス付きランダム選択を行うことができます。戻り値に対する確率分布を表します。

**構造**:
```python
@gen
def model(params):
    # Random choices with addresses
    x = distribution(params) @ "address"
    y = another_distribution(x) @ "another_address"
    return result
```

**主な特徴**:
- 名前付きアドレスでランダム選択を行う
- 観測値への条件付けが可能
- 推論操作をサポートする

**関連項目**: @gen デコレータ, トレース, ChoiceMap
{{% /expand %}}

### 重点サンプリング 💻📊
{{% expand "重点サンプリング" %}}
事後分布を以下のように近似する推論手法：
1. 提案分布からサンプルを生成する
2. 各サンプルを観測値への適合度で重み付けする
3. 重み付きサンプルを使って事後分布を近似する

**GenJAXでは**:
```python
trace, log_weight = target.importance(key, choicemap)
```

**重要な概念**: [有効サンプルサイズ](#effective-sample-size-)は重みの分布がどれほど良いかを測定します。提案分布がターゲットと一致するときはサンプル数に近く、1つのサンプルが支配するときは約1になります。[重要度重み](#importance-weight-) $w = p/q$ が核心的な補正項です。

**📊での使用**: ベイズモデルおよびDPMMの事後推論；完全な形は[第16章：モンテカルロ](../intro2/16_monte_carlo/#importance-sampling-sample-the-wrong-distribution)（自己正規化形式、尤度重み付け）で導入され、逐次バージョンが[第17章](../intro2/17_particle_filtering/)の[粒子フィルタ](#particle-filter-)を駆動します。

**関連項目**: [重要度重み](#importance-weight-), [有効サンプルサイズ](#effective-sample-size-), [提案分布](#proposal-distribution-), [Target](#target-), [重み退化](#weight-degeneracy-)
{{% /expand %}}

### JAXキー 💻
{{% expand "JAXキー" %}}
JAXはランダム性を制御するために明示的なランダムキーを使用します（NumPyのグローバルなランダム状態とは異なります）。明示的に渡すシードのようなものだと考えてください。

**理由**: 再現性と関数型プログラミングパターンを実現するため

**使い方**:
```python
import jax

# Create a key
key = jax.random.key(42)  # 42 is the seed

# Split into multiple keys
keys = jax.random.split(key, num=100)  # Get 100 independent keys

# Use a key
trace = model.simulate(keys[0], ())
```

**ベストプラクティス**: 常にキーを分割し、同じキーを2回使い回さない

**関連項目**: vmap（よく一緒に使われる）
{{% /expand %}}

### モンテカルロシミュレーション 📘💻
{{% expand "モンテカルロシミュレーション" %}}
多数のランダムサンプルを生成し結果を数え上げることで確率を近似する計算手法。モナコのモンテカルロカジノにちなんで命名されました。

**手順:**
1. 多数のランダムな結果を生成する（例：10,000日のシミュレーション）
2. 事象を満たすものを数える
3. 割合を計算する

**GenJAXでは**:
```python
# Generate 10,000 samples
keys = jax.random.split(key, 10000)
samples = jax.vmap(lambda k: model.simulate(k, ()).get_retval())(keys)

# Count event occurrences
event_count = jnp.sum(samples >= threshold)
probability = event_count / 10000
```

**有効な場面:** 手で列挙するには結果空間が大きすぎるとき

**📊での展開**: [第16章：モンテカルロ](../intro2/16_monte_carlo/#the-monte-carlo-estimator)では推定量 $\hat\mu_n$ をゼロから構築します（サイコロ、ダーツによる$\pi$の推定）。$1/\sqrt{n}$ の誤差率、[棄却サンプリング](#rejection-sampling-)、[重点サンプリング](#importance-sampling-)も扱います。

**関連項目**: [vmap](#vmap-), [トレース](#trace-), [重点サンプリング](#importance-sampling-), [棄却サンプリング](#rejection-sampling-), [有効サンプルサイズ](#effective-sample-size-)
{{% /expand %}}

### 正規分布 💻📊
{{% expand "正規分布" %}}
**ガウス分布**を参照してください（同じものです）
{{% /expand %}}

### simulate() 💻
{{% expand "simulate()" %}}
`simulate()` メソッドは、生成関数の1回のランダムな実行を生成します。

**シグネチャ**:
```python
trace = model.simulate(key, args)
```

**パラメータ**:
- `key`: JAXランダムキー
- `args`: 生成関数への引数のタプル
- オプション: 条件付けのための `observations`（ChoiceMap）

**返り値**: すべてのランダム選択と戻り値を含むトレース

**例**:
```python
@gen
def coin_flip():
    return bernoulli(0.5) @ "flip"

trace = coin_flip.simulate(key, ())
result = trace.get_retval()  # 0 or 1
```

**関連項目**: トレース, importance(), JAXキー
{{% /expand %}}

### Target 💻
{{% expand "Target" %}}
GenJAXでは、`Target` は生成関数を観測値に条件付けることで作成されます。事後分布を表します。

**Targetの作成**:
```python
from genjax import Target

# Observe some data
observations = ChoiceMap.d({"x_0": 2.5, "x_1": 3.0})

# Create target (posterior)
target = Target(model, (params,), observations)
```

**推論への使用**:
```python
# Importance sampling
trace, log_weight = target.importance(key, ChoiceMap.empty())
```

**重要な概念**: Targetは $P(\text{潜在変数} \mid \text{観測値})$ を表します

**関連項目**: ChoiceMap, 重点サンプリング, 事後分布
{{% /expand %}}

### トレース 💻
{{% expand "トレース" %}}
確率的プログラミングにおいて、*トレース*は生成関数の1回の実行中に行われたすべてのランダム選択を、そのアドレス（名前）と戻り値とともに記録します。

**考え方:** 確率的プログラムの1回の実行において「何が起こったか」の完全な記録

**構造**:
```python
trace = model.simulate(key, args)

# Access components
retval = trace.get_retval()         # Return value
choices = trace.get_choices()        # ChoiceMap with all random choices
log_prob = trace.get_score()         # Log probability of this trace
```

**例**:
```python
@gen
def example():
    x = flip(0.5) @ "x"
    y = normal(0, 1) @ "y"
    return x + y

trace = example.simulate(key, ())
print(trace.get_choices()["x"])  # e.g., True or False
print(trace.get_choices()["y"])  # e.g., 0.234
print(trace.get_retval())        # e.g., 1.234
```

**使用場所:** GenJAXおよびその他の確率的プログラミングシステム

**関連項目**: ChoiceMap, 生成関数
{{% /expand %}}

### vmap 💻
{{% expand "vmap" %}}
JAXの「ベクトル化マップ」——関数を多数の入力に並列で適用します（非常に高速！）。

**概念**: forループで逐次実行する代わりに、vmapはGPU/CPU上で操作を並列に実行します。

**使い方**:
```python
import jax

# Regular loop (slow)
results = []
for key in keys:
    results.append(model.simulate(key, ()).get_retval())

# vmap (fast!)
def run_once(key):
    return model.simulate(key, ()).get_retval()

results = jax.vmap(run_once)(keys)
```

**考え方**: 「この関数を10,000回実行するが、すべて一度にまとめて行う」

**高速な理由**: 並列ハードウェア（GPU、ベクトル化CPU演算）を活用するため

**関連項目**: JAXキー, モンテカルロシミュレーション
{{% /expand %}}

---

## 連続確率（チュートリアル3）

### ベータ分布 📊
{{% expand "ベータ分布" %}}
区間 [0,1] 上の連続確率分布で、2つの形状パラメータ $\alpha$ と $\beta$ によってパラメータ化されます。

**パラメータ**:
- $\alpha$（アルファ）——形状パラメータ
- $\beta$（ベータ）——形状パラメータ

**PDF**: $p(x \mid \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}$

**GenJAXでは**:
```python
@gen
def stick_breaking(alpha):
    # Beta(1, alpha) for stick-breaking
    beta_k = beta(1.0, alpha) @ f"beta_{k}"
    return beta_k
```

**特殊なケース**:
- Beta(1,1) = Uniform(0,1)
- Beta(α,α) は 0.5 を中心に対称

**📊での使用**:
- ディリクレ過程のスティック破砕法
- 確率と比率のモデリング
- ベルヌーイ/二項分布の共役事前分布

**関連項目**: ディリクレ分布, スティック破砕法
{{% /expand %}}

### 中国料理店過程（CRP） 📊
{{% expand "中国料理店過程" %}}
ディリクレ過程を理解するためのメタファーとアルゴリズム。無限のテーブルがあるレストランに客が入ってくると想像してください：
- 最初の客はテーブル1に座る
- 次の客：占有されているテーブルにその占有率に比例した確率で座るか、あるいは α に比例した確率で新しいテーブルに座る

**パラメータ**: α（集中パラメータ）

**性質**:
- 「富めるものはさらに富む」——人気のテーブルにはさらに多くの客が集まる
- しかし常に新しいテーブルを始める可能性がある
- α は新しいクラスタを作る傾向を制御する

**DPMMとの接続**: 各テーブル = クラスタ。CRPがクラスタ割り当てを決定し、各クラスタは独自のガウス分布を持つ。

**コードでは直接使用しない**: スティック破砕法は数学的に同等だが実装上より実用的

**関連項目**: ディリクレ過程, DPMM, スティック破砕法
{{% /expand %}}

### 集中パラメータ（α） 📊
{{% expand "集中パラメータ（α）" %}}
ディリクレ過程および関連モデルにおけるパラメータ α は、新しいクラスタを作る傾向と既存のクラスタを再利用する傾向を制御します。

**効果**:
- **小さな α**（例：0.1）：クラスタ数が少なく、既存クラスタへの強い選好
- **中程度の α**（例：1〜5）：探索と活用のバランス
- **大きな α**（例：10以上）：クラスタ数が多く、新しいクラスタを作る高い確率

**スティック破砕法では**:
```python
beta_k = beta(1.0, alpha) @ f"beta_{k}"
```

**直感**: α は新しいクラスタに対する「事前の強さ」のようなものです。α が高いほど、既存クラスタへの当てはめよりも新しいクラスタでデータを説明することを好みます。

**典型的な範囲**: ほとんどの応用では 0.1 から 10

**関連項目**: ディリクレ過程, DPMM, スティック破砕法
{{% /expand %}}

### 共役事前分布 📊
{{% expand "共役事前分布" %}}
事前分布がある尤度に対して*共役*であるとは、事後分布が事前分布と同じ分布族に属するときを指します。

**有用な理由**: 閉じた形の事後計算を可能にします（サンプリング不要）

**古典的な例**:
- **ベータ-二項**: ベータ事前分布 × 二項尤度 = ベータ事後分布
- **ガンマ-ポアソン**: ガンマ事前分布 × ポアソン尤度 = ガンマ事後分布
- **ガウス-ガウス**: 正規事前分布 × 正規尤度 = 正規事後分布

**例（ガウス-ガウス）**:
```python
# Prior: μ ~ Normal(μ₀, σ₀²)
# Likelihood: x | μ ~ Normal(μ, σ²)
# Posterior: μ | x ~ Normal(μ_post, σ_post²)  # Still Gaussian!

# Posterior parameters:
# μ_post = (σ²·μ₀ + σ₀²·x) / (σ² + σ₀²)
# σ_post² = (σ²·σ₀²) / (σ² + σ₀²)
```

**トレードオフ**: 数学的な便利さとモデリングの柔軟性

**チュートリアル3、第4章**でガウス-ガウス共役性を詳しく扱います

**関連項目**: 事前分布, 事後分布, ベイズ学習
{{% /expand %}}

### 累積分布関数（CDF） 📊
{{% expand "累積分布関数（CDF）" %}}
連続確率変数について、CDFはその変数がある値以下になる確率を与えます：

$$F(x) = P(X \leq x) = \int_{-\infty}^x p(t)   dt$$

**主な性質**:
- 常に増加（または一定）
- 0 から 1 の範囲
- $F(-\infty) = 0$ かつ $F(\infty) = 1$
- CDFの導関数 = PDF: $\frac{dF}{dx} = p(x)$

**解釈**: 「この値以下の値を得る確率はどれくらいか？」

**例（標準正規）**:
- CDF(0) ≈ 0.5（≤ 0 になる確率が 50%）
- CDF(1.96) ≈ 0.975（≤ 1.96 になる確率が 97.5%）

**コードでは**: GenJAXでは通常直接必要としません（代わりにサンプリングを使用）が、分位数や確率を理解するのに役立ちます

**関連項目**: PDF, 分位数
{{% /expand %}}

### ディリクレ分布 📊
{{% expand "ディリクレ分布" %}}
ベータ分布の多変量への一般化。合計が1になる確率ベクトルを生成します。

**パラメータ**: α = (α₁, α₂, ..., αₖ) — 集中パラメータ

**出力**: すべての pᵢ > 0 かつ Σpᵢ = 1 を満たすベクトル (p₁, p₂, ..., pₖ)

**GenJAXでは**:
```python
@gen
def mixture_weights(alpha_vector):
    # Returns a probability distribution over K categories
    probs = dirichlet(alpha_vector) @ "probs"
    return probs
```

**特殊なケース**: Dirichlet(1,1,1,...,1) = 確率シンプレックス上の一様分布

**直感**: 重み自体がランダムな重み付きサイコロを転がすようなもの

**使用場所**:
- GMM における混合重みの事前分布
- ~~DPMM~~（直接は使わない——スティック破砕法が代わりに使われる）
- トピックモデリング（LDA）

**関連項目**: ベータ分布, カテゴリカル分布
{{% /expand %}}

### ディリクレ過程（DP） 📊
{{% expand "ディリクレ過程" %}}
分布上の分布。クラスタ/成分の数が不明な混合モデルの*事前分布*です。

**パラメータ**:
- α（集中パラメータ）——クラスタ形成を制御
- G₀（基底分布）——クラスタの「プロトタイプ」分布

**主な性質**:
- **無限混合**: 任意の数のクラスタを持てる
- **自動モデル選択**: データが有効なクラスタ数を決定する
- **クラスタリング性質**: 一部のサンプルが同じパラメータ値（クラスタ）を共有することを強制する

**2つの表現**:
1. **中国料理店過程**（CRP）——比喩的、逐次的
2. **スティック破砕法**——構成的、実装上実用的

**「ディリクレ過程」の理由**: ディリクレ分布を無限次元に一般化したもの

**実際には**: K を指定せずにクラスタリングするために DPMM を介して使用される

**チュートリアル3、第6章**でDPを詳しく扱います

**関連項目**: DPMM, スティック破砕法, 中国料理店過程
{{% /expand %}}

### ディリクレ過程混合モデル（DPMM） 📊
{{% expand "ディリクレ過程混合モデル（DPMM）" %}}
データからクラスタ数を自動的に決定する無限混合モデル。

**構造**:
```
1. Generate cluster parameters using stick-breaking:
   - β₁, β₂, ... ~ Beta(1, α)
   - π₁ = β₁, π₂ = β₂(1-β₁), π₃ = β₃(1-β₁)(1-β₂), ...

2. For each data point:
   - z ~ Categorical(π)  # Assign to cluster
   - x | z ~ Normal(μ_z, σ²)  # Generate from that cluster's Gaussian
```

**パラメータ**:
- α — クラスタ数を制御
- μ₀, σ₀ — クラスタ平均の事前分布
- σ — 観測ノイズ

**GenJAXでは**:
```python
@gen
def dpmm(alpha, mu0, sig0, sigx):
    # Stick-breaking for mixture weights
    pis = stick_breaking_construction(alpha, K)

    # Cluster means
    mus = [normal(mu0, sig0) @ f"mu_{k}" for k in range(K)]

    # Assign data points and generate observations
    for i in range(N):
        z_i = categorical(pis) @ f"z_{i}"
        x_i = normal(mus[z_i], sigx) @ f"x_{i}"
```

**利点**:
- 事前に K を指定する必要がない
- 原理に基づいたベイズ的不確実性
- モデルの複雑さを自動で制御する

**課題**:
- 打ち切り近似が必要（K クラスタで近似）
- 大規模データセットでは推論が遅くなることがある
- α の選択に敏感

**チュートリアル3、第6章**に完全な実装とインタラクティブなノートブックがあります

**関連項目**: GMM, ディリクレ過程, スティック破砕法
{{% /expand %}}

### 期待値 📊
{{% expand "期待値" %}}
確率によって重み付けされた確率変数の平均値。*平均*または*期待値*とも呼ばれます。

**離散の場合**: $E[X] = \sum_{x} x \cdot P(X=x)$

**連続の場合**: $E[X] = \int_{-\infty}^{\infty} x \cdot p(x)   dx$

**GenJAXでは**（サンプリングによる近似）:
```python
# Generate many samples
samples = [model.simulate(key_i, ()).get_retval() for key_i in keys]

# Expected value ≈ average of samples
expected_value = jnp.mean(samples)
```

**性質**:
- 線形性: $E[aX + bY] = aE[X] + bE[Y]$
- 独立な変数に対して: $E[XY] = E[X]E[Y]$

**解釈**: 「この実験を何度も繰り返したら、平均的な結果はどうなるか？」

**チュートリアル3、第1章**では「謎の弁当」のパラドックスで期待値を扱います

**関連項目**: 分散, 反復期待値の法則
{{% /expand %}}

### ガウス分布 📊
{{% expand "ガウス分布" %}}
*正規分布*とも呼ばれます。統計学と機械学習に広く見られる有名なベル曲線です。

**パラメータ**:
- μ（ミュー）——平均（ベルの中心）
- σ²（シグマ二乗）——分散（ベルの幅）

**PDF**:
$$p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**GenJAXでは**:
```python
@gen
def gaussian_model():
    x = normal(mu, sigma) @ "x"  # Note: sigma, not sigma²
    return x
```

**68-95-99.7則**:
- データの68%が μ ± σ の範囲内
- データの95%が μ ± 2σ の範囲内
- データの99.7%が μ ± 3σ の範囲内

**なぜよく使われるか**:
- 中心極限定理（和はガウス分布に収束する）
- 与えられた平均と分散に対して最大エントロピーを持つ分布
- 数学的に扱いやすい（共役事前分布！）

**チュートリアル3、第3章**でガウス分布を詳しく扱います

**関連項目**: 正規分布（同じもの）, 標準正規分布
{{% /expand %}}

### ガウス混合モデル（GMM） 📊
{{% expand "ガウス混合モデル（GMM）" %}}
それぞれ固有の平均、分散、混合重みを持つ複数のガウス分布の混合。

**構造**:
```
1. Choose cluster k with probability πₖ
2. Sample from Normal(μₖ, σₖ²)
```

**パラメータ**:
- K — 成分数（事前に指定が必要）
- π₁, ..., πₖ — 混合重み（合計が1）
- μ₁, ..., μₖ — 成分平均
- σ₁², ..., σₖ² — 成分分散

**GenJAXでは**:
```python
@gen
def gmm(pis, mus, sigmas):
    # Choose component
    z = categorical(pis) @ "z"

    # Sample from chosen component
    x = normal(mus[z], sigmas[z]) @ "x"
    return x
```

**使用場面**:
- 複数グループを持つデータのクラスタリング
- 多峰性分布のモデリング
- 密度推定

**制限**: 事前に K を指定する必要がある（DPMMがこれを解決します！）

**チュートリアル3、第5章**でGMMを扱います

**関連項目**: DPMM, 混合モデル
{{% /expand %}}

### 尤度 📊
{{% expand "尤度" %}}
特定のパラメータ値が与えられたときにデータを観測する確率：$P(D \mid \theta)$

**重要な区別**:
- データの関数として（θ 固定）：**確率**
- パラメータの関数として（データ固定）：**尤度**

**ベイズの定理では**:
$$P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}$$
- $P(D \mid \theta)$ は**尤度**
- $P(\theta)$ は**事前分布**
- $P(\theta \mid D)$ は**事後分布**

**例**:
```python
# Observed data: x = [2.5, 3.0, 2.8]
# Model: x[i] ~ Normal(μ, 1.0)

# Likelihood of μ = 3.0:
likelihood = product([
    normal_pdf(2.5, mu=3.0, sigma=1.0),
    normal_pdf(3.0, mu=3.0, sigma=1.0),
    normal_pdf(2.8, mu=3.0, sigma=1.0)
])
```

**GenJAXでは**: トレースの対数確率には尤度が含まれます

**関連項目**: 事後分布, 事前分布, ベイズの定理
{{% /expand %}}

### 混合モデル 📊
{{% expand "混合モデル" %}}
ある確率でそれぞれが活性化される複数の成分分布を組み合わせた確率モデル。

**一般形**:
$$p(x) = \sum_{k=1}^K \pi_k \cdot p_k(x)$$

ここで：
- πₖ = 混合重み（確率、合計が1）
- pₖ(x) = 成分分布

**生成過程**:
1. 確率 πₖ で成分 k を選択する
2. 成分 pₖ からサンプリングする

**よくある種類**:
- **ガウス混合モデル（GMM）**: 成分がガウス分布
- **DPMM**: 無限混合（K → ∞）

**なぜ有用か**:
- 複雑な多峰性分布をモデル化できる
- ソフトクラスタリングを実現できる
- 異質な集団を表現できる

**チュートリアル3、第5章**で有限混合（GMM）を扱います
**チュートリアル3、第6章**で無限混合（DPMM）を扱います

**関連項目**: GMM, DPMM, カテゴリカル分布
{{% /expand %}}

### 事後分布 📊
{{% expand "事後分布" %}}
データを観測した後のパラメータ上の更新された確率分布：$P(\theta \mid D)$

**ベイズの定理より**:
$$P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}$$

- $P(\theta)$ = **事前分布**（データを見る前）
- $P(D \mid \theta)$ = **尤度**（θ がデータをどの程度説明するか）
- $P(D)$ = **エビデンス**（正規化定数）
- $P(\theta \mid D)$ = **事後分布**（データを見た後）

**GenJAXでは**:
```python
# Specify observations
observations = ChoiceMap.d({"x_0": 2.5, "x_1": 3.0})

# Create posterior target
target = Target(model, (params,), observations)

# Sample from posterior
trace, log_weight = target.importance(key, ChoiceMap.empty())
```

**解釈**: 「観測したことを踏まえると、どのパラメータ値が最もありそうか？」

**チュートリアル3、第4章**でベイズ学習と事後分布を扱います

**関連項目**: 事前分布, 尤度, ベイズの定理
{{% /expand %}}

### 予測分布 📊
{{% expand "予測分布" %}}
すでに観測したデータが与えられたときの、新しい未観測データ上の分布。

**事後予測分布**: $P(x_{\text{new}} \mid D) = \int P(x_{\text{new}} \mid \theta) \cdot P(\theta \mid D)   d\theta$

**言い換えると**:
1. 可能なすべてのパラメータ値 θ を考慮する
2. それぞれを事後確率 P(θ | D) で重み付けする
3. 新しいデータへの予測を平均する

**GenJAXでは**（サンプリングによる）:
```python
# 1. Get posterior samples for θ
posterior_samples = []
for key in keys:
    trace, _ = target.importance(key, ChoiceMap.empty())
    theta = trace.get_choices()["theta"]
    posterior_samples.append(theta)

# 2. For each θ, generate predictions
predictions = []
for theta in posterior_samples:
    x_new = generate_new_data(theta)
    predictions.append(x_new)

# predictions is now a sample from the predictive distribution!
```

**なぜ重要か**: パラメータと新しいデータの両方に関する不確実性を捉えます

**チュートリアル3、第4章**でベイズ学習の予測分布を示します

**関連項目**: 事後分布, 事前分布
{{% /expand %}}

### 事前分布 📊
{{% expand "事前分布" %}}
データを観測する*前*のパラメータ上の確率分布：$P(\theta)$

**ベイズの定理では**:
$$P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}$$

- $P(\theta)$ = **事前分布**（初期の確信）
- $P(\theta \mid D)$ = **事後分布**（データ D を見た後の更新された確信）

**事前分布の種類**:
- **情報的事前分布**: 強い確信（例：Normal(0, 0.1²) は μ が 0 に近いことを示す）
- **弱情報的事前分布**: 穏やかな誘導（例：Normal(0, 10²)）
- **無情報/平坦事前分布**: 好みなし（例：Uniform(-∞, ∞)）

**GenJAXでは**:
```python
@gen
def bayesian_model(mu0, sigma0):
    # Prior: μ ~ Normal(mu0, sigma0)
    mu = normal(mu0, sigma0) @ "mu"

    # Likelihood: x | μ ~ Normal(μ, 1.0)
    x = normal(mu, 1.0) @ "x"
    return x
```

**論争点**: 事前分布の主観性は、ベイズ的手法の特徴（知識をエンコードできる）でもあり批判点（結果を偏らせる）でもあります

**チュートリアル3、第4章**でベイズ学習における事前分布を論じます

**関連項目**: 事後分布, 尤度, 共役事前分布
{{% /expand %}}

### 確率密度関数（PDF） 📊
{{% expand "確率密度関数（PDF）" %}}
連続確率変数について、PDFは各値における確率の*密度*を表します。

**重要な洞察**: $p(x)$ は確率ではありません！それは**密度**です。

**理由**:
- 厳密に特定の値をとる確率はゼロです（取りうる値が無限にあるため）
- 確率とは区間上のPDF曲線の**面積**です：
  $$P(a \leq X \leq b) = \int_a^b p(x)   dx$$

**性質**:
- $p(x) \geq 0$（非負）
- $\int_{-\infty}^{\infty} p(x)   dx = 1$（全面積 = 1）
- $p(x)$ は 1 より大きくなることがある！（確率ではなく密度であるため）

**例（ガウス）**:
$$p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**GenJAXでは**: 通常PDFを直接計算するのではなく、PDFからサンプリングします

**離散版📘との接続**: PDFは確率質量関数（PMF）の連続アナログです

**チュートリアル3、第2章**でPDFを導入します

**関連項目**: CDF, 連続確率変数
{{% /expand %}}

### 標準正規分布 📊
{{% expand "標準正規分布" %}}
μ=0、σ²=1 のガウス分布。

**PDF**:
$$p(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)$$

**記法**: $X \sim \mathcal{N}(0,1)$

**なぜ特別か**:
- 基準分布（z スコア）
- 任意の Normal(μ, σ²) を標準化できる：$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0,1)$
- 表や関数はしばしば標準正規を使用する

**GenJAXでは**:
```python
z = normal(0.0, 1.0) @ "z"  # Standard normal
```

**関連項目**: ガウス分布, z スコア
{{% /expand %}}

### スティック破砕法 📊
{{% expand "スティック破砕法" %}}
「スティックを折る」ことによってディリクレ過程の無限混合重みを構成する方法。

**メタファー**: 長さ1のスティックから始めて、繰り返し以下を行う：
1. 残りのスティックの一部（β）を折り取る
2. その部分が次のクラスタの重みになる
3. 残りのスティックで続ける

**数学的過程**:
```
β₁, β₂, β₃, ... ~ Beta(1, α)

π₁ = β₁
π₂ = β₂ · (1 - β₁)
π₃ = β₃ · (1 - β₁) · (1 - β₂)
...
πₖ = βₖ · ∏(1 - βⱼ) for j < k
```

**性質**:
- すべての πₖ > 0
- Σ πₖ = 1（合計が1）
- πₖ は k が増えるにつれて（平均的に）減少する

**GenJAXでは**:
```python
@gen
def stick_breaking(alpha, K):
    betas = []
    pis = []

    for k in range(K):
        beta_k = beta(1.0, alpha) @ f"beta_{k}"
        betas.append(beta_k)

    # Convert betas to pis
    remaining = 1.0
    for k in range(K):
        pis.append(betas[k] * remaining)
        remaining *= (1.0 - betas[k])

    return jnp.array(pis)
```

**使用場所**: DPMM の実装

**チュートリアル3、第6章**でスティック破砕法を詳しく説明します

**関連項目**: ディリクレ過程, DPMM, ベータ分布
{{% /expand %}}

### 打ち切り（DPMMにおける） 📊
{{% expand "打ち切り" %}}
ディリクレ過程は理論上無限ですが、実際には K 成分に制限することで近似します。

**必要な理由**:
- 実際にはコードで無限次元を実装できない
- K 成分以降、残りの重みは無視できるほど小さくなる

**動作の仕組み**:
```python
# Truncated stick-breaking
K_max = 20  # Truncation level

# First K-1 components use stick-breaking
for k in range(K_max - 1):
    beta_k = beta(1.0, alpha) @ f"beta_{k}"
    pis[k] = beta_k * remaining
    remaining *= (1.0 - beta_k)

# Last component gets all remaining weight
pis[K_max - 1] = remaining
```

**K の選択**:
- 小さすぎる：真のクラスタ数を捉えられない
- 大きすぎる：推論が遅くなるが数学的には問題ない
- 経験則：K = 期待されるクラスタ数の 2〜3 倍

**品質チェック**: 最も大きなクラスタインデックスが有意な重みを持つ場合は K を増やす

**チュートリアル3、第6章**でDPMMにおける打ち切りを議論します

**関連項目**: DPMM, スティック破砕法
{{% /expand %}}

### 一様分布 📊
{{% expand "一様分布" %}}
範囲 [a, b] のすべての値が等しく起こりやすい連続分布。

**パラメータ**:
- a — 最小値
- b — 最大値

**PDF**:
$$p(x) = \begin{cases}
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}$$

**GenJAXでは**:
```python
@gen
def uniform_example():
    x = uniform(a, b) @ "x"
    return x
```

**性質**:
- 平均: (a + b) / 2
- 分散: (b - a)² / 12

**使用例**:
- ランダムな初期化
- 有界なパラメータの無情報事前分布
- 範囲内での「完全な無知」のモデリング

**離散版📘との接続**: 「すべての結果が等しく起こりやすい」の連続アナログ

**チュートリアル3、第2章**で一様分布を導入します

**関連項目**: PDF, 連続確率変数
{{% /expand %}}

### 分散 📊
{{% expand "分散" %}}
分布における広がり・ばらつきの尺度。平均からの二乗偏差の期待値。

**公式**: $\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$

**記法**:
- Var(X) または σ²
- 標準偏差: σ = √(Var(X))

**GenJAXでは**（サンプリングによる近似）:
```python
# Generate samples
samples = jnp.array([model.simulate(key_i, ()).get_retval() for key_i in keys])

# Variance ≈ sample variance
variance = jnp.var(samples)
std_dev = jnp.sqrt(variance)
```

**性質**:
- 常に非負
- Var(aX + b) = a² · Var(X)
- 独立な X, Y に対して: Var(X + Y) = Var(X) + Var(Y)

**解釈**: 「データはどれほど散らばっているか？」

**関連項目**: 期待値, 標準偏差, ガウス分布
{{% /expand %}}

### 重み退化 📊
{{% expand "重み退化" %}}
重点サンプリングにおける問題で、ほとんどのサンプルの重みが無視できるほど小さく、1つまたは少数のサンプルのみが意味のある寄与をする状態。

**症状**: 有効サンプルサイズ（ESS）≪ サンプル数

**例**:
<!-- validate: skip -->
```python
# Suppose 100 importance-sampling weights, but one dominates all the rest:
weights = [0.97] + [0.03 / 99] * 99   # one huge weight, 99 tiny ones

# Compute the effective sample size (ESS)
total = sum(weights)
normalized_weights = [w / total for w in weights]
ESS = 1.0 / sum(w**2 for w in normalized_weights)

# ESS ≈ 1.06 out of 100 — severe weight degeneracy!
```

**原因**:
- 事前分布と事後分布が大きく異なる
- 提案分布が事後分布に合っていない
- モデルの誤設定

**解決策**:
- より多くのサンプルを使う
- より良い提案分布を使う
- 別の推論手法（MCMC）を使う
- モデルを修正する（例：余分なランダム化を取り除く）

**チュートリアル3、第6章**: DPMMノートブックでは二重ランダム化バグにより重み退化（ESS=1/10）が発生しており、修正されました。逐次設定では、[粒子フィルタ](#particle-filter-)が毎ステップ[リサンプリング](#resampling-)しなければならない理由となっています（[第17章](../intro2/17_particle_filtering/#resampling-and-degeneracy)）。

**関連項目**: [重点サンプリング](#importance-sampling-), [有効サンプルサイズ](#effective-sample-size-), [リサンプリング](#resampling-), [粒子フィルタ](#particle-filter-)
{{% /expand %}}

### 驚き（情報量） 📊
{{% expand "驚き（情報量）" %}}
結果 $x$ の*驚き*（または*情報量*）は $-\log_2 P(x)$ であり、**ビット**単位で測られます。ある結果の確率が低いほど、それが起こったときにより驚くべきです；対数関数により、驚きは**独立な事象にわたって加法的**になります（独立な確率は乗法的なため）。

**公式:** $\text{surprise}(x) = -\log_2 P(x)$

**例:** $P=0.01$ と見積もった事象の驚きは $-\log_2 0.01 \approx 6.6$ ビット；$P=0.99$ と見積もった事象の驚きは $\approx 0.014$ ビット。

**登場箇所:** [チュートリアル3、第11章：情報理論](../intro2/11_information_theory/#surprise--log-px)

**関連項目:** [エントロピー](#entropy-)
{{% /expand %}}

### エントロピー 📊
{{% expand "エントロピー" %}}
確率変数 $X$ の*エントロピー*は、その**期待される驚き**——結果の平均的な不確実性（ビット単位）——です。決定論的な変数に対してはゼロで、一様分布（公平なコインでは1ビット）に対して最大になります。

**公式:** $H(X) = -\sum_x P(x) \log_2 P(x) = \mathbb{E}\bigl[-\log_2 P(X)\bigr]$

**直感:** 「平均的に $X$ にどれほど驚くか？」言い換えると、$X$ を特定するために必要なyes/no質問の平均回数。

**例:** 公平なコイン → 1ビット；$\text{Bernoulli}(0.7)$ → 0.881ビット；確定的な結果 → 0ビット。

**登場箇所:** [チュートリアル3、第11章：情報理論](../intro2/11_information_theory/#entropy--expected-surprise)

**関連項目:** [驚き](#surprise-information-content-), [同時エントロピー](#joint-entropy-), [条件付きエントロピー](#conditional-entropy-), [相互情報量](#mutual-information-)
{{% /expand %}}

### 同時エントロピー 📊
{{% expand "同時エントロピー" %}}
2変数の*同時エントロピー*は、ペア $(X, Y)$ を1つの結合した結果として扱ったときのエントロピーです——両方の合計的な不確実性です。

**公式:** $H(X, Y) = -\sum_{x,y} P(x, y) \log_2 P(x, y)$

**重要な等式（連鎖則）:** $H(X, Y) = H(X) + H(Y \mid X)$ ——全体の不確実性 = $X$ の不確実性 + $X$ を知った後の $Y$ の残りの不確実性。（これは確率の連鎖則 $P(x,y)=P(x)P(y\mid x)$ の対数です。）

**登場箇所:** [チュートリアル3、第11章：情報理論](../intro2/11_information_theory/#joint-and-conditional-entropy)

**関連項目:** [エントロピー](#entropy-), [条件付きエントロピー](#conditional-entropy-), [相互情報量](#mutual-information-)
{{% /expand %}}

### 条件付きエントロピー 📊
{{% expand "条件付きエントロピー" %}}
*条件付きエントロピー* $H(Y \mid X)$ は、$X$ を知った後に $Y$ に**残る**不確実性です——$P(Y \mid X = x)$ のエントロピーを $X$ について平均したもの。

**公式:** $H(Y \mid X) = -\sum_{x,y} P(x, y) \log_2 P(y \mid x) = H(X, Y) - H(X)$

**限界値:** $X$ が $Y$ を決定するなら $H(Y \mid X) = 0$（学ぶべきことが残っていない）；$X$ が $Y$ と独立なら $H(Y \mid X) = H(Y)$（$X$ を知っても助けにならなかった）。

**登場箇所:** [チュートリアル3、第11章：情報理論](../intro2/11_information_theory/#joint-and-conditional-entropy)

**関連項目:** [同時エントロピー](#joint-entropy-), [相互情報量](#mutual-information-)
{{% /expand %}}

### 相互情報量 📊
{{% expand "相互情報量" %}}
*相互情報量* $I(X; Y)$ は、一方の変数を知ることで他方についての不確実性がどれだけ減るか——**2つの変数が共有するビット数**——です。**対称的**であり、**$X$ と $Y$ が独立のときに限ってゼロ**になります。

**同値な公式:**
$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y).$$

**独立性:** $X \perp Y \iff I(X; Y) = 0$；条件付きでは $X \perp Y \mid Z \iff I(X; Y \mid Z) = 0$。

**図解:** $H(X)$ と $H(Y)$ を重なり合う円と見なすと、$I(X;Y)$ はその*重なり*です（同時エントロピーを和集合とした包除原理）。

**登場箇所:** [チュートリアル3、第11章：情報理論](../intro2/11_information_theory/#mutual-information)。**コライダー**への条件付けにより相互情報量が正に増加します——説明除去効果をビット単位で測定。

**関連項目:** [エントロピー](#entropy-), [条件付きエントロピー](#conditional-entropy-), [条件付き独立性（d分離）](#markov-equivalence-class-)
{{% /expand %}}

### クロスエントロピー 📊
{{% expand "クロスエントロピー" %}}
*クロスエントロピー* $H(P, Q)$ は、**現実が $P$ であるにもかかわらず $Q$ で予測したときの平均的な驚き**です——誤ったモデルを使ったときに実際に感じる驚きです。

**公式:** $H(P, Q) = -\sum_x P(x) \log_2 Q(x)$

**重要な等式:** $H(P, Q) = H(P) + D_{\text{KL}}(P \parallel Q)$ ——全体の驚き = 避けられない部分 $H(P)$ + 誤りのペナルティ。$H(P)$ は $Q$ に依存しないため、**クロスエントロピーを最小化することはKLダイバージェンスを最小化すること**と同値です——これが「クロスエントロピー損失」が分類器や言語モデルを学習させる理由です。

**注意:** $H(P, P) = H(P)$ ——完全なモデルのクロスエントロピーは単純なエントロピーに等しくなります。

**登場箇所:** [チュートリアル3、第11章：情報理論](../intro2/11_information_theory/#cross-entropy-and-kl-divergence)

**関連項目:** [エントロピー](#entropy-), [KLダイバージェンス](#kl-divergence-)
{{% /expand %}}

### KLダイバージェンス 📊
{{% expand "KLダイバージェンス" %}}
*カルバック・ライブラー・ダイバージェンス* $D_{\text{KL}}(P \parallel Q)$ は、モデル $Q$ が真の分布 $P$ からどれだけ離れているかを、**驚きの余分なビット数**——現実が $P$ であるにもかかわらず $Q$ を信じるコスト——で測定します。

**公式:** $D_{\text{KL}}(P \parallel Q) = \sum_x P(x) \log_2 \frac{P(x)}{Q(x)} = H(P, Q) - H(P)$

**性質:** $D_{\text{KL}}(P \parallel Q) \ge 0$ であり、$Q = P$ のときに限り等号成立（**ギブスの不等式**）——誤った分布は平均的な驚きを*増やす*ことしかできません。**対称ではない**（一般に $D_{\text{KL}}(P \parallel Q) \ne D_{\text{KL}}(Q \parallel P)$）ため、真の距離ではなく*ダイバージェンス*です。

**登場箇所:** [チュートリアル3、第11章：情報理論](../intro2/11_information_theory/#cross-entropy-and-kl-divergence)

**関連項目:** [クロスエントロピー](#cross-entropy-), [エントロピー](#entropy-)
{{% /expand %}}

### マルコフ性 📊
{{% expand "マルコフ性" %}}
過程が*マルコフ性*を持つとは、**未来が過去から独立で現在のみに依存する**ときを指します：$P(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = P(X_{t+1} \mid X_t)$。現在の状態 $X_t$ さえ知れば、過去の履歴全体は $X_{t+1}$ の予測に何も加えません。

**直感:** 現在は未来を予測するための過去の完全な要約です——歴史はその痕跡をすべて現在の状態に残しています。

**登場箇所:** [チュートリアル3、第13章：マルコフ連鎖](../intro2/13_markov_chains/#naming-it-the-markov-property)

**関連項目:** [マルコフ連鎖](#markov-chain-), [推移行列](#transition-matrix-)
{{% /expand %}}

### マルコフ連鎖 📊
{{% expand "マルコフ連鎖" %}}
*マルコフ連鎖*は、[マルコフ性](#markov-property-)を持つ状態の列 $X_0, X_1, X_2, \dots$ です：各状態は直前の状態にのみ依存します。有限個の状態上の連鎖は[推移行列](#transition-matrix-)で完全に記述されます。

**例:** Chibany が毎日とんかつかハンバーグを選ぶとき、今日の選択が昨日の選択にのみ依存する場合。

**登場箇所:** [チュートリアル3、第13章：マルコフ連鎖](../intro2/13_markov_chains/)（連鎖の仕組み）、[第14章](../intro2/14_random_walks_networks/)でのネットワーク上の[ランダムウォーク](#random-walk-)、[第15章](../intro2/15_memory_search/)での記憶モデル。

**関連項目:** [マルコフ性](#markov-property-), [定常分布](#stationary-distribution-), [ランダムウォーク](#random-walk-)
{{% /expand %}}

### 推移行列 📊
{{% expand "推移行列" %}}
マルコフ連鎖の*推移行列* $P$ は、すべての1ステップ確率を集めたものです：$P_{ij} = P(X_{t+1} = j \mid X_t = i)$（現在状態 $i$ にいるとき次の状態が $j$ になる確率）。各**行**は次の状態上の確率分布であり、その和は1になります——そのような行列は**行確率行列**と呼ばれます。

**重要な理由:** 行列は*サンプラー*でもあります——ランダム数の系列と組み合わせることで列全体を生成します——また分布に $P$ を掛けることで1単位時間分だけ進められます。

**登場箇所:** [チュートリアル3、第13章：マルコフ連鎖](../intro2/13_markov_chains/#two-views-of-the-same-chain)。[第14章](../intro2/14_random_walks_networks/#from-a-graph-to-a-transition-matrix)ではネットワークの[隣接行列](#adjacency-matrix-and-degree-)を行正規化することで構築されます。

**関連項目:** [マルコフ連鎖](#markov-chain-), [定常分布](#stationary-distribution-), [隣接行列と次数](#adjacency-matrix-and-degree-)
{{% /expand %}}

### 定常分布 📊
{{% expand "定常分布" %}}
マルコフ連鎖の*定常分布* $\pi$ は、連鎖が各状態に留まる長期的な時間の割合です——同値に、1ステップ後も変わらない唯一の分布：$\pi P = \pi$。現在の状態についての確信がすでに $\pi$ であれば、それは永遠に $\pi$ のまま保たれます。

**求め方:** [べき乗反復法](#power-iteration-)（ただ連鎖を実行する）、または**固有値1の $P$ の左固有ベクトル**（すべての行確率行列はこれを持つ）として求めます。無向ネットワーク上のランダムウォークでは $\pi_i \propto \deg(i)$ という単純な形になります。

**登場箇所:** [チュートリアル3、第13章：マルコフ連鎖](../intro2/13_markov_chains/#the-stationary-distribution)；次数の形と**PageRank**は[第14章](../intro2/14_random_walks_networks/#the-stationary-distribution-of-a-walk)。

**関連項目:** [べき乗反復法](#power-iteration-), [エルゴード性](#ergodicity-), [PageRank](#pagerank-)
{{% /expand %}}

### べき乗反復法 📊
{{% expand "べき乗反復法" %}}
*べき乗反復法*は、任意の分布 $\mathbf{v}$ から始めて[推移行列](#transition-matrix-)を繰り返し掛けることで連鎖の[定常分布](#stationary-distribution-)を求めます：$\mathbf{v}, \mathbf{v}P, \mathbf{v}P^2, \dots \to \pi$。この系列は出発点によらず（[エルゴード](#ergodicity-)連鎖において） $\pi$ に収束します。

**登場箇所:** [チュートリアル3、第13章：マルコフ連鎖](../intro2/13_markov_chains/#finding-π-just-run-it-power-iteration)

**関連項目:** [定常分布](#stationary-distribution-), [エルゴード性](#ergodicity-)
{{% /expand %}}

### エルゴード性 📊
{{% expand "エルゴード性" %}}
マルコフ連鎖が*エルゴード的*であるとは、（場合によっては複数ステップで）任意の状態から任意の状態に到達でき、固定したサイクルに閉じ込められないときを指します。エルゴード的な連鎖は**混合**します：すべての出発点から同じ[定常分布](#stationary-distribution-)に収束するため、$\pi$ は連鎖の性質であり出発点の性質ではありません。

**有用な事実:** どんな連鎖も、任意の状態にジャンプする小さな確率 $\varepsilon$ を加えることでエルゴード的にできます——[PageRank](#pagerank-)を well-defined にするトリック（その「テレポート」/ダンピング項）です。

**登場箇所:** [チュートリアル3、第13章：マルコフ連鎖](../intro2/13_markov_chains/#why-the-start-doesnt-matter-ergodicity)；ε トリックは[第14章](../intro2/14_random_walks_networks/#pagerank-the-same-π-at-web-scale)で再利用；MCMC の**混合**の「出発点を忘れる」基礎として[第18章](../intro2/18_markov_chain_monte_carlo/#mixing-burn-in-and-the-multimodal-trap)で再利用。

**関連項目:** [定常分布](#stationary-distribution-), [PageRank](#pagerank-), [混合](#mixing-), [バーンイン](#burn-in-)
{{% /expand %}}

### ランダムウォーク 📊
{{% expand "ランダムウォーク" %}}
ネットワーク上の*ランダムウォーク*は、状態がグラフの**ノード**である[マルコフ連鎖](#markov-chain-)です：各ステップでウォーカーは一様ランダムに隣接ノードに移動します。その[推移行列](#transition-matrix-)は、各行の和が1になるように正規化された[隣接行列](#adjacency-matrix-and-degree-)です。

**重要な結果:** 無向・無重みネットワーク上では、ウォークの[定常分布](#stationary-distribution-)は $\pi_i \propto \deg(i)$ です——接続が多いノードほど頻繁に訪問されます。

**登場箇所:** [チュートリアル3、第14章：ネットワーク上のランダムウォーク](../intro2/14_random_walks_networks/)；**検閲された**ランダムウォークが[第15章](../intro2/15_memory_search/)で記憶想起をモデル化します。

**関連項目:** [マルコフ連鎖](#markov-chain-), [隣接行列と次数](#adjacency-matrix-and-degree-), [PageRank](#pagerank-), [検閲関数](#censoring-function-)
{{% /expand %}}

### 隣接行列と次数 📊
{{% expand "隣接行列と次数" %}}
グラフ $G = (V, E)$ は**ノード** $V$ と**エッジ** $E$ で構成されます。その*隣接行列* $L$ はエッジを記録します：ノード $i$ と $j$ が接続されているとき $L_{ij} = 1$、そうでなければ $0$（無向グラフでは対称）。ノードの*次数* $\deg(i)$ は、そのノードに接するエッジの数——同値に $L$ の行の和——です。

**重要な理由:** $L$ を行正規化すると[ランダムウォーク](#random-walk-)の[推移行列](#transition-matrix-)が得られ、無向ウォークでは $\pi_i \propto \deg(i)$ ——次数が長期的な訪問頻度そのものになります。

**登場箇所:** [チュートリアル3、第14章：ネットワーク上のランダムウォーク](../intro2/14_random_walks_networks/#chibanys-animal-network)

**関連項目:** [ランダムウォーク](#random-walk-), [推移行列](#transition-matrix-)
{{% /expand %}}

### PageRank 📊
{{% expand "PageRank" %}}
*PageRank*——元のGoogle検索エンジンのアルゴリズム——は、有向グラフのノードを、そのグラフ上の[ランダムウォーク](#random-walk-)の[定常分布](#stationary-distribution-)によってランク付けします：リンクをたどり、小さな確率 $\varepsilon$ でランダムなノードにテレポートする（[エルゴード性](#ergodicity-)修正；Googleの*ダンピング係数*は $1 - \varepsilon$）「ランダムサーファー」です。重要なノードはランダムウォーカーが頻繁に訪問するものです。

**認知科学との接続:** Griffiths、Steyvers & Firl (2007) は、*意味的*ネットワーク上のPageRankが流暢性課題で人々が産出する単語を予測することを示しました。

**登場箇所:** [チュートリアル3、第14章：ネットワーク上のランダムウォーク](../intro2/14_random_walks_networks/#pagerank-the-same-π-at-web-scale)

**関連項目:** [定常分布](#stationary-distribution-), [エルゴード性](#ergodicity-), [意味ネットワーク](#semantic-network-)
{{% /expand %}}

### 意味ネットワーク 📊
{{% expand "意味ネットワーク" %}}
*意味ネットワーク*は知識をグラフとして表現します：**概念**がノードであり**連想**がエッジです（例：*dog*–*cat*）。このようなネットワークは語連想データから推定されることが多く——多くの人々にあるキュー単語について何を思い浮かべるかを尋ねます。

**重要な理由:** 意味記憶をネットワークとして扱うことで、単一の[ランダムウォーク](#random-walk-)によって人々の*想起*をモデル化できます——第15章の記憶探索モデルの基盤です。

**登場箇所:** [チュートリアル3、第14章：ネットワーク上のランダムウォーク](../intro2/14_random_walks_networks/#whats-a-graph)と[第15章：記憶探索](../intro2/15_memory_search/)。

**関連項目:** [ランダムウォーク](#random-walk-), [検閲関数](#censoring-function-), [PageRank](#pagerank-)
{{% /expand %}}

### 検閲関数 📊
{{% expand "検閲関数" %}}
記憶探索のランダムウォークモデル（Abbott、Austerweil & Griffiths 2012）では、*検閲関数*は潜在的なウォークを観測されたリストに写像します：ウォークが初めてそのノードに到達したとき、かつそれがターゲットカテゴリにある場合にのみ単語を**報告**します；再訪問とカテゴリ外のノードは*検閲*（非記録）されます。借用された統計用語は「起きたが記録されなかった」を意味します。

**結果:** 連続する*最初到達時刻* $\tau(k)$——ウォークが $k$ 番目の報告アイテムに初めて到達するとき——の間隔が**項目間反応時間** $\text{IRT}(k) = \tau(k) - \tau(k-1) + \text{単語の長さ}$ を駆動し、明示的なスイッチルールなしに人間の「スイッチコスト」曲線を再現します。

**登場箇所:** [チュートリアル3、第15章：記憶探索](../intro2/15_memory_search/#the-censoring-function)

**関連項目:** [ランダムウォーク](#random-walk-), [意味ネットワーク](#semantic-network-)
{{% /expand %}}

### 重要度重み 📊
{{% expand "重要度重み" %}}
関心のある**ターゲット** $p$ ではなく**提案分布** $q$ からサンプリングする場合、*重要度重み* $w(x) = p(x)/q(x)$ はそのミスマッチを補正し、$q$ の下での加重平均が $p$ の下での期待値を推定できるようにします：$\mathbb{E}_p[f] = \mathbb{E}_q[f \cdot w]$。

**良い重みと悪い重み:** ターゲットに似た提案分布は重みを1に近くします（均一、健全）；合わない提案分布は少数の巨大な重みと多数のゼロに近い重みを生成します（推定が不安定になります）。[有効サンプルサイズ](#effective-sample-size-)がこれを測定します。

**登場箇所:** [チュートリアル3、第16章：モンテカルロ](../intro2/16_monte_carlo/#importance-sampling-sample-the-wrong-distribution)

**関連項目:** [重点サンプリング](#importance-sampling-), [提案分布](#proposal-distribution-), [有効サンプルサイズ](#effective-sample-size-)
{{% /expand %}}

### 有効サンプルサイズ 📊
{{% expand "有効サンプルサイズ" %}}
正規化された重み $w_t$（合計が1）の集合に対して、*有効サンプルサイズ*は $N_{\text{eff}} = 1 / \sum_t w_t^2$ です。「$T$ 個の重み付きサンプルは等しく重み付きサンプル何個分の価値があるか？」という問いに答えます。

**2つの極限:** 完全に均一な重みでは $N_{\text{eff}} = T$（すべてのサンプルが重要）；1つの支配的な重みでは $N_{\text{eff}} \approx 1$。これは**提案分布 $q$ がターゲット $p$ にどれほどよく合っているかの診断**であり、推定値の精度の直接的な測定ではありません。

**登場箇所:** [チュートリアル3、第16章：モンテカルロ](../intro2/16_monte_carlo/#effective-sample-size)；逐次バージョン（重み退化）は[第17章：粒子フィルタリング](../intro2/17_particle_filtering/#resampling-and-degeneracy)。

**関連項目:** [重要度重み](#importance-weight-), [重点サンプリング](#importance-sampling-), [重み退化](#weight-degeneracy-), [リサンプリング](#resampling-)
{{% /expand %}}

### 棄却サンプリング 📊
{{% expand "棄却サンプリング" %}}
評価できるが直接サンプリングできないターゲット密度 $p$ からサンプリングする方法：$p$ 上に簡単な**エンベロープ**を置き、エンベロープの下に点を一様に投げ、**$p$ の下に落ちたものだけを保持**します。生き残ったものは $p$ からの厳密なサンプルです。

**トレードオフ:** エンベロープが $p$ の面積よりはるかに大きければ、ほとんどの提案が棄却されます——無駄な仕事。その非効率を[重点サンプリング](#importance-sampling-)は棄却の代わりに*重み付け*することで回避します。

**登場箇所:** [チュートリアル3、第16章：モンテカルロ](../intro2/16_monte_carlo/#when-you-cant-sample-p-rejection-and-inverse-cdf)

**関連項目:** [重点サンプリング](#importance-sampling-), [モンテカルロシミュレーション](#monte-carlo-simulation-)
{{% /expand %}}

### 提案分布 📊
{{% expand "提案分布" %}}
重点サンプリングとMCMCにおいて、*提案分布* $q$ は実際にサンプリングする分布です——通常サンプリングが容易なもの——難しい**ターゲット**の代用として使います。重点サンプリングでは[重要度重み](#importance-weight-) $p/q$ で交換を補正し；[メトロポリス–ヘイスティングス](#metropolishastings-)では提案分布が候補となる次の状態を生成し、その後受け入れまたは棄却されます。

**適切な選択:** ターゲットに近い提案分布は重みを均一に保ち（重点サンプリング）または混合を速めます（MCMC）；悪い提案分布は両方を台無しにします。

**登場箇所:** [チュートリアル3、第16章：モンテカルロ](../intro2/16_monte_carlo/#importance-sampling-sample-the-wrong-distribution)と[第18章：マルコフ連鎖モンテカルロ](../intro2/18_markov_chain_monte_carlo/#metropolishastings)。

**関連項目:** [重要度重み](#importance-weight-), [メトロポリス–ヘイスティングス](#metropolishastings-), [受け入れ比](#acceptance-ratio-)
{{% /expand %}}

### 粒子フィルタ 📊
{{% expand "粒子フィルタ" %}}
時間とともに変化する隠れた状態についての**逐次的な**推論手法。重み付きサンプル（*粒子*）の群れで事後分布を表現し、新しいデータが到着するたびに**重み付け → リサンプリング → 伝播**のループで更新します——リサンプリングステップを伴う*逐次重点サンプリング*。指針となるアイデア：*昨日の事後分布が今日の事前分布です。*

**過程モデルとして:** 少数の粒子フィルタ——少数の推測を左から右へ更新したもの——は、人間の限られた記憶、順序効果、試行間の変動性を予測します。

**登場箇所:** [チュートリアル3、第17章：粒子フィルタリング](../intro2/17_particle_filtering/)

**関連項目:** [重点サンプリング](#importance-sampling-), [リサンプリング](#resampling-), [有効サンプルサイズ](#effective-sample-size-)
{{% /expand %}}

### リサンプリング 📊
{{% expand "リサンプリング" %}}
[粒子フィルタ](#particle-filter-)において、*リサンプリング*は現在の粒子から**重みに比例した確率で**（インデックスのカテゴリカル抽出によって）新しい粒子集合を抽出します：重い粒子は複製され、軽い粒子は削除され、すべての重みが均一にリセットされます。

**必要な理由:** これなしには重みが時間とともに積み重なって1つの粒子がすべてを担うようになります——[有効サンプルサイズ](#effective-sample-size-)の崩壊で測られる*重み退化*。リサンプリングは群れを重要な場所に集中させ、フィルタを無期限に有用に保ちます。

**登場箇所:** [チュートリアル3、第17章：粒子フィルタリング](../intro2/17_particle_filtering/#sequential-importance-sampling)

**関連項目:** [粒子フィルタ](#particle-filter-), [重み退化](#weight-degeneracy-), [有効サンプルサイズ](#effective-sample-size-)
{{% /expand %}}

### マルコフ連鎖モンテカルロ（MCMC） 📊
{{% expand "マルコフ連鎖モンテカルロ（MCMC）" %}}
ターゲット分布 $\pi$（典型的にはサンプリングが困難なベイズ事後分布）から、**[定常分布](#stationary-distribution-)が正確に $\pi$ になるようなマルコフ連鎖を設計**し、それを実行して訪問した状態を収集することでサンプリングする手法の族。[第13章](../intro2/13_markov_chains/)の論理を逆に走らせます：連鎖が与えられて $\pi$ を求めるのではなく、求めたい $\pi$ から出発して連鎖を構築します。

**主力手法:** [メトロポリス–ヘイスティングス](#metropolishastings-)（提案＋受け入れ/棄却）と[ギブスサンプリング](#gibbs-sampling-)（条件付き分布から1つの座標をリサンプリング）。

**登場箇所:** [チュートリアル3、第18章：マルコフ連鎖モンテカルロ](../intro2/18_markov_chain_monte_carlo/)と[第19章：心のサンプリング](../intro2/19_sampling_the_mind/)。

**関連項目:** [メトロポリス–ヘイスティングス](#metropolishastings-), [ギブスサンプリング](#gibbs-sampling-), [定常分布](#stationary-distribution-), [バーンイン](#burn-in-), [混合](#mixing-)
{{% /expand %}}

### メトロポリス–ヘイスティングス 📊
{{% expand "メトロポリス–ヘイスティングス" %}}
最も一般的なMCMCのレシピ。現在の状態 $x$ から：[提案分布](#proposal-distribution-)から候補 $x'$ を**提案**し、[受け入れ比](#acceptance-ratio-) $A = \min(1, P(x')/P(x))$（対称な提案の場合）で与えられる確率でそれを受け入れます；そうでなければ $x$ に留まります。

**なぜ機能するか:** このルールは $P$ に対して*詳細つり合い*を強制するため、$P$ が連鎖の定常分布になります。**比率** $P(x')/P(x)$ のみが現れるため、正規化定数がキャンセルされます——*非正規化*事後分布からサンプリングできます。

**登場箇所:** [チュートリアル3、第18章：マルコフ連鎖モンテカルロ](../intro2/18_markov_chain_monte_carlo/#metropolishastings)と[第19章：心のサンプリング](../intro2/19_sampling_the_mind/)。

**関連項目:** [マルコフ連鎖モンテカルロ（MCMC）](#markov-chain-monte-carlo-mcmc-), [受け入れ比](#acceptance-ratio-), [提案分布](#proposal-distribution-), [ギブスサンプリング](#gibbs-sampling-)
{{% /expand %}}

### 受け入れ比 📊
{{% expand "受け入れ比" %}}
[メトロポリス–ヘイスティングス](#metropolishastings-)において、現在の状態 $x$ から提案状態 $x'$ に移動する確率：対称な提案の場合 $A = \min\left(1, \frac{P(x')}{P(x)}\right)$。**上り坂**の移動（$P(x') > P(x)$）は常に受け入れられます；**下り坂**の移動はその相対的な高さに比例して受け入れられます。

**重要な特徴:** ターゲット確率の*比率*のみが重要なため、あらゆる正規化定数がキャンセルされます——これがMCMCがエビデンス $p(\text{data})$ まで知らない事後分布に対して機能する理由です。

**登場箇所:** [チュートリアル3、第18章：マルコフ連鎖モンテカルロ](../intro2/18_markov_chain_monte_carlo/#metropolishastings)

**関連項目:** [メトロポリス–ヘイスティングス](#metropolishastings-), [提案分布](#proposal-distribution-)
{{% /expand %}}

### ギブスサンプリング 📊
{{% expand "ギブスサンプリング" %}}
**1つの座標ずつ**更新するMCMC手法で、各座標をその完全条件付き分布 $P(x_i \mid x_{-i})$（他のすべての座標の現在値が与えられたときの $x_i$ の分布）から正確にサンプリングします。**棄却なし**——真の条件付き分布からサンプリングすることで自動的に詳細つり合いが満たされます——ただしそれらの条件付き分布が利用可能である必要があり、モデルが**[共役](#conjugate-prior-)**な部品から構築されているときに当てはまります。

**登場箇所:** [チュートリアル3、第18章：マルコフ連鎖モンテカルロ](../intro2/18_markov_chain_monte_carlo/#gibbs-sampling)と[第19章：心のサンプリング](../intro2/19_sampling_the_mind/#step-1--gibbs-the-θᵢ-conjugate)。

**関連項目:** [マルコフ連鎖モンテカルロ（MCMC）](#markov-chain-monte-carlo-mcmc-), [メトロポリス–ヘイスティングス](#metropolishastings-), [共役事前分布](#conjugate-prior-)
{{% /expand %}}

### バーンイン 📊
{{% expand "バーンイン" %}}
MCMCの実行の最初の部分で、サンプルを収集する前に*破棄*されます。それらの初期状態は任意の出発点を反映しており、ターゲット分布を反映していません；連鎖が**混合**（出発点を忘れる）した後、残りの状態がターゲットを近似し、保持されます。

**登場箇所:** [チュートリアル3、第18章：マルコフ連鎖モンテカルロ](../intro2/18_markov_chain_monte_carlo/#mixing-burn-in-and-the-multimodal-trap)

**関連項目:** [混合](#mixing-), [マルコフ連鎖モンテカルロ（MCMC）](#markov-chain-monte-carlo-mcmc-), [エルゴード性](#ergodicity-)
{{% /expand %}}

### 混合 📊
{{% expand "混合" %}}
MCMCの連鎖が*混合した*とは、出発点を忘れてターゲット分布全体を探索しているときを指します——[エルゴード性](#ergodicity-)と同じ「出発点を忘れる」性質。十分に混合した連鎖は、異なる出発点から実行しても同じ答えを与えます。

**罠:** **多峰性**のターゲット上では、連鎖が完全に健全な局所的な受け入れ率を持ちながらも1つのモードに留まり続け、峰の間の低確率の谷を越えられないことがあります。*良い局所的な受け入れは良いグローバルな混合を意味しません*——これが異なる出発点から複数の連鎖を実行して一致するか確認することが重要な理由です。

**登場箇所:** [チュートリアル3、第18章：マルコフ連鎖モンテカルロ](../intro2/18_markov_chain_monte_carlo/#mixing-burn-in-and-the-multimodal-trap)

**関連項目:** [バーンイン](#burn-in-), [エルゴード性](#ergodicity-), [マルコフ連鎖モンテカルロ（MCMC）](#markov-chain-monte-carlo-mcmc-)
{{% /expand %}}

### 人間によるMCMC 📊
{{% expand "人間によるMCMC" %}}
メトロポリスサンプラーの受け入れステップとして*人*を扱う手法（Sanborn & Griffiths、2007）：2つの選択肢を見せ、どちらかを選ばせ、繰り返す。人が自分の事後分布に比例して受け入れるならば、選択の連鎖はその事後分布に収束します——そしてデータなしには、それは**頭の中の事前分布**に収束します。アニメの動物に対して実行すると、人々の精神的なカテゴリのプロトタイプを復元します。

**登場箇所:** [チュートリアル3、第19章：心のサンプリング](../intro2/19_sampling_the_mind/#when-a-person-is-the-accept-step)

**関連項目:** [マルコフ連鎖モンテカルロ（MCMC）](#markov-chain-monte-carlo-mcmc-), [メトロポリス–ヘイスティングス](#metropolishastings-), [事前分布](#prior-distribution-)
{{% /expand %}}

### 偽信念課題 📊
{{% expand "偽信念課題" %}}
心の理論を測る代表的なテスト：自分が**誤り**だと知っている信念を表現できるか？ 古典的な**サリー・アン課題**（Baron-Cohen, Leslie & Frith 1985）では、サリーがビー玉をかごに隠して部屋を出る。留守の間にアンがそれを箱へ移す。そして「サリーはどこを探すか？」と尋ねる。正解するには、現実（「箱」）とは**異なる**サリーの信念（「かご」）を保持しなければならず、子どもが安定して通過するのは**4歳ごろ**である。形式的にはこれは**POMDP**である：ビー玉の真の位置は隠れた世界状態であり、サリーの信念は真実から乖離しうる*別個の*潜在変数である。

**登場箇所:** [チュートリアル3、第23章：逆強化学習](../intro2/23_inverse_rl_goal_inference/), [第24章：POMDP](../intro2/24_pomdps_belief_inference/)

**関連項目:** [心の理論](#theory-of-mind-), [失言課題](#faux-pas-test-), [信念状態](#belief-state-)
{{% /expand %}}

### 失言課題 📊
{{% expand "失言課題" %}}
より難しい心の理論のテスト：社会的な失言（フォーパ）を見抜くには、**入れ子になった**二次の偽信念が必要である——話し手の偽信念（それが秘密だと知らなかった）に加えて、その発言がどう受け取られるかについての聞き手の知識である。ある心的状態を別の心的状態の中に積み重ねるため、基本的な偽信念課題よりずっと後、**9〜11歳ごろ**に通過する。この入れ子構造こそ、[第25章](../intro2/25_modern_rl_world_models/)で大規模言語モデルの心の理論テストが調べているものである。

**登場箇所:** [チュートリアル3、第23章：逆強化学習](../intro2/23_inverse_rl_goal_inference/)

**関連項目:** [偽信念課題](#false-belief-task-), [心の理論](#theory-of-mind-)
{{% /expand %}}

### コミュニケーション的実演 📊
{{% expand "コミュニケーション的実演" %}}
教示のための**信念志向プランニング**（Ho, Cushman, Littman & Austerweil 2021）：**レベル0**の観察者はあなたの行動を反転してゴールに関する事後分布を求める（逆計画）。あなた（**レベル1**の実演者）は、その事後分布を真実へ向かわせるように行動を選ぶ。観察者の信念はあなたには**隠れている**ため、実演を計画すること自体が、**隠れ状態が観察者の信念であるPOMDP**になる——すなわち教示とは、逆計画を一段上で実行することである。これは読みやすさ（legibility）効果を*予測*する：読みやすい経路は最初の一手で観察者を$0.61$へ引き上げ、単に効率的な経路の$0.50$を上回る。

**登場箇所:** [チュートリアル3、第24章：POMDP](../intro2/24_pomdps_belief_inference/)

**関連項目:** [読みやすさ](#legibility-), [協調的逆強化学習](#cooperative-inverse-rl-), [部分観測マルコフ決定過程（POMDP）](#partially-observable-mdp-pomdp-)
{{% /expand %}}

---

## ナビゲーション

**チュートリアル別**:
- [チュートリアル1：離散確率](../intro/) - 📘 タグ付き用語
- [チュートリアル2：GenJAXプログラミング](../genjax/) - 💻 タグ付き用語
- [チュートリアル3：連続確率](../intro2/) - 📊 タグ付き用語

**トピック別**:
- **確率の基礎**: 集合, 結果空間, 事象, 確率, 条件付き確率
- **プログラミング**: @gen, トレース, ChoiceMap, simulate(), importance(), vmap
- **分布**: ベルヌーイ分布, カテゴリカル分布, 正規/ガウス分布, ベータ分布, 一様分布
- **ベイズ学習**: 事前分布, 尤度, 事後分布, 予測分布
- **発展モデル**: GMM, DPMM, ディリクレ過程, スティック破砕法

---

*この用語集はチュートリアルとともに成長するよう設計されています。用語が欠けていたらお知らせください！*
