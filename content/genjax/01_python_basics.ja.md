+++
title = "GenJAXのためのPython基礎"
weight = 2
+++

## プログラマーになる必要はありません！

この章では、GenJAXのコードを読んで実行するために**必要最低限のPython**を学びます。ソフトウェア開発者になる必要はありませんが、次のことができるようになります：
- コードが何をしているかを理解する
- 値を変えて実験する
- 例を実行して結果を確認する

レストランで注文できる程度のイタリア語を学ぶようなものです — 流暢に話せなくても、実用的な知識があれば十分です！

---

## 1. 変数：物に名前をつける

Pythonでは、後で使えるように値に名前をつけます。

```python
probability_hamburger = 0.5
probability_tonkatsu = 0.5
```

**意味：**「これらの数値を記憶して、この名前で呼ぶ」

**確率との対応：** 数学で $P(H) = 0.5$ と書いたように、その値を保存しています。

**試してみよう：**
```python
x = 10
y = 20
result = x + y
print(result)  # Shows: 30
```

{{% notice style="info" title="`#`記号について" %}}
`#` の後に続くものはすべて**コメント**です — 人間のためのメモで、コンピュータには無視されます。

```python
# This is a comment
x = 5  # Comments can go after code too
```
{{% /notice %}}

---

## 2. 関数：アクションのレシピ

**関数**とは、命令に名前をつけたものです。レシピのようなものだと思ってください。

```python
def greet_chibany():
    print("Hello, Chibany!")
    print("Time for tonkatsu!")

greet_chibany()  # "Calls" the function (runs the recipe)
```

**出力：**
```
Hello, Chibany!
Time for tonkatsu!
```

### 入力のある関数（パラメータ）

関数は**入力**（パラメータと呼ばれる）を受け取れます：

```python
def greet_cat(name):
    print(f"Hello, {name}!")

greet_cat("Chibany")  # Output: Hello, Chibany!
greet_cat("Felix")    # Output: Hello, Felix!
```

**文字列の前の `f`** によって、テキストの中の `{}` 内に変数を埋め込めます。

### 出力のある関数（戻り値）

関数は値を**返す**ことができます：

```python
def add_numbers(a, b):
    result = a + b
    return result

total = add_numbers(5, 3)  # total is now 8
```

**確率との対応：** $f(\omega)$ が結果を受け取って数値を返す関数であることを思い出してください。同じ考え方です！

---

## 3. リスト：物のコレクション

**リスト**は買い物リストのようなもの — 順序のあるアイテムの集まりです。

```python
meals = ["HH", "HT", "TH", "TT"]
```

**集合との対応：** これは $\Omega = \{HH, HT, TH, TT\}$ に順序がついたものです！

### アイテムへのアクセス

```python
meals = ["HH", "HT", "TH", "TT"]

first_meal = meals[0]   # "HH" (Python counts from 0!)
second_meal = meals[1]  # "HT"
```

{{% notice style="warning" title="Pythonはゼロから数える！" %}}
最初のアイテムは `[0]`、2番目は `[1]`、というようになります。

最初は誰でもつまずきます。Pythonは少し独特だと覚えておいてください！
{{% /notice %}}

### アイテムの数は？

```python
meals = ["HH", "HT", "TH", "TT"]
count = len(meals)  # 4
```

**対応：** これは $|\Omega|$（濃度）のようなものです！

---

## 4. ループ：繰り返し実行する

**forループ**はアクションを繰り返します：

```python
for meal in ["HH", "HT", "TH", "TT"]:
    print(meal)
```

**出力：**
```
HH
HT
TH
TT
```

**読み方：**「このリストの各mealについて、mealを表示する」

### カウントループ

```python
for i in range(5):
    print(f"Day {i}")
```

**出力：**
```
Day 0
Day 1
Day 2
Day 3
Day 4
```

**対応：** 10,000日をシミュレートしたい場合は `range(10000)` を使えばよいです！

---

## 5. 条件分岐：判断を下す

**if文**を使うとコードが選択を行えます：

```python
meal = "TT"

if "T" in meal:
    print("Contains tonkatsu!")
else:
    print("No tonkatsu today :(")
```

**読み方：**「TがmealにあればこちらをThenする。そうでなければあちらをする。」

### 複数の条件

```python
tonkatsu_count = 2

if tonkatsu_count == 2:
    print("Two tonkatsus!")
elif tonkatsu_count == 1:
    print("One tonkatsu!")
else:
    print("No tonkatsu!")
```

**注意：**
- `==` は「等しい」（比較）を意味します
- `=` は「代入」（値を与える）を意味します

---

## 6. デコレータ：特別な機能を追加する

**デコレータ**は関数に機能を追加します。GenJAXでは `@gen` を使います：

<!-- validate: skip -->
```python
@gen
def my_function():
    pass  # Placeholder - your code goes here
```

**`@gen` が行うこと：** GenJAXに「これは生成的関数です — すべての確率的選択を追跡してください！」と伝えます。

**デコレータを完全に理解する必要はありません。** 次のことだけ知っておいてください：
- 関数定義のすぐ前に置く
- 関数の動作を変更する
- GenJAXでは `@gen` は確率モデルに不可欠

---

## 7. GenJAXの`@`記号（アドレッシング）

GenJAXでは `@` を使って**確率的選択に名前をつける**：

<!-- validate: skip -->
```python
lunch = flip(0.5) @ "lunch"
```

**読み方：**「表（true/1/とんかつ）が50%の確率のコインを投げて、この選択を'lunch'と呼ぶ」

**ベルヌーイ確率変数とは？** ベルヌーイ確率変数は、コイン投げのような単一のyes/no結果を表します。0（失敗/false/裏/とんかつ）または1（成功/true/表/ハンバーガー）のいずれかで、1になる確率は $p$ です。GenJAXでは `flip(p)` を使ってベルヌーイ分布からサンプリングします — コイン投げのメタファーにちなんだ名前です！

**確率との対応：** これは「$L$ を昼食の確率変数とし、$L$ は $p=0.5$ のベルヌーイ分布に従う」と言うことに相当します。

---

## 8. ライブラリとインポート

ライブラリは、私たちが使えるあらかじめ書かれたコードの集まりです：

```python
import jax
import matplotlib.pyplot as plt
from genjax import gen, flip
```

**意味：**
- `import jax` — JAXライブラリをロードする（計算のため）
- `import matplotlib.pyplot as plt` — プロットツールをロードし、`plt` と呼ぶ
- `from genjax import gen, flip` — GenJAXから特定のツールをロードする

**これらを暗記する必要はありません。** 各ノートブックの最初にインポートセルを実行するだけです！

---

## 9. ドット記法でメソッドを呼び出す

オブジェクト「に対して」関数を呼び出すこともあります：

<!-- validate: skip -->
```python
trace = model.simulate(key, args)
choices = trace.get_choices()
```

**読み方：**「modelに属するsimulate関数を呼び出す」

`.` は「～に属する」または「～の一部」を意味します。

---

## 10. コメントとドキュメント

### 一行コメント

```python
# This is a comment
x = 5  # This is also a comment
```

### 複数行コメント（docstring）

```python
def my_function():
    """
    This is a docstring.
    It explains what the function does.
    """
    # ... code ...
```

**重要な理由：** コードが何をしているかを理解するのに役立ちます！

---

## クイックリファレンス：GenJAXコードを読む

典型的なGenJAX関数を分解して説明します：

```python
@gen                                    # Decorator: makes this a generative function
def chibany_meals():                    # Function name
    """Generate one day of meals."""   # Docstring: what it does

    # Random choice: lunch
    lunch = flip(0.5) @ "lunch"         # @ names the choice

    # Random choice: dinner
    dinner = flip(0.5) @ "dinner"       # Another named choice

    # Return both meals as a pair
    return (lunch, dinner)              # Return value
```

**GenJAXコードを読むには：**
1. `@gen` を見つける — それは生成的関数
2. docstringを読む — 何をするのか？
3. `@` 記号を探す — それらが確率的選択
4. 何を返すかを確認する — それが結果

---

## 練習：このコードが読めますか？

```python
@gen
def coin_flips(n):
    """Flip a fair coin n times."""
    heads_count = 0

    for i in range(n):
        coin = flip(0.5) @ f"flip_{i}"
        if coin == 1:
            heads_count = heads_count + 1

    return heads_count
```

{{% expand "このコードは何をするでしょうか？" %}}
**行ごとの解説：**
1. `@gen` — これは生成的関数
2. `def coin_flips(n):` — 数値 `n` を入力として受け取る
3. `heads_count = 0` — 0からカウントを開始
4. `for i in range(n):` — n回繰り返す
5. `coin = flip(0.5) @ f"flip_{i}"` — 公正なコインを投げる（p=0.5のベルヌーイ）、"flip_0"、"flip_1"などと名付ける
6. `if coin == 1:` — 表（1）であれば
7. `heads_count = heads_count + 1` — カウントに1を加える
8. `return heads_count` — 表が出た回数を返す

**機能：** コインをn回投げて表の回数を数えます！

**対応：** これは確率論の二項分布に相当します。
{{% /expand %}}

---

## 学ばなくてよいこと

**心配しなくてよいこと：**
- ❌ オブジェクト指向プログラミング
- ❌ 高度なデータ構造
- ❌ ファイル入出力
- ❌ エラー処理
- ❌ Pythonのほとんどの機能！

**集中すること：**
- ✅ コードを読んで何をするかを理解する
- ✅ ノートブックのコードセルを実行する
- ✅ パラメータの値を変えて実験する
- ✅ 確率との対応を理解する

---

## 成功のためのヒント

### 1. 暗記する必要はありません

この章を参照用に開いておいてください。GenJAXコードの中で認識できないものを見かけたら、ここに戻ってきてください！

### 2. コードを実行して理解する

ただ読むだけでなく、コードを**実行**してください！出力を見ることで、すべてがより明確になります。

### 3. 実験しましょう！

値を変えてみてください：
- `0.5` を `0.8` に変えると何が起こりますか？
- シミュレーション回数を変えたら？
- わざと壊してどんなエラーが出るか確かめてみてください！

### 4.「これは何をしているのか？」と問いかけよう

「これはどのように動くのか？」ではなく、「これは何を達成しようとしているのか？」

---

## GenJAXの準備完了！

これでPythonの知識が十分身につきました：
- ✅ GenJAXコードを読む
- ✅ 生成的関数が何をするかを理解する
- ✅ Colabで例を実行する
- ✅ 値を変えて実験する

**次は：** 最初の生成的関数を書いてみましょう！

---

|[← 前の章：はじめに](./00_getting_started.md) | [次の章：最初のGenJAXモデル →](./02_first_model.md)|
| :--- | ---: |
