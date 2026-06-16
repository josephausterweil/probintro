+++
date = "2026-06-16"
title = "統計的決定理論：信念から行動へ"
weight = 20
+++

## 信念から行動へ

19章にわたって、私たちは一つの問いを、十数通りの姿に変えながら何度も問い続けてきた——*見たものを踏まえて、何を信じるべきか？* 私たちは事前分布を構築し、データを尤度に変え、事後分布を読み取ってきた——[ベイズ学習](../04_bayesian_learning/)の機構の全てだ。しかし信念はまだ選択ではない。どこかの時点で、Chibanyは電卓を置いて**弁当を食べ**なければならない。

> **Jamal：**「よし、弁当はたぶん新鮮だって計算したわけだ。たぶん、ね。で——食べるの？」
>
> **Chibany：**「90%新鮮。たぶん……うん？」
>
> **Alyssa：**「でも、何を天秤にかけてるのか*考えて*みて。大丈夫なら、お昼が食べられる。そうじゃないなら、食中毒になる。『たぶん新鮮』は何をすべきかを教えてくれない——それぞれの間違いが*いくらかかるか*を比べる必要があるの。」

Alyssaは本章が埋めるギャップに名前を付けた。事後分布は世界についての分布であり、**行動**は一つのコミットされた一手だ。両者をつなぐ橋が**統計的決定理論**——間違いのコストがわかったとき、信念をどう決定に変えるかについての規範的な説明だ。これはコース全体の蝶番である：今日までの全ては*何を信じるべきか？*に答えてきた；今日からは、問いは*何をすべきか？*になる。

本章はワンショット版だ——一度だけ下す、ただ一つの決定。次の章ではこれを時間にわたって引き伸ばし、一つの行動が次の行動につながり、コストが積み重なっていく。

---

## 決定問題

あらゆる決定問題は同じ四つの要素を持つ。進みながら、それぞれとその記号に名前を付けていこう。

- **世界の状態** $\theta$ ——あなたが知らないもの（弁当は新鮮か、傷んでいるか？）。これはまさに、ずっと事後分布を当ててきた未知の量だ。
- **観測** $x$ ——行動する前に見ることのできるデータ（一嗅ぎ、賞味期限）；第1〜7週は $x$ を事後分布 $p(\theta \mid x)$ に変えてきた。記法を安定させるために $x$ を*単一の*観測として保つが、それがまとまった一群であっても以下では何も変わらない——単にそのすべてで条件付けすればよい、$p(\theta \mid x_1, \dots, x_n)$。
- **行動** $a$ ——利用可能な行動の集合 $A$ から引かれる（食べる、または堆肥にする）。
- **損失** $L(\theta, a)$ ——世界が本当は $\theta$ だったときに行動 $a$ を取ったことをどれだけ後悔するか。低い損失が良い（損失は報酬の鏡像であり、次章で報酬に出会う）。

**決定規則** $d(x)$ は戦略だ：それは可能な各観測を、あなたが取る行動へと写像する。流れは左から右に進む——世界は隠れていて、あなたは手がかりを見て、あなたの規則が行動を選び、世界の真の状態がその行動のコストを決める：

```mermaid
graph LR
    T["hidden state θ"] -.clue.-> X["observation x"]
    X --> D["decision rule d(x)"]
    D --> A["action a"]
    T --> L["loss L(θ, a)"]
    A --> L
    classDef node fill:none,stroke:#9bbcff,stroke-width:2px,color:#fff
    class T,X,D,A,L node
    linkStyle default stroke:#9bbcff,stroke-width:2px,color:#fff
```

弁当を具体的にしよう。世界は $\theta \in \{\text{新鮮}, \text{傷んでいる}\}$；行動は $\{\text{食べる}, \text{堆肥にする}\}$。損失は各組み合わせのコストを表す：

| | 食べる | 堆肥にする |
|---|---|---|
| **新鮮** ($\theta=0$) | $0$ ——お昼！ | $3$ ——良い弁当を無駄にした |
| **傷んでいる** ($\theta=1$) | $10$ ——食中毒 | $1$ ——軽い無駄だが、安全 |

新鮮な弁当を食べるのは完璧（$0$）；傷んだものを食べるのは大惨事（$10$）；堆肥にするのはどちらでも少し無駄になる。これらの数字はある価値判断——*食中毒は無駄よりずっと悪い*——を符号化しており、決定理論はその判断を入力として受け取る。それは何を大切にすべきかは教えてくれない；何を大切にするかを言ったあとで、何を**すべきか**を教えてくれる。

表は*結果*を固定する；**観測**こそが、一度きりの推測を*規則*に変えるものだ。Chibanyが得る唯一の手がかりが、**弁当が何日前に買われたか**、$x$ ——一つの数だとしよう。決定規則 $d(x)$ は、その数から行動への任意の方策であり、自然なのは、ある閾値 $k$ における**閾値**だ：

$$d_k(x) = \begin{cases} \text{食べる} & \text{if } x \le k, \\ \text{堆肥にする} & \text{if } x > k. \end{cases}$$

閾値 $k$ *こそが*規則だ：慎重な $k = 1$ は安全を取って良い弁当を堆肥にし（しばしば $3$ を払う）、無謀な $k = 6$ は傷んだものを食べる（$10$ を冒す）。どの $k$ が正しいのか？ 日が経つほど弁当は傷みやすくなる——だから答えは**日数 $\to$ P(新鮮)** の変換にかかっており、それは下の *GenJAXで* セクションで正確にし、GenJAXに解かせる。まずは規則を*採点する*方法が必要であり、それがまさに**リスク**だ。

---

## リスク：あなたが期待する損失

行動するとき、あなたは決して $\theta$ を知らないので、$L(\theta, a)$ を直接最小化することはできない——それは隠れているまさにそのものに依存するからだ。あなたが*できる*のは、**期待する**損失を最小化することだ。決定規則の**リスク**は、その平均損失である、

$$R(\theta, d) = \mathbb{E}_x\big[\, L(\theta, d(x)) \,\big],$$

固定された真の状態 $\theta$ に対して、データ $x$ が変化するときの期待損失だ。（この $\mathbb{E}$ は[第1章](../01_mystery_bentos/)と[第16章](../16_monte_carlo/)の期待値——確率で重み付けした平均だ。）リスクは*規則の成績表*である：世界が $\theta$ のとき、$d$ は平均してどれだけ悪くやるか？

{{% notice style="info" title="記法：E の添字を読む" %}}
期待値の添字は、**どの変数について平均を取っているか**を示す。$\mathbb{E}_x$ はデータについて平均する；$\mathbb{E}_\theta$ は**事前分布**（データを見る*前*のあなたの信念）について；$\mathbb{E}_{\theta \mid x}$ は**事後分布**（データを見た*後*のあなたの信念）について。下のベイズ対ミニマックスの分かれ目は、まるごと $\mathbb{E}$ に*どの*添字が乗っているかにかかっているので、ここはゆっくり進む価値がある。そして $\arg\min_a f(a)$ は「$f(a)$ を最小にする行動 $a$」を意味する——最小値ではなく、その最小の*位置*だ。
{{% /notice %}}

しかし $\theta$ 自体が未知なので、一つの規則はリスクの*曲線*まるごとを持つ——可能な $\theta$ ごとに一つの $R$ の値。（$\theta$ が連続なら——たとえば弁当が新鮮か傷んでいるかではなく*重さ*なら——その曲線は文字通りのものになる；二値の弁当ではそれはただの2点、一つは新鮮、一つは傷んでいる場合だ。）これは、二つの有名な答えを持つ正真正銘の問いを残す：リスクの曲線を、最小化すべき一つの数にどう潰すか？

---

## 最適であるための二つの道：ベイズ対ミニマックス

最初の答えはあなたの**信念**を使う。世界のある状態が他より起こりやすいと思うなら、その信念で損失を重み付けし、その*平均*を最小化する。**ベイズ規則**は事前期待リスクを最小化する、

$$d_{\text{Bayes}} = \arg\min_d\; \mathbb{E}_\theta\big[\, R(\theta, d) \,\big],$$

そしてひとたび実際に手がかり $x$ を見たなら、これはコースの残りの間あなたが取る一手と同値になる：**事後期待損失を最小化する行動を選べ**、

$$d_{\text{Bayes}}(x) = \arg\min_a\; \mathbb{E}_{\theta \mid x}\big[\, L(\theta, a) \,\big].$$

添字を読もう：私たちは今 $x$ を固定した（一嗅ぎを済ませた）ので、残る唯一の不確実性は $\theta$ についてだけであり、損失を**事後分布** $p(\theta \mid x)$ に対して平均する——リスクの式がやったように $x$ について平均するのでは*ない*。あなたは事後分布を持っている；損失をそれに対して平均する；最も安い行動を取る。（規則レベルの $\arg\min_d$ とこの行動レベルの $\arg\min_a$ が一致するのはちょっとした定理だ；私たちの目的には行動版だけがあれば十分だ。）

二つ目の答えは信念をまったく信用することを拒む。**ミニマックス規則**は*最悪のケース*を最小化する——それは、起こりうる最高のリスクをできるだけ低く抑える規則を選ぶ：

$$d_{\text{minimax}} = \arg\min_d\; \max_\theta\; R(\theta, d).$$

ミニマックスは悲観主義者の基準だ：世界は敵対的だと仮定し、それができる最悪に対して身を守る。（それは $\theta$ にわたってリスクを*平坦にする*ように作られている——「イコライザー」規則——だから下の図では平らな線として現れる。）二つの基準は本当に食い違うことがある。

最初の例題では、わざと日数を取り除く——Chibanyは**観測なし**でただ一つの弁当にコミットしなければならない——なので決定規則 $d$ は一つの行動 $a$ に潰れ、リスクは損失そのものに潰れる、$R(\theta, a) = L(\theta, a)$。ベイズはその損失を信念にわたって平均する（$\mathbb{E}_\theta$）；ミニマックスは $\theta$ にわたるその最悪値を取る（各行動の列を下に見たときの最悪の項目）。弁当が*たぶん*新鮮だと言う信念のもとで、両者が分かれる様子を見よう：

<!-- validate: skip-output -->
```python
import jax
import jax.numpy as jnp
import jax.random as jr

# states theta in {fresh(0), stale(1)}; actions {eat(0), compost(1)}
L = jnp.array([[0.0, 3.0],      # theta = fresh:  eat -> 0,   compost -> 3
               [10.0, 1.0]])     # theta = stale:  eat -> 10,  compost -> 1
belief = jnp.array([0.9, 0.1])   # belief over theta: P(fresh) = 0.9, P(stale) = 0.1
acts = ["eat", "compost"]
```

```python
exp_loss   = belief @ L                 # belief-weighted (expected) loss: sum over theta of P(theta)*L[theta, a]
worst_case = jnp.max(L, axis=0)         # max over theta (axis 0 = the rows): one worst case per action

print("             eat   compost")
print(f"E[L]       {float(exp_loss[0]):5.2f} {float(exp_loss[1]):5.2f}   ->  Bayes picks {acts[int(jnp.argmin(exp_loss))]}")
print(f"max L      {float(worst_case[0]):5.2f} {float(worst_case[1]):5.2f}   ->  minimax picks {acts[int(jnp.argmin(worst_case))]}")
```

**出力：**
```
             eat   compost
E[L]        1.00  2.80   ->  Bayes picks eat
max L      10.00  3.00   ->  minimax picks compost
```

ベイズ規則は**食べる**：食中毒の小さな $10\%$ の可能性を通常の結果と天秤にかけると、食べることの期待損失（$1.00$）が堆肥にすること（$2.80$）に勝つ。ミニマックス規則は**堆肥にする**：それは $90\%$ を無視し、最悪の列だけを見つめる——食べることは $10$ を冒すが、堆肥にすることは最大でも $3$ で頭打ちなので、賭けることを拒む。どちらも「間違い」ではない。両者は異なるものを最適化している：ベイズは事前分布に賭け、平均で勝つ；ミニマックスは大惨事に対する保険を買い、典型的なケースでその代償を払う。

下の図は、そのトレードオフの規則レベルの眺めだ。ベイズ規則のリスクは、事前分布が $\theta$ は通常ここにいると言うまさにその場所で低く沈む——その代償は、平坦なミニマックス規則より悪くやる薄い切れ端だ——そしてその切れ端こそが、肝心なところで悪くなる代わりに、いたるところでミニマックスが買う*全て*の保護だ：

![世界の状態 theta にわたるリスク曲線。ベイズ規則は、事前分布が質量を置く中央で低く、両端で上昇する U 字を描く；ミニマックス規則は、ベイズ曲線の最悪点の高さに位置する平らな水平線だ。小さな影付きの切れ端が、ベイズ規則のリスクがミニマックス線を超える唯一の領域を示す。](../../images/intro2/dt_risk_curve.png)

---

## あなたはどんな損失を最小化しているのか？

ここに本章全体で最も役立つ事実、そして暗記する価値のある一つがある。これまで $\theta$ はあなたが反応する離散的な*状態*であり、$A$ は短いメニュー（食べる／堆肥にする）だった。今、$\theta$ を**あなたが推定したい連続量**——たとえばChibanyの弁当の重さ——とし、行動を*あなたが報告する数*としよう、すると行動集合 $A$ は正の実数になる（重さは負になれない）。行動がこのように*推定値そのもの*であるとき、**あなたが選ぶ損失関数が、事後分布のどの要約が最適かをひそかに選んでいる。** 三つの損失、三つの要約：

- **0–1損失**、$L = \mathbb{1}[a \neq \theta]$（ぴったり正しくない限り $1$ を払う）→ **事後最頻値**、別名 **MAP** 推定値（*maximum a posteriori*、最大事後確率）。
- **二乗損失**、$L = (\theta - a)^2$（大きな誤りは二次的に痛む）→ **事後平均**。
- **絶対損失**、$L = |\theta - a|$（誤りは比例して痛む）→ **事後中央値**。

これは三回別々に暗記しなければならない偶然ではない——同じ事後分布に対する「期待損失を最小化する」の三通りの読み方だ。Chibanyの弁当の重さについての歪んだ事後分布を取り、力ずくで各要約を*導出*しよう：あらゆる候補の推定値をなめ、事後分布のもとでのその期待損失を計算し、最も安いものを残す。

<!-- validate: tol=0.05 -->
```python
grid = jnp.linspace(0.0, 10.0, 1001)             # candidate weights / estimates (×100 g)
dens = grid**2.3 * jnp.exp(-grid / 1.15)         # a skewed posterior over the weight
dens = dens / jnp.trapezoid(dens, grid)          # normalize it

# the three summaries, read straight off the posterior
mode   = grid[jnp.argmax(dens)]
mean   = jnp.trapezoid(grid * dens, grid)
cdf    = jnp.cumsum(dens) * (grid[1] - grid[0])
median = grid[jnp.argmin(jnp.abs(cdf - 0.5))]
print(f"read off the posterior:   mode = {float(mode):.2f}   mean = {float(mean):.2f}   median = {float(median):.2f}")

# now DERIVE each as the Bayes estimator: the action a minimizing expected loss
def bayes_estimator(loss):
    T, A = grid[None, :], grid[:, None]                       # theta (cols) vs action (rows)
    expected_loss = jnp.trapezoid(loss(T, A) * dens[None, :], grid, axis=1)
    return grid[jnp.argmin(expected_loss)]

a_01  = bayes_estimator(lambda t, a: (jnp.abs(t - a) > 0.05).astype(float))   # 0–1 loss
a_sq  = bayes_estimator(lambda t, a: (t - a) ** 2)                            # squared loss
a_abs = bayes_estimator(lambda t, a: jnp.abs(t - a))                          # absolute loss
print(f"argmin 0–1 loss      = {float(a_01):.2f}   (lands on the mode)")
print(f"argmin squared loss  = {float(a_sq):.2f}   (lands on the mean)")
print(f"argmin absolute loss = {float(a_abs):.2f}   (lands on the median)")
```

**出力：**
```
read off the posterior:   mode = 2.64   mean = 3.70   median = 3.39
argmin 0–1 loss      = 2.62   (lands on the mode)
argmin squared loss  = 3.70   (lands on the mean)
argmin absolute loss = 3.39   (lands on the median)
```

三つのargminは三つの要約に着地する（$0$–$1$ の結果はグリッドの解像度の範囲で最頻値に乗っている）。歪んだ事後分布では、これらは正真正銘*異なる数*だ——最頻値 $2.64$、中央値 $3.39$、平均 $3.70$——だからあなたが選ぶ損失は些末事ではない：それはあなたの答えを動かす。下の左パネルは、各損失が*なぜ*自分の要約へと引き寄せるのかを示す。$0$–$1$ 損失は、真値の毛一本の幅以内に着地したときだけ何も払わないので、事後*密度*が最も高い場所——**最頻値**——に推定値を置くことに報いる（その毛一本の幅がコードの `0.05` の許容帯だ）。二乗損失の急な放物線は、遠くの誤りを厳しく罰するので、**平均**を追いかける。そして一定の率で増える絶対損失は、事後質量を半分に分けるとき最小になる——**中央値**だ。右パネルは三つすべてを事後分布上に印す：

![二つのパネル。左には誤差の関数としての三つの損失曲線：0-1 損失の平らな底のノッチ、二乗損失の放物線、絶対損失の V 字。右には theta にわたる右に歪んだ事後分布があり、三本の縦の破線が、最も左の最頻値、次に中央値、次に最も右の平均を印しており、歪んだ事後分布が三つの要約を引き離すことを示している。](../../images/intro2/dt_loss_estimators.png)

{{% expand "なぜか、代数三行で（任意）" %}}
各要約は一つの短い導出から落ちてくる、[第16章](../16_monte_carlo/)の*確率は指示関数の期待値である*という恒等式を使って。

- **0–1損失 → 最頻値。** $a$ を報告することの期待損失は $\mathbb{E}\big[\mathbb{1}[a \neq \theta]\big] = 1 - P(\theta = a)$。それを最小化することは、$a$ における事後質量を*最大化*することを意味する——それは**最頻値**にある。
- **二乗損失 → 平均。** 導関数をゼロに置く：$\frac{d}{da}\,\mathbb{E}\big[(\theta - a)^2\big] = -2\,\mathbb{E}[\theta - a] = 0 \implies a = \mathbb{E}[\theta]$、すなわち**平均**。
- **絶対損失 → 中央値。** $\frac{d}{da}\,\mathbb{E}\big[\,|\theta - a|\,\big] = P(\theta < a) - P(\theta > a) = 0 \implies P(\theta < a) = P(\theta > a) = \tfrac12$ ——事後質量を半分に分ける値、すなわち**中央値**。
{{% /expand %}}

自分で損失タイプを切り替えてみて、最適な推定値が三つの要約の間を跳ぶのを見よう——そしてその下の期待損失曲線が、それに合わせて最小値をスライドさせるのを見よう：

<iframe src="../../widgets/decision-loss-explorer.html"
        width="100%" height="560"
        frameborder="0"
        style="background:#111111; border-radius:6px; margin:1rem 0;"
        title="Interactive loss-to-estimator explorer: switch between 0-1, squared, and absolute loss and watch the Bayes estimate move between mode, mean, and median">
</iframe>

（二乗損失に切り替えて最小値がどこに座るか見よう；次に絶対損失にして、それが左へ中央値までスライドするのを見よう；次に $0$–$1$ にして、それがピークへ跳ぶのを見よう。）

{{% notice style="tip" title="なぜこれが2026年でも重要なのか" %}}
「平均を報告せよ」と「最も可能性の高いラベルを報告せよ」は中立的な既定値ではない——それぞれが*特定の*損失への最適な答えだ。二乗誤差で訓練された回帰モデルは事後平均にコミットしている；自分のトップクラスを報告する分類器は $0$–$1$ 損失にコミットしている；*較正された*中央値予測を求められたモデルは絶対損失にコミットしている。あるシステムの出力が自分の問題に対して較正がずれているように感じられるなら、それが訓練された損失こそが最初に見るべき場所だ。
{{% /notice %}}

---

## 認知についての余談：決定理論家としての脳

{{% notice style="info" title="人は確率だけでなく損失も天秤にかける" %}}
決定理論は機械のためのレシピであるだけではない——それは*人間*が何をするかの驚くほど良いモデルでもある。古典的な実験で、Körding & Wolpert (2004) は被験者に、自分の手がどこにあるかが不確実な状況で素早いリーチング運動をさせ、人工的な損失関数で報酬を与えた。人々のリーチは、ベイズ的決定者がするのとまさに同じように動いた：彼らは手の位置についての**事前分布**を、報酬を受け取った**損失**と組み合わせ、*期待*損失を最小化するように狙った——最も可能性の高い正解を狙ったのではない。感覚運動系は、結局のところ、本章の問題を静かに解いているのだ。このテーマ——*ベイズ的決定としての認知*——は、強化学習と脳に至ったときに再び戻ってくる。
{{% /notice %}}

---

## GenJAXで

これで**観測**へと輪を閉じられる。閾値規則 $d_k$ は閾値 $k$ を必要とし、閾値は日数から新鮮さについての信念への橋を必要とする——本章冒頭の**日数 → P(新鮮)** の変換だ。GenJAXでは、その変換は一行の生成モデル*そのもの*だ：日数が新鮮さの確率を設定し、`flip` が隠れた状態を引く。

<!-- validate: skip-output -->
```python
from genjax import gen, flip

@gen
def freshness(days):
    p_fresh = 1.0 / (1.0 + jnp.exp((days - 5.0) / 1.5))   # the days -> P(fresh) conversion
    return flip(p_fresh) @ "fresh"                          # True = fresh, False = stale
```

これで決定は新しいアイデアなしに落ちてくる——それは[第16章](../16_monte_carlo/)のモンテカルロのループだ：ある日数に対して、新鮮さを何度も*サンプリング*し、各行動の損失を表に対して平均し、より安い方を取る。日数をなめて、最適な行動が反転するのを見よう：

<!-- validate: tol=0.4 -->
```python
def decide(days, key, n=20000):
    fresh = jax.vmap(lambda k: freshness.simulate(k, (float(days),)).get_retval())(jr.split(key, n))
    f = fresh.astype(float); stale = 1.0 - f
    el_eat     = jnp.mean(f * L[0, 0] + stale * L[1, 0])    # L[theta, action]; action 0 = eat
    el_compost = jnp.mean(f * L[0, 1] + stale * L[1, 1])    # action 1 = compost
    return float(el_eat), float(el_compost)

print(" day  P(fresh)  E[L:eat]  E[L:compost]   decision")
for d in range(8):
    ee, ec = decide(d, jr.fold_in(jr.key(0), d))
    pf = float(1.0 / (1.0 + jnp.exp((d - 5.0) / 1.5)))
    print(f"  {d}      {pf:.2f}     {ee:5.1f}      {ec:5.1f}        {'eat' if ee < ec else 'compost'}")
```

**出力：**
```
 day  P(fresh)  E[L:eat]  E[L:compost]   decision
  0      0.97       0.4        2.9        eat
  1      0.94       0.6        2.9        eat
  2      0.88       1.2        2.8        eat
  3      0.79       2.1        2.6        eat
  4      0.66       3.4        2.3        compost
  5      0.50       5.0        2.0        compost
  6      0.34       6.6        1.7        compost
  7      0.21       7.9        1.4        compost
```

決定は**4日目**で反転する：直近3日以内に買ったものは何でも食べ、残りは堆肥にする。その閾値 $k = 3$ は、私たちが選んだ数ではない——それは損失（食中毒に $10$、無駄に $3$）と日数→新鮮さモデルがともに*強いる*ものであり、サンプルを引いて損失を平均する以上の何ものでもなく見つけ出される。本章冒頭の抽象的な閾値規則 $d_k$ は、ちょうどその $k$ を得たのだ。

これはコースの残りの間あなたが走らせる同じループだ。しかしこれは、私たちが静かにかわしてきた問いを提起する：各決定は**20,000**サンプルを引いた。もし各サンプルが何かを*要する*としたら？

---

## サンプルは何個？一つで十分

一回のお昼を決めるために二万サンプルを引くのは、各サンプルが一秒の思考であるなら馬鹿げている。現実のエージェント——人、動物、締め切りに追われるロボット——は、すべてのサンプルを**時間**で支払い、熟慮に費やす時間は次の決定に費やさない時間だ。だから問いは「どうすれば最良の推定値が得られるか？」ではなくなり、「どれだけ良い決定を*払えるか*？」になる。

Vul, Goodman, Griffiths & Tenenbaum (2014) は、答えを明かすタイトルの論文でこれを正確にした：*One and Done?*。各決定について、信念から $k$ 個のサンプルを引き、**多数決**に従う、そんな決定の流れを思い描こう。サンプルが多いほど、より信頼できる選択になる——しかしあなたが実際に最大化したいものは、あなたの**報酬率**だ：単位時間あたりの良い決定。各サンプルが時間を要するなら、その率は

$$\text{reward rate}(k) = \frac{P(\text{correct} \mid k)}{1 + c\,k},$$

ここで $c$ は決定の残りに対する一サンプルの時間コストだ。分子の正確さは $k$ とともに上がる；分母の時間も上がる。（私たちはあなたの信念が**較正されている**と仮定する——単一のサンプルがちょうど確率 $p$ でより良い選択肢を指す——ので、「信念からサンプリングする」と「確率 $p$ で正しい」が一致する；偶数の $k$ での同票は公正なコインで決める。）率はどこでピークに達するか？

<!-- validate: skip-output -->
```python
import jax.numpy as jnp
from jax.scipy.special import gammaln

def p_correct(k, p):
    # P(majority of k samples from your belief favors the better option), ties split fairly.
    # p = your belief that the better option really is better.
    j = jnp.arange(k + 1)
    log_choose = gammaln(k + 1.) - gammaln(j + 1.) - gammaln(k - j + 1.)
    pmf = jnp.exp(log_choose + j * jnp.log(p) + (k - j) * jnp.log1p(-p))
    win = (j > k / 2) + 0.5 * (j == k / 2)
    return float(jnp.sum(pmf * win))
```

```python
p, cost = 0.75, 0.1               # belief: 75% sure; each sample costs 0.1 of a decision's time
ks   = list(range(1, 13))
acc  = [p_correct(k, p) for k in ks]
rate = [a / (1 + cost * k) for a, k in zip(acc, ks)]
best = ks[int(jnp.argmax(jnp.array(rate)))]

print(" k   P(correct)   reward rate")
for k, a, r in zip(ks, acc, rate):
    print(f"{k:2d}     {a:.3f}        {r:.3f}{'   <- best' if k == best else ''}")
print(f"\noptimal number of samples: k* = {best}  (one and done)")

# what the k=1 policy actually DOES: it follows its single sample, so it picks the
# option it believes is better with probability p -- it MATCHES its belief.
print(f"one-and-done picks the believed-better option with prob {p:.2f}  (probability matching)")
print(f"'always pick the more likely' (maximizing) would pick it with prob 1.00")
```

**出力：**
```
 k   P(correct)   reward rate
 1     0.750        0.682   <- best
 2     0.750        0.625
 3     0.844        0.649
 4     0.844        0.603
 5     0.896        0.598
 6     0.896        0.560
 7     0.929        0.547
 8     0.929        0.516
 9     0.951        0.501
10     0.951        0.476
11     0.966        0.460
12     0.966        0.439

optimal number of samples: k* = 1  (one and done)
one-and-done picks the believed-better option with prob 0.75  (probability matching)
'always pick the more likely' (maximizing) would pick it with prob 1.00
```

一つのサンプル。思考が時間を要するとき、報酬率を最適にする方策は、信念から**単一の**サンプルを引いてそれに基づいて行動することだ——*one and done*。（偶数のサンプルが、その下の奇数に決して勝たないことに注意せよ——二つ目のサンプルは同票を作ることしかできず、最初の票を破ることはできない——なので $k=2$ は正確さで $k=1$ に並ぶが、時間で負ける。サンプルがほぼ無料の場合だけ、あなたはもっと引くだろう：図はピークが $c$ が縮むにつれて右へスライドするのを示す。）

![サンプル数 k を1から12までプロットした図。緑の正確さ曲線が 0.75 から 0.97 に向かって着実に上昇する。二つの報酬率曲線も示される：高いサンプルコストでは率は k が1のとき最も高く、その後低下する；低いサンプルコストでは率は後で、k が7あたりでピークに達する。メッセージは、正確さは上がり続けるが報酬率は早くピークに達するということだ。](../../images/intro2/one_and_done.png)

ここに、これを*心*についての章にする決め台詞がある。一サンプルの決定は、あなたの信念に等しい確率 $0.75$ で「新鮮」を選ぶ——それは常に最も可能性の高い選択肢を取るのではなく、事後分布に**マッチ**する。その振る舞い、**確率マッチング**は、選択の心理学において最も古い「非合理性」の一つだ：人々は常に最良のものに従うのではなく、確率に比例して選択肢を選ぶ。*One and Done* はそれを**最適**として捉え直す——時間を大切にするサンプラーがまさにすべきことだ。同じ筋が[第19章](../19_sampling_the_mind/)を貫いている：少数のサンプルから推論する心は、壊れたベイズ主義者ではなく、効率的なベイズ主義者なのだ。

これは決定理論、モンテカルロ、認知が出会う継ぎ目であり——そしてそれがコースの残りを発進させる。次の二章は「自分のモデルをサンプリングして決める」ループを保ちつつ、単一の行動を*列*へと引き伸ばす、そこでは今日の選択が明日の世界を変え、**報酬率**こそがエージェントが最大化することを学ぶまさにそのものになる。

{{% notice style="success" title="今できること" %}}
あなたは**決定問題**——世界の状態 $\theta$、観測 $x$、行動集合 $A$、決定規則 $d(x)$、損失 $L(\theta, a)$ ——を述べることができ、信念が選択になるのは、間違いが何を**要する**かを言ったときだけだと知っている。あなたは規則の**リスク**（その期待損失）を計算でき、**ベイズ**基準（事前／事後期待損失を最小化する）と**ミニマックス**基準（最悪のケースを最小化する）の間で選ぶことができ、両者が食い違いうることを知っている。あなたは暗記する価値のある規則を知っている——**0–1損失 → 最頻値（MAP）、二乗 → 平均、絶対 → 中央値**——だから「どの要約を報告するか？」は二度と恣意的にはならない。あなたは事後分布が**サンプル**だけのときでも決定を下すことができ、GenJAXで期待損失を推定してargminを取る。そしてサンプリングが時間を要するときに*何個*サンプルを引くか——しばしばたった**一つ**——を知っており、その「one and done」方策がなぜ人間の**確率マッチング**を再現するのかを知っている。

次に、[第21章](../21_markov_decision_processes/)はこの単一の選択を*列*に変える：今日の行動が明日の世界を変えるとき、一つの決定はもはや十分ではない——あなたは**方策**を必要とする。

*用語集：* [決定理論](../../glossary/#statistical-decision-theory-), [決定規則](../../glossary/#decision-rule-), [観測](../../glossary/#observation-), [損失関数](../../glossary/#loss-function-), [リスク](../../glossary/#risk-), [ベイズ推定量](../../glossary/#bayes-estimator-), [ミニマックス](../../glossary/#minimax-), [MAP推定値](../../glossary/#map-estimate-), [確率マッチング](../../glossary/#probability-matching-).
{{% /notice %}}

---

## 演習

{{% notice style="info" title="自分で試してみよう" %}}
1. **事前分布を反転させる。** ベイズ対ミニマックスのセルで、信念を `P(stale) = 0.4`（怪しげに見える弁当）に変えよう。各行動の事後期待損失を再計算しよう——ベイズ規則はまだ食べるか？ ベイズが食べるから堆肥にするへ反転する `P(stale)` の値を見つけ、なぜミニマックスは決して反転しないのかを説明しよう。
2. **四つ目の損失。** 損失→推定量のセルに*非対称な*損失を加えよう：過大推定を過小推定の二倍厳しく罰する（$L = (\theta-a)^2$ if $a < \theta$、それ以外は $2(\theta-a)^2$）。最適な推定値は平均からどちらの方向へ動くか、そしてそれはあなたの直感と一致するか？
3. **サンプリング対厳密。** `decide` のセルで、`n` を $50$ に下げて、異なるキーで3日目の弁当に対して何度か呼び出そう（`decide(3, jr.key(1))`、`decide(3, jr.key(2))`、……）。二つの期待損失はどれだけ揺れるか——そして*決定*は反転することがあるか？ 見たものを[第16章](../16_monte_carlo/)の $1/\sqrt{n}$ の規則に関連付けよう。
4. **一つでは*足りない*のはいつ？** *One and Done* のセルで、サンプルコスト `cost` を $0.2$ から $0.001$ までなめよう。最適な $k^*$ が初めて $1$ を超えて跳ぶのは、おおよそどのコストでか？ 次に信念 `p` を $0.95$ に向けて上げ（容易な決定）、再実行しよう——より自信のあるエージェントは*より多くの*サンプルを必要とするか、より少なくか、そしてなぜか？
{{% /notice %}}

これらすべてをインタラクティブに進める対応ノートブック：

**📓 [Colab で開く: `20_statistical_decision_theory.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/20_statistical_decision_theory.ipynb)**

---

## 参考文献

- Körding, K. P., & Wolpert, D. M. (2004). Bayesian integration in sensorimotor learning. *Nature, 427*(6971), 244–247. <https://doi.org/10.1038/nature02169>
- Vul, E., Goodman, N. D., Griffiths, T. L., & Tenenbaum, J. B. (2014). One and done? Optimal decisions from very few samples. *Cognitive Science, 38*(4), 599–637. <https://doi.org/10.1111/cogs.12101>

---

Special thanks to [JPPCA](https://jpcca.org/) for their generous support of this tutorial series.
