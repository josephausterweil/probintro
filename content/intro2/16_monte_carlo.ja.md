+++
date = "2026-06-16"
title = "モンテカルロ法：サンプリングによる推定"
weight = 16
+++

## 合計できない問い

この3章にわたって、マルコフ連鎖を*動かし*、どこに落ち着くかを見てきた。[第15章](../15_memory_search/)では約束を残して終わった：分布について学ぶために連鎖を走らせるという考え方には名前がある——**モンテカルロ**——そしてこれから先の章でそれをツールとして磨き上げる、と。本章がその出発点だ。

動機となる問いを示そう。Chibanyは一学期を通じて毎日2つの弁当を食べており、その重さは日々変わる——軽いおにぎりセットの日もあれば、重いカツ丼の日もある。Jamalは気になっている。

> **Jamal：**「今学期の弁当1つの*平均*重量ってどのくらい？」
>
> **Chibany：**「全部の弁当の重量を足して、個数で割らないといけないね。」
>
> **Alyssa：**「でもリストがないでしょう。仮に重量が滑らかな分布に従っていたとしても、*積分*——ありとあらゆる重量に対して、重量×確率を合計——しないといけない。それはものすごく大変な計算だよ。」

Alyssaが問題の核心を突いている。Jamalが求めているのは**期待値**であり、それを厳密に計算するにはすべての可能性について和（または積分）を取る必要がある。弁当の場合は可能性が多すぎるし、[第12章](../12_hierarchical_bayes/)のベイズ事後分布に至っては、積分に閉じた形の解が存在しない——まさにあの章で重点サンプリングを「粗削りなツール」と表現した理由だ。

リストも積分も不要な抜け道がある。**サンプリングして、平均する。**これがモンテカルロ法の全てであり、本章ではサイコロを一振りすることから始めて、手計算では絶対に求められない確率を推定できるほど鋭いツールへと磨き上げていく。

---

## モンテカルロ推定量

まず、求めたいものを確認しよう。分布 $P$ から確率変数 $X$ が引かれるとき、関数 $f$ の**期待値**とは、長期的な平均値のことだ。連続の場合は積分になる：

$$\mathbb{E}_P[f(X)] = \int f(x) p(x) dx,$$

これは「各 $x$ がどれだけ起こりやすいかで重み付けして $f(x)$ を合計せよ」という意味だ。（[第1章](../01_mystery_bentos/)で*離散*の和として $\mathbb{E}$ を学んだ；これは同じ考え方で、和が積分になったものだ。）通常この積分は計算できない。しかし**推定**はできる。$P$ から独立に $n$ 個のサンプル $x_1, \dots, x_n$ を引き、$f$ の平均を取ればよい：

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^{n} f(x_i), \qquad x_i \sim P.$$

$\hat\mu_n$ のハットは「サンプルから推定された」という意味だ。これが**モンテカルロ推定量**であり、これが機能する理由は、あなたがすでに「十分長く走らせれば」と言うたびに非公式に使ってきた法則にある：**大数の法則**は $n$ が大きくなるにつれて $\hat\mu_n$ が真の $\mathbb{E}_P[f(X)]$ に収束することを保証する。

どのくらいの*速さ*で？誤差は $1/\sqrt{n}$ のように縮小する：誤差を半分にするには*4倍*のサンプルが必要だ。遅く聞こえるかもしれないが、驚くべき性質がある——$1/\sqrt{n}$ の収束率は $x$ が何次元に存在するかに左右されない。手計算の積分は次元が増えるにつれて指数関数的に難しくなるが、モンテカルロ法はそうではない。この次元への無関心さが、現代統計学の主力手法になった理由だ。

最もシンプルな期待値——公正なサイコロの平均の目、真の値は $\tfrac{1+2+3+4+5+6}{6} = 3.5$——での収束を観察してみよう。

<!-- validate: tol=0.15 -->
```python
import jax
import jax.numpy as jnp
import jax.random as jr

def die_estimate(key, n):
    rolls = jr.randint(key, (n,), 1, 7)        # n uniform draws from {1, ..., 6}
    return jnp.mean(rolls.astype(float))       # the Monte Carlo average

key = jr.key(0)
for n in [10, 100, 1000, 100000]:
    est = die_estimate(jr.fold_in(key, n), n)
    print(f"n = {n:6d}:  estimate of E[die] = {float(est):.3f}")
```

**出力：**
```
n =     10:  estimate of E[die] = 3.000
n =    100:  estimate of E[die] = 3.370
n =   1000:  estimate of E[die] = 3.484
n = 100000:  estimate of E[die] = 3.499
```

$n=10$ では推定値が半目ほどずれているが、$n=100{,}000$ では $3.5$ との差が千分の一以内に収まる。「サイコロの平均は3.5」と書き下したわけではない——振ることで*発見した*のだ。全体像を一枚の図に示す——推定値が序盤は大きく揺れ、その後 $1/\sqrt{n}$ の収束エンベロープが縮むにつれて 3.5 に落ち着いていく様子だ：

![サイコロの期待値のモンテカルロ推定量の推移を、振り回数（対数スケール）に対してプロットした図。推定値は100回未満では大きく振れ、その後3.5の破線に落ち着き、n の平方根の逆数のように縮む陰影のエンベロープ内に収まる。](../../images/intro2/mc_die_convergence.png)

### 少ないサンプルは大きくブレる：病院問題

$1/\sqrt{n}$ の収束率には有名な裏面があり、先に進む前に体感しておく価値がある。次は古典的な問題だ（Kahneman & Tversky）：

> **大きな**病院（1日約45件の出産）と**小さな**病院（1日約15件の出産）が、1年間にわたって**生まれた赤ちゃんの60%以上が男の子だった日**を記録した。どちらの病院がそのような日を**より多く**記録したか？

ほとんどの人は「同じくらい——男の子は50%で変わらない」と答える。しかしあなたはもっとよく知っている：45件の出産は15件より大きなサンプルなので、大きな病院の日々の男の子の割合は50%の周りに*ぴったりと*集まるが、小さな病院の割合は**大きく揺れ**、60%を超えることがはるかに多い。小さいサンプルほど変動が大きい；これは大数の法則を逆から読んだものだ。そして確認に使える方法に注目してほしい：何も計算しない——**1年分をシミュレートして数える**。この動きこそがモンテカルロ法であり、それ自身の収束に適用したものだ。

<!-- validate: tol=10 -->
```python
def days_over_60(key, births_per_day, n_days=365):
    boys = jr.binomial(key, n=births_per_day * 1.0, p=0.5, shape=(n_days,))  # a year of days
    frac = boys / births_per_day
    return int(jnp.sum(frac > 0.6))                       # count the >60%-boys days

big   = days_over_60(jr.key(0), 45)
small = days_over_60(jr.key(1), 15)
print(f"large hospital (45 births/day): {big:3d} days over 60% boys")
print(f"small hospital (15 births/day): {small:3d} days over 60% boys")
```

**出力：**
```
large hospital (45 births/day):  27 days over 60% boys
small hospital (15 births/day):  55 days over 60% boys
```

小さな病院は偏った日をほぼ**2倍**記録している。多数の日のヒストグラムを描くとメカニズムは明白だ——小さな病院の日々の男の子の割合は単純に*幅広い*分布であるため、60%の線を超える部分がはるかに多い：

![大きな病院（1日45件）と小さな病院（1日15件）の、日々の男の子の割合のヒストグラムを2つ並べた図。どちらも50%を中心としているが、小さな病院のヒストグラムははるかに幅広く、破線の60%を超える赤い領域が大きな病院より数倍大きい。](../../images/intro2/mc_hospital_tails.png)

同じ事実が本章の後半で重要になって戻ってくる：*実効的に少ない*サンプルで構築された推定量はまさにこの形で大きくブレる。

---

## ダーツを投げて π を求める

サイコロは準備運動だった：答えがわかっていた。次は数えることでは求められないもの——数 $\pi$ ——のモンテカルロ推定で、乱数点の列だけを使う。

単位正方形 $[0,1] \times [0,1]$ の中に、角を中心とした半径1の四分円を描いたとしよう。正方形の面積は $1$；四分円の面積は $\tfrac{\pi}{4}$ だ。したがって正方形上にダーツを一様にばらまくと、四分円の内側に落ちる*割合*はおよそ $\tfrac{\pi}{4}$ になるはずだ——その割合の4倍が $\pi$ の推定値となる。

$(x, y)$ のダーツが内側かどうか調べるには、$x^2 + y^2 \le 1$ を確認すればよい。このyes/noを平均できる数値に変えるために、**指示関数** $\mathbb{1}[\cdot]$ を使う：

$$\mathbb{1}[\text{事象}] = \begin{cases} 1 & \text{事象が真の場合} \\ 0 & \text{偽の場合。} \end{cases}$$

つまり $\mathbb{1}[x^2 + y^2 \le 1]$ は四分円内のダーツで $1$、外側で $0$ となり、多数のダーツにわたる**平均**が内側の割合そのものだ。（このトリックに注目：*確率*とは*指示関数の期待値*に他ならない。これを繰り返し使う。）

```mermaid
graph LR
    A["ダーツ (x,y) ~ 一様正方形"] --> B{"x² + y² ≤ 1 ?"}
    B -->|はい| C["1[内側] = 1"]
    B -->|いいえ| D["1[内側] = 0"]
    C --> E["π̂ = 4 × 平均"]
    D --> E
    classDef node fill:none,stroke:#9bbcff,stroke-width:2px,color:#fff
    class A,B,C,D,E node
    linkStyle default stroke:#9bbcff,stroke-width:2px,color:#fff
```

<!-- validate: tol=0.05 -->
```python
def estimate_pi(key, n):
    xy = jr.uniform(key, (n, 2))                   # n darts in the unit square
    inside = (xy[:, 0]**2 + xy[:, 1]**2) <= 1.0    # indicator: inside the quarter-circle
    return 4.0 * jnp.mean(inside.astype(float)), int(jnp.sum(inside))

pi_hat, n_in = estimate_pi(jr.key(1), 100000)
print(f"darts inside the circle: {n_in} / 100000")
print(f"pi estimate = 4 x {n_in}/100000 = {float(pi_hat):.3f}")
```

**出力：**
```
darts inside the circle: 78345 / 100000
pi estimate = 4 x 78345/100000 = 3.134
```

10万個のランダムなダーツで $\pi$ を小数点以下2桁まで絞り込む。$1/\sqrt{n}$ のルールで3桁目を絞るには約100倍のダーツが必要——遅いが完全に機械的であり、「内側とは $x^2 + y^2 \le 1$ を意味する」以外に円に関する事実は一つも必要としない。ダーツが積み重なるにつれ、図がどのように鮮明になるか見てみよう：

![n = 10、100、1000、10000 のときに単位正方形に散らばったダーツの4パネル図。四分円の内側にある点は緑、外側にある点は赤。各タイトルのπの推定値は少ないn（10個で3.6）のとき揺れ、1万個までに3.14に近づく。](../../images/intro2/mc_pi_darts.png)

自分でダーツを投げてみよう——最初は推定値の経路が激しく跳ね、その後 $1/\sqrt{n}$ のルールが約束するとおり落ち着くのを体感してほしい：

<iframe src="../../widgets/mc-darts.html"
        width="100%" height="540"
        frameborder="0"
        style="background:#111111; border-radius:6px; margin:1rem 0;"
        title="Interactive dart-throwing estimator of pi">
</iframe>

（まず1本ずつ投げて——推定値がどれほど不安定か感じて——次に一度に1万本追加して固定していく様子を見よう。）

---

## P からサンプリングできない場合：棄却法と逆CDF法

これまでの例はどちらも簡単なものからサンプリングしていた——サイコロ、一様な正方形。現実のターゲットはそう都合よくない。「簡単な」サンプルを「難しい」サンプルに変える2つの古典的テクニックを紹介する。

**棄却サンプリング。**ターゲット密度 $p(x)$ を*評価*できるが直接引けない場合、上にシンプルなエンベロープ——*サンプリングできる*箱——を被せ、箱の下に一様に点を投げて、**$p$ の下に落ちた点だけを保持する**。残った点は $p$ からの厳密なサンプルであり、残りは棄却される。欠点は効率だ：箱が $p$ の下の面積より大幅に大きいと、作業のほとんどが棄却される。

次は、$[0,1]$ 上で $p(x) \propto x$ という斜面状のターゲット（弁当は重い方に偏りやすい）を、高さ2の平らな箱の下での棄却によってサンプリングする例だ。

<!-- validate: tol=0.05 -->
```python
def rejection_sample(key, n_tries):
    kx, ky = jr.split(key)
    xs = jr.uniform(kx, (n_tries,))            # propose x ~ Uniform(0, 1)
    heights = jr.uniform(ky, (n_tries,)) * 2.0 # a height under the box (top = 2)
    keep = heights <= (2.0 * xs)               # accept if under p(x) = 2x
    return xs[keep], float(jnp.mean(keep))

samples, acc = rejection_sample(jr.key(2), 100000)
print(f"acceptance fraction: {acc:.3f}  (theory 0.5)")
print(f"kept {samples.shape[0]} samples; their mean = {float(jnp.mean(samples)):.3f}  (theory 2/3)")
```

**出力：**
```
acceptance fraction: 0.501  (theory 0.5)
kept 50077 samples; their mean = 0.667  (theory 2/3)
```

提案の半分が捨てられる——そして残った点の平均は $\tfrac{2}{3}$、$[0,1]$ 上の斜面の平均にぴったり一致する。図にすると、手法はほぼ自明だ：

![破線の矩形ボックスを埋める点の散布図。ターゲット密度 p(x) = 2x が斜めの線として描かれている。線の下の点は緑で保持され、上の点は赤で棄却されている——作業のほぼ半分が捨てられる。](../../images/intro2/mc_rejection.png)

**逆CDF サンプリング。**累積分布関数 $F(x) = P(X \le x)$ を書き下してその逆関数を求められる場合、無駄は一切ない：$u \sim \text{Uniform}(0,1)$ を引いて $F^{-1}(u)$ を返す。一様な引きが全てターゲットからの1サンプルになる。（[第13章](../13_markov_chains/)ですでにこれの1段階版を行った——「$u$ を引いて行列の行と比較する」は2結果分布の逆CDF サンプリングだった。）

棄却法はエンベロープが緩いとサンプルを無駄にする。その無駄を解決するのが次のアイデアだ——サンプルを捨てるのではなく、*全て保持して重みを調整する*ことで。

---

## 重点サンプリング：別の分布からサンプリングする

エンベロープすら見つけにくい場合や、ランダムサンプリングがほとんど訪れない細い領域が関心対象の場合がある。解決策は大胆だ：**意図的に別の、より簡単な分布からサンプリングして、その嘘を補正する。**

これを正確にするために2つの分布が必要だが、この章の残り——そして次の2章——がこの区別に依拠するので、明確に名前をつけておく価値がある：

- **ターゲット** $p(x)$ ——実際に関心のある分布（評価できるが簡単にはサンプリングできないことが多い）、
- **提案分布** $q(x)$ ——代わりにサンプリングする、引きやすい代替分布。

（[第12章](../12_hierarchical_bayes/)ではこれの離散版を見た。「提案」が候補事前分布の有限グリッドだった。ここでは $q$ と $p$ は*あなた*が選べる完全な連続分布だ。）

トリックは1行の代数だ。$q$ で掛けて割る：

$$\mathbb{E}_P[f(X)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_q\left[ f(X) \frac{p(X)}{q(X)} \right].$$

つまり $p$ に関する期待値は $q$ に関する期待値と等しい。各サンプルに**重点重み**

$$w(x) = \frac{p(x)}{q(x)}$$

を掛ける限りにおいて。*良い*提案 $q$ は $p$ に似ている——$p$ が広いところでは広く、重みは1に近い。*悪い*提案は $p$ が存在する場所を外し、少数のサンプルに巨大な重みが付き、残りはほぼゼロになる。（その思考を保留しておいてほしい；それが本章最後のアイデアになる。）

次は利益確定の例：直接サンプリングではほぼ命中しない**裾確率**の推定だ。Chibanyの弁当重量は $p = \mathcal{N}(620, 50^2)$ グラムに従うとする。弁当の中で**700 g**を超えるもの——かなりしっかりしたランチ——の割合は？$p$ から直接サンプリングすると、700を超えるのはごく少数なので推定がノイジーになる。代わりに $q = \mathcal{N}(700, 50^2)$、つまり*重い裾にシフト*した分布から提案して、再重み付けする。図はその動きを示す——提案が関心のある事象の真上に駐在している：

![620グラムを中心とした弁当重量のターゲット正規曲線。700グラムを超える小さな陰影の裾が関心のある事象を示し、同じ幅の破線の提案曲線がちょうど700を中心として——ターゲットがほとんど届かない裾をカバーしている。](../../images/intro2/mc_is_tail.png)

<!-- validate: tol=0.02 -->
```python
from jax.scipy.stats import norm

mu_p, sd_p = 620.0, 50.0      # target: where bento weights actually live
mu_q, sd_q = 700.0, 50.0      # proposal: shifted onto the heavy tail we care about

def is_tail(key, n):
    x = mu_q + sd_q * jr.normal(key, (n,))                    # sample from q
    w = norm.pdf(x, mu_p, sd_p) / norm.pdf(x, mu_q, sd_q)     # importance weight p/q
    f = (x > 700.0).astype(float)                            # indicator of "heavy bento"
    return float(jnp.mean(f * w))

est = is_tail(jr.key(3), 100000)
truth = float(1 - norm.cdf(700.0, mu_p, sd_p))
print(f"IS estimate of P(W > 700) = {est:.4f}")
print(f"true value                = {truth:.4f}")
```

**出力：**
```
IS estimate of P(W > 700) = 0.0542
true value                = 0.0548
```

*行動が起きる場所*からサンプリングして重みで補正することで、重点サンプリングは裾確率を綺麗に推定できる。これが本章の残りを支える動きだ——そして課題で自分のターゲットに対して引くよう求められるレバーでもある。

### 自己正規化 IS と非正規化 p

繊細だが非常に有用なバリエーションがある。ベイズ推論ではターゲットが事後分布 $p(x) = \tfrac{1}{Z}\tilde p(x)$ であり、正規化定数 $Z = p(\text{data})$ はまさに計算*できない*積分だ。重みの**比**だけを取ることにすると何が起きるか見てみよう：

$$\mathbb{E}_P[f(X)] \approx \frac{\sum_i f(x_i) w(x_i)}{\sum_i w(x_i)}, \qquad w(x_i) = \frac{\tilde p(x_i)}{q(x_i)}.$$

未知の $Z$ は全ての重みに同じように現れるため、分子と分母の間で**相殺**される。これが**自己正規化**推定量であり、*非正規化のターゲットでも重点サンプリングができる*ことを意味する——まさに全ての事後分布が置かれる状況だ。（この相殺は[第18章](../18_markov_chain_monte_carlo/)でMCMCを機能させるものと同じだ。）

コインの例で示す：コイン表の確率 $\theta$ に対して $\text{Beta}(2,2)$ の事前分布を置き、表が1回観測されて、事後平均を回収する。

<!-- validate: tol=0.02 -->
```python
def snis_coin(key, n):
    theta = jr.beta(key, 2.0, 2.0, (n,))   # sample theta from the PRIOR (our proposal q)
    w = theta                              # likelihood of one head is theta, so w is proportional to it
    return float(jnp.sum(theta * w) / jnp.sum(w))   # self-normalized: the constant cancels

post_mean = snis_coin(jr.key(4), 200000)
print(f"self-normalized IS posterior mean of theta = {post_mean:.3f}")
print(f"closed form (Beta(3,2) mean = 3/5)         = {3/5:.3f}")
```

**出力：**
```
self-normalized IS posterior mean of theta = 0.601
closed form (Beta(3,2) mean = 3/5)         = 0.600
```

### 尤度重み付け

そのコインの例には美しい特別なケースが隠れていた。提案 $q$ として**事前分布**を選んだ。$q = p(\text{仮説})$ のとき、重点重みは

$$w \propto \frac{p(\text{仮説}) p(\text{データ} \mid \text{仮説})}{p(\text{仮説})} = p(\text{データ} \mid \text{仮説}),$$

つまり**尤度**に崩れる。したがって「事前分布からサンプリングして、尤度で重み付けする」ことは、事前分布を提案とする重点サンプリング*そのもの*であり——[第12章](../12_hierarchical_bayes/)が重点サンプリングを「粗削りなツール」と呼んだのもまさにこの方法だ。粗削りである理由は、事前分布が提案として貧弱なことが多いためだ：サンプルを広く散らばらせ、尤度が少数のサンプルに重みを集中させ、推定がノイジーになる。どのくらいノイジーか？診断指標がある——しかしその前に、学期を通して繰り返し登場する認知的余談を挟む。

### 例示モデルは重点サンプリング器だ

人間のカテゴリ化の最も成功したモデルの一つは、**記憶された例への類似度重み付き投票**によって質問に答える。**例示モデル**（Nosofsky, 1986）では、「とんかつ弁当」の抽象的なルールを持ち歩くのではなく、*覚えている弁当*を持ち歩き、新しい弁当が来たときに各記憶にどれほど似ているか尋ねる：

$$\text{応答}(x) = \frac{\sum_i s(x, x_i) f(x_i)}{\sum_i s(x, x_i)},$$

ここで $s(x, x_i)$ は問い合わせ $x$ と記憶された例 $x_i$ の間の**類似度**、$f(x_i)$ は例 $i$ に記憶された**ラベル**（とんかつなら1、そうでなければ0）だ。

この式をよく見てほしい。これは先ほどの**自己正規化重点サンプリング推定量** $\sum_i w_i f(x_i) / \sum_i w_i$ と**全く同じ**だ——記憶された例が*サンプル*の役割を担い、類似度 $s$ が*重み* $w$ の役割を担う。類似度重み付き投票で記憶から分類する心は、機械的には自分の経験に対して重点サンプリングを実行している。重さからとんかつかどうかを判断する様子を示す：

```python
# Chibany's memory of past bentos: weight (g) and whether it was tonkatsu (1) or not (0).
weights = jnp.array([520., 560., 590., 610., 640., 660., 700., 730.])
is_tonk = jnp.array([0.,   0.,   0.,   1.,   0.,   1.,   1.,   1.])

def exemplar_vote(x, width=40.0):
    s = jnp.exp(-0.5 * ((x - weights) / width) ** 2)   # similarity to each stored example
    return float(jnp.sum(s * is_tonk) / jnp.sum(s))   # similarity-weighted vote

for query in [550.0, 650.0, 720.0]:
    print(f"a {query:.0f} g bento: P(tonkatsu) by exemplar vote = {exemplar_vote(query):.2f}")
```

**出力：**
```
a 550 g bento: P(tonkatsu) by exemplar vote = 0.13
a 650 g bento: P(tonkatsu) by exemplar vote = 0.61
a 720 g bento: P(tonkatsu) by exemplar vote = 0.94
```

軽い弁当は「とんかつではない」に投票し、重い弁当は「とんかつ」に投票し、中間は曖昧になる——重み付けされた記憶だけから生まれる、段階的な般化曲線だ：

![弁当重量の関数としてとんかつである確率のなめらかなS字曲線。550グラムのゼロ近くから720グラムの1近くまで上昇する。非とんかつの例示が青い三角として軽い重量で下部に、とんかつの例示がオレンジの三角として重い重量で上部に位置し、コードの3つの問い合わせ点に0.13、0.61、0.94の値が記されている。](../../images/intro2/mc_exemplar_vote.png)

この橋——*認知の古典的プロセスモデルは変装したモンテカルロ推定量だ*——はこれらの章における最初の例であり、[第17章](../17_particle_filtering/)と[第19章](../19_sampling_the_mind/)ではさらに押し進める。

### 有効サンプルサイズ

少数の重みが支配的になると、1000個の重み付きサンプルは1000個の均等なサンプルの価値がない——ほとんどがほぼゼロの重みしか持たない。**有効サンプルサイズ**は、実際に*影響を与えている*サンプルがいくつあるかを測る。正規化された重み $w_t$（$\sum_t w_t = 1$）を使うと、

$$N_{\text{eff}} = \frac{1}{\sum_{t=1}^{T} w_t^{2}}.$$

2つの極端を読み解こう：

- **完全に均等な重み**（全ての $t$ で $w_t = 1/T$）：$\sum_t w_t^2 = T \cdot (1/T)^2 = 1/T$ なので $N_{\text{eff}} = T$。全てのサンプルが完全に効いている。
- **1つの重みが支配的**（$w_1 \approx 1$、残りは $\approx 0$）：$\sum_t w_t^2 \approx 1$ なので $N_{\text{eff}} \approx 1$。1000個のサンプルが1個分の価値しかない。

つまり $N_{\text{eff}}$ は**重みがどれだけ均等に広がっているかの診断指標**だ——言い換えれば、*提案 $q$ がターゲット $p$ をどれだけよくカバーしているか*だ。うまく合った $q$ はほぼ均等な重みを生み $N_{\text{eff}}$ は $T$ に近くなる；不一致の $q$ は重みを少数のサンプルに崩壊させ $N_{\text{eff}}$ は急落する。2つの提案——一方は $p$ に似た形、もう一方は裾の外れすぎた幅の狭い——でChibanyの弁当ターゲットに対して観察しよう。

<!-- validate: tol=0.3 -->
```python
def ess_of(key, mu_q, sd_q, n=100000):
    x = mu_q + sd_q * jr.normal(key, (n,))
    logw = norm.logpdf(x, mu_p, sd_p) - norm.logpdf(x, mu_q, sd_q)  # log importance weight
    w = jnp.exp(logw - jnp.max(logw)); w = w / jnp.sum(w)           # normalize the weights
    return float(1.0 / jnp.sum(w**2))                              # effective sample size

ess_good = ess_of(jr.key(5), 630.0, 55.0)   # q close to p
ess_poor = ess_of(jr.key(6), 760.0, 30.0)   # q far out in the tail, too narrow
print(f"well-matched q ~ N(630,55):   ESS = {ess_good:8.0f}  out of 100000")
print(f"poorly-matched q ~ N(760,30): ESS = {ess_poor:8.0f}  out of 100000")
```

**出力：**
```
well-matched q ~ N(630,55):   ESS =    95719  out of 100000
poorly-matched q ~ N(760,30): ESS =       22  out of 100000
```

うまく合った提案は100,000サンプルのほぼ全てを保持するが、悪い提案は22個分の価値しか残らない。この対比は並べて見るのが最もわかりやすい——提案がターゲットを外すと、重みのヒストグラムがほぼゼロの塊に崩壊し、少数の巨大な重みに支配される：

![弁当重量ターゲットに対する2つの提案の比較図（2列）。左は提案がターゲットと重なっていて、近似的に均等な重みのコンパクトなヒストグラムと数万の有効サンプルサイズ。右は提案が裾から大きく外れ、少数の巨大な重みがほぼゼロの塊を圧倒し、有効サンプルサイズが数十になっている。](../../images/intro2/mc_weight_variance.png)

それが $N_{\text{eff}}$ の*目的*だ：提案がターゲットをカバーしているかどうかの素早い確認。次は自分の手で感じてみよう——提案をドラッグして、重みとESSがどう反応するか観察しよう：

<iframe src="../../widgets/is-proposal.html"
        width="100%" height="560"
        frameborder="0"
        style="background:#111111; border-radius:6px; margin:1rem 0;"
        title="Interactive importance-sampling proposal explorer with live ESS">
</iframe>

（3つの位置を試してみよう：提案がターゲットの真上、裾から遠く離れた位置、陰影の $W > 700$ 事象に駐在。毎回*両方*の表示を確認しよう。）

{{% notice style="tip" title="診断指標であって、全ての話ではない" %}}
$N_{\text{eff}}$ は*重み*が健全かどうかを示す。高い $N_{\text{eff}}$ が*特定の量の推定*も精確であることを意味するか——そして均等な重みを自明に持つ通常のモンテカルロ法が常により良い選択かどうか——はより鋭くより意外な問いだ。それはまさに**課題の問題3で解き明かす**内容だ。今は $N_{\text{eff}}$ を「重み付きサンプルは現実的な影響を持っているか？」と読んで、パズルを胸のポケットにしまっておこう。
{{% /notice %}}

---

## GenJAX で

上記は全て手で書いたものだ。GenJAX は重点サンプリングを組み込み機能として提供する：モデルの `importance` メソッドが提案から引いて、サンプルを*その対数重みとともに*返す。以前の章で `simulate` と `generate` を使った；`importance` がここでの新しいプリミティブだ。

`model.importance(key, constraint, args)` は `constraint` 内の観測値に**強制的に一致させて**モデルを実行し、未観測の選択をモデルの事前分布からサンプリングして、`(trace, log_weight)` ——手計算で求めるべき重点重みの対数——を返す。観測された表を `True` に拘束し、$\theta$ を事前分布から引き、対数重みを自己正規化すれば、事後平均が導き出される。

<!-- validate: tol=0.02 -->
```python
from genjax import gen, beta as gbeta, flip, ChoiceMap

@gen
def coin_model():
    theta = gbeta(2.0, 2.0) @ "theta"     # prior over the head-probability
    head  = flip(theta) @ "head"          # one coin flip
    return theta

def genjax_snis(key, n):
    keys = jr.split(key, n)
    def one(k):
        tr, lw = coin_model.importance(k, ChoiceMap.d({"head": True}), ())  # observe a head
        return tr.get_choices()["theta"], lw
    thetas, lws = jax.vmap(one)(keys)
    w = jnp.exp(lws - jnp.max(lws)); w = w / jnp.sum(w)   # self-normalize the weights
    return float(jnp.sum(w * thetas))

print(f"GenJAX self-normalized IS posterior mean = {genjax_snis(jr.key(7), 200000):.3f}  (closed form 0.600)")
```

**出力：**
```
GenJAX self-normalized IS posterior mean = 0.600  (closed form 0.600)
```

手書きバージョンと同じ答えが、重みの管理を自動でやってくれる形で得られる。（事前分布ではない*カスタム*提案の場合、GenJAX は任意のサンプルをスコアリングする `model.assess(choices, args)` を提供する——そのスコアリングプリミティブを使って[第18章](../18_markov_chain_monte_carlo/)でMCMCを構築する。）

{{% notice style="success" title="今できること" %}}
**確率**を含む**期待値**——確率は**指示関数**の期待値なので——を、サンプルを引いて平均することで推定できる（**モンテカルロ推定量** $\hat\mu_n$）。その誤差は任意の次元で $1/\sqrt{n}$ のように縮小することを知っている。難しいターゲットを**棄却法**や**逆CDF**でサンプリングでき、それらが失敗したときは**重点サンプリング**ができる：簡単な**提案分布** $q$ から引いて $w = p/q$ で再重み付けし——比を取れば——ターゲットが非正規化でも可能だ。提案がターゲットをカバーしているかどうかの確認として**有効サンプルサイズ** $N_{\text{eff}}$ を読める。

次の[第17章](../17_particle_filtering/)では重点サンプリングを動かす：データが1つずつ届くとき、*昨日の事後分布が今日の事前分布になり*、重み付きサンプルの雲が動くターゲットを追跡する。

*用語集：* [モンテカルロ](../../glossary/#monte-carlo-simulation-), [期待値](../../glossary/#expected-value-), [重点サンプリング](../../glossary/#importance-sampling-), [重点重み](../../glossary/#importance-weight-), [有効サンプルサイズ](../../glossary/#effective-sample-size-), [棄却サンプリング](../../glossary/#rejection-sampling-), [提案分布](../../glossary/#proposal-distribution-).
{{% /notice %}}

---

## 演習

{{% notice style="info" title="自分で試してみよう" %}}
1. **知っている値を推定する。** モンテカルロ推定量を使って $X \sim \text{Uniform}(0,1)$ に対する $\mathbb{E}[X^2]$ を推定しよう（真の値 $1/3$）。$n = 10, 100, 10^3, 10^5$ で推定値を表示する。$n$ を100倍にすると誤差はどのくらい下がるか——$1/\sqrt{n}$ ルールに合っているか？
2. **悪い提案。** 弁当裾ISのセルで、提案を $q = \mathcal{N}(620, 50^2)$ ——ターゲット $p$ と*同じ*——に変えよう。何度か再実行する：シフトした $q$ のときより $P(W > 700)$ の推定はノイジーか？両方の提案について $N_{\text{eff}}$ を計算し、違いを言葉で説明しよう。
3. **自分で指示関数を作る。** $\pi$-ダーツ推定量を書き直して、$x^2 + y^2 \le 1$ **かつ** $y \ge x$（くさび形）の領域の面積を推定しよう。どんな割合を期待するか、そして推定は一致するか？
{{% /notice %}}

付属ノートブックがこれら全てをインタラクティブに解説している：

**📓 [Colab で開く: `16_monte_carlo.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/16_monte_carlo.ipynb)**

---

## 参考文献

- Nosofsky, R. M. (1986). Attention, similarity, and the identification–categorization relationship. *Journal of Experimental Psychology: General, 115*(1), 39–61. <https://doi.org/10.1037/0096-3445.115.1.39>
- Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. *Science, 185*(4157), 1124–1131. <https://doi.org/10.1126/science.185.4157.1124>

---

Special thanks to [JPPCA](https://jpcca.org/) for their generous support of this tutorial series.
