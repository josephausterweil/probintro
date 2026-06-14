+++
date = "2026-06-14"
title = "連続的概念とシェパードの法則"
weight = 3
+++

## 入門3: 連続的概念と長方形ゲーム

これまでの仮説空間はすべて**有限リスト** ── §4〜§6の7つの数値規則 ── でした。だからこそ*列挙*できたのです。すべての規則にスコアを付け、正規化するだけで完了です。しかし実際の概念の多くは連続スケール上に存在します。「健康的な血糖値」「快適な室温」「だいたいお昼ごろ」── これらはそれぞれある軸上の**区間**であり、候補となる区間は*無限に*あります。このフレームワークはそれでも機能するのでしょうか？

機能します。そしてほとんど何も変わりません。テネンバウムの**長方形ゲーム**がこれを具体的に示しています。ある性質が未知の区間内（2次元では未知の長方形内）のアイテムに当てはまるとします。その性質を持つとわかっているいくつかの例を見て、*どの位置*にその性質があるかを判断しなければなりません。仮説空間 $\mathcal{H}$ は今や「すべての区間 $[\text{lo}, \text{hi}]$」となり、仮説の大きさ $|h|$ はその**長さ** $\text{hi} - \text{lo}$ です。強サンプリング尤度はそのままで、長さが集合のサイズの役割を果たします。長さ $L$ の区間は各例を $1/L$ の確率にするので、$n$ 個の例の確率は $(1/L)^n$ になります。サイズ原理はそのまま引き継がれます ── *短い*区間は密なクラスターの例に長い区間よりはるかによく適合します。

![2次元長方形ゲーム: いくつかの黄色い点が入れ子状の候補長方形の中に並んでいる。点を囲む狭い長方形ほど明るく描かれ（事後確率が高い）、点をはるかに超えて広がる緩い長方形は暗い。点のスパンには $r$（データの範囲）とラベルが付いており、候補長方形がそのスパンを超えて延びる余分な距離には $d$ とラベルが付いている。](../../../images/intro2/cc_2d.png)

上の図は2次元版（Tenenbaum, 1999）です。点が観測された例であり、**それらをすべて囲む**すべての長方形が候補仮説です。サイズ原理により、*最小の*囲む長方形が最も大きな事後確率の重みを得て（明るく表示されます）、汎化がデータの近傍に集中します。人のデータと比較する際に重要になる2つの量があります。$r$（点が広がる範囲）と $d$（長方形または人の判断がその範囲をどれだけ超えて延びるか）です。1次元の区間はこれを1つの軸で行ったものであり、以下で実際に計算するケースです。

{{% notice style="info" title="仮説が無限にある場合はグリッドを使う" %}}
すべての実数区間を文字通り列挙することはできませんが、その必要はありません。候補の端点の細かい**グリッド**を敷き、その間の区間を列挙します ── §5の「すべての仮説にスコアを付けて正規化する」動作と全く同じで、手書きのリストではなくグリッド上で行うだけです。グリッドを細かくすれば、答えは連続的なものに収束します。（これは第2章の「曲線の下の面積」と同じ精神です。細かい離散化によって連続量を近似します。）
{{% /notice %}}

### 区間学習器の構築

コードは §5/§6 の列挙であり、数値規則のリストから区間のグリッドに適応されています。密なクラスターの例 ── たとえば位置9、10、11 ── を観測し、**汎化勾配**と呼ばれるものを計算します。グリッド上のすべての位置 $y$ について、それを含む区間の事後確率加重投票を求めます。（これは §5 の数値ごとの投票の正確な連続アナログです ── 同じ $\sum_h \mathbf{1}[y\in h] \cdot p(h\mid X)$ を、クエリ位置 $y$ の軸全体に渡って計算します。）

```python
import jax
import jax.numpy as jnp

# A grid of candidate endpoints (and query positions) along a 1-D axis, 0 to 20, step 0.25.
GRID = jnp.linspace(0.0, 20.0, 81)

def all_intervals(grid):
    """Every interval [lo, hi] with endpoints on the grid and hi > lo."""
    # jnp.meshgrid builds every combination of a lo-endpoint and a hi-endpoint;
    # .reshape(-1) then flattens those grids into two long parallel lists of lo's and hi's.
    # (indexing="ij" just fixes the row/column convention; the order doesn't matter here
    # because we flatten both grids anyway.)
    los, his = jnp.meshgrid(grid, grid, indexing="ij")
    lo = los.reshape(-1)
    hi = his.reshape(-1)
    proper = hi > lo                                      # True for real intervals (hi above lo)
    # Indexing an array with a boolean mask keeps only the entries where the mask is True --
    # here, only the proper intervals.
    return lo[proper], hi[proper]

def gradient(observed, exp_rate=None, grid=GRID):
    """Generalization gradient over the grid, by enumerating interval hypotheses.

    exp_rate=None gives a flat/uniform prior over intervals; a number gives an
    exponential prior on interval length (see the next section).
    """
    lo, hi = all_intervals(grid)
    length = hi - lo                              # |h| for an interval is its length
    n = observed.shape[0]

    # Which intervals contain ALL observed examples? (consistent, like §5/§6)
    contains_all = jnp.ones(lo.shape, dtype=bool)
    for x in observed:
        contains_all = contains_all & (lo <= x) & (x <= hi)

    # Strong-sampling likelihood (1/length)^n, in log space; inconsistent intervals -> log 0.
    log_like = jnp.where(contains_all, -n * jnp.log(length), -jnp.inf)
    # Prior over intervals: flat for now (exponential prior comes next section).
    log_prior = jnp.zeros(lo.shape) if exp_rate is None else -exp_rate * length

    log_post = log_prior + log_like
    log_post = log_post - jnp.max(log_post)      # numerical stability, as before
    post = jnp.exp(log_post)
    post = post / post.sum()

    # Posterior-weighted vote at each query position y: sum the posterior of intervals containing y.
    def vote(y):
        contains_y = (lo <= y) & (y <= hi)
        return jnp.sum(jnp.where(contains_y, post, 0.0))
    return jax.vmap(vote)(grid)

observed = jnp.array([9.0, 10.0, 11.0])
g = gradient(observed)                            # flat prior over intervals

for y in [10.0, 12.0, 13.0, 15.0, 18.0]:
    i = int(jnp.argmin(jnp.abs(GRID - y)))        # nearest grid point to y
    print(f"  g({y:4.1f}) = {round(float(g[i]), 3)}")
```

**出力:**
```
  g(10.0) = 1.0
  g(12.0) = 0.545
  g(13.0) = 0.339
  g(15.0) = 0.15
  g(18.0) = 0.039
```

勾配は**観測されたクラスター全体で1.0のフラット**な値を示し（すべての整合区間はデータ内の位置を含みます）、その後 **$y$ が離れるにつれて滑らかに減衰します** ── 汎化は距離とともに低下し、まさにシェパードが測定した挙動です。そして数値ゲームと*同じ*事後確率加重投票です。変わったのは $\mathcal{H}$ だけで、集合のリストから区間のグリッドになりました。

### モデルから現れるシェパードの法則

§3 からの約束を思い出してください。シェパードは汎化が距離とともに*指数関数的に*減衰することを発見し、合理的な学習者が*そのような*指数関数を生み出すべきであることを解析的に示しました。ここでは彼の解析的証明を再現せず、代わりに区間モデルから指数関数が現れることを**計算的に実証**します ── 指数関数をどこにも組み込まなかったにもかかわらず、コードが生成する勾配は（近似的に）指数関数的です。この確認には除算以上のものは何も必要ありません。*指数関数的*減衰の特徴は**一定の比率**です。データからの各固定ステップで勾配が同じ係数（例えば距離1単位あたり $e^{-1}$）倍されます。代わりに勾配が線形またはベル曲線状に減衰するなら、ステップごとの比率は変動するはずです。そこで、データの端から外側に歩き出し、各値と直前の値の比率を表示してみましょう。

```python
import jax.numpy as jnp

# Walk outward from the edge of the data (which ends at position 11) and, at each step,
# print g and its ratio to the previous step. A roughly CONSTANT ratio == exponential decay.
print("distance past data | g      | ratio to previous")
previous = None
for d in range(1, 7):
    y = 11.0 + d
    i = int(jnp.argmin(jnp.abs(GRID - y)))        # nearest grid point to y
    g_d = float(g[i])
    ratio = "  (first point)" if previous is None else f"{g_d / previous:.3f}"
    print(f"        {d}          | {g_d:.4f} | {ratio}")
    previous = g_d
```

**出力:**
```
distance past data | g      | ratio to previous
        1          | 0.5452 |   (first point)
        2          | 0.3387 | 0.621
        3          | 0.2230 | 0.659
        4          | 0.1503 | 0.674
        5          | 0.1009 | 0.671
        6          | 0.0656 | 0.650
```

1ステップ外に進むたびに、勾配はほぼ同じ係数（〜0.65）倍されます ── これはほぼ一定の比率であり、**指数関数的減衰**の特徴です。（0.62〜0.67の小さな揺れは有限グリッドの離散化アーティファクトであり、指数関数からの逸脱ではありません。グリッドを細かくすれば安定します。）モデルには指数関数を一切組み込みませんでした。区間の仮説空間と強サンプリング尤度を仮定しただけです。シェパードの普遍法則がベイズ的な区間上の汎化から*現れた*のです。これは §3 の約束の計算的対応物です。指数勾配は私たちが組み込んだ仮定ではなく、モデルの**帰結**です ── ここでは実証的に示され、理想化された連続ケースについては（シェパードが行ったように）解析的に証明可能です。

### 一つの欠点と一つの修正: 指数事前分布

落とし穴があります。そしてそれはテネンバウムがこのモデルを人間のデータと比較したときに発見したものと同じです。区間に対する**平坦な**事前分布 ── すべての区間の長さが事前に等しく確からしい ── では、モデルは**過剰拡張**します。非常に長い区間に対して頑固な量の確信を保ち続け、勾配の減衰が遅すぎてデータから遠くの性質を過剰に予測します。人はこうしません。人はより密に汎化します。

![人間と平坦事前分布モデルがどれだけ汎化するかを、例がどれだけ広がっているかに対してプロット。横軸は $r$（例が広がる範囲）、縦軸は $d$（汎化がその範囲をどれだけ超えて延びるか）。例の数 $n$ ごとに1本の曲線が示されている。人間の曲線は上昇後に平らになり、十分なデータがあると人は拡張を止めるが、平坦事前分布モデルの曲線は人間の曲線をはるかに超えて上昇し続ける。特に例が少ない場合や広く分散している場合にそれが顕著である。](../../../images/intro2/tg_results.png)

上の図（Tenenbaum, 1999 のデータ）は、例の数 $n$ ごとに1本の曲線で $d$ を $r$ に対してプロットしたものです。**人間**と**平坦事前分布モデル**の比較です。人間の汎化は飽和します ── ある点を超えると、広がりが増えても人間はそれほど遠くまで拡張しなくなります ── しかし平坦事前分布モデルは外側に達し続けます。2つの曲線の間のギャップが修正する必要のある過剰拡張です。

修正は区間の長さに対するより良い**事前分布**です。長い区間は短い区間よりも事前に確からしくないべきであり、自然な選択は**指数分布** ── この章で初めて登場する真に新しい分布です。

{{% notice style="info" title="指数分布（使用前の定義）" %}}
**指数分布**は単一の非負の数 $s \ge 0$（ここでは区間の長さ）に対する確率分布です。その密度は

$$p(s) = \lambda e^{-\lambda s}, \qquad s \ge 0,$$

1つのパラメータ、**率** $\lambda > 0$ を持ちます。$s$ の小さな値が最も確からしく、$s$ が増えるにつれて密度が ── 指数関数的に ── 落ちると読みます。平均は $1/\lambda$ なので、大きな率 $\lambda$ は典型的な値をより小さく引き寄せます（短い区間をより強く優遇します）。これは §3 で出会った $e^{-(\text{何か})}$ 減衰形と同じであり、今は誠実な確率分布として機能しています（$s \ge 0$ 上で積分して1になります）。コードでは対数 $\log p(s) = \log \lambda - \lambda s$ だけが必要です。定数 $\log \lambda$ は事後確率の正規化の際に消えるので、事前分布は $-\lambda s$ だけを寄与します ── これはまさに上の `gradient` 関数の中にすでに存在する `-exp_rate * length` の行です。
{{% /notice %}}

ですから新しいコードも必要ありません ── 事前分布をオンにするだけです。平坦事前分布の勾配と指数事前分布のもの（率 $\lambda = 0.5$）を比較してみましょう。

```python
import jax.numpy as jnp

observed = jnp.array([9.0, 10.0, 11.0])
g_flat = gradient(observed, exp_rate=None)     # flat prior over intervals
g_exp  = gradient(observed, exp_rate=0.5)      # exponential prior on length, rate 0.5

print("       distance:   +1     +2     +4     +7")
for label, grad in [("flat ", g_flat), ("exp  ", g_exp)]:
    vals = []
    for y in [12.0, 13.0, 15.0, 18.0]:
        i = int(jnp.argmin(jnp.abs(GRID - y)))
        vals.append(round(float(grad[i]), 3))
    print(f"  {label} prior:  {vals[0]:<6} {vals[1]:<6} {vals[2]:<6} {vals[3]:<6}")
```

**出力:**
```
       distance:   +1     +2     +4     +7
  flat  prior:  0.545  0.339  0.15   0.039
  exp   prior:  0.263  0.086  0.013  0.001
```

指数事前分布はデータ外のすべての予測値を引き下げます ── 勾配は今や過剰に届かず、**データに密着**します。その締まった曲線が人間の汎化に一致するものです。以下は私たち自身のコードで計算された2つの勾配を並べたものです。

![同じ軸上の2つの1次元汎化勾配。どちらも観測された例（中央付近にマーク）の上で平らにピークを迎え、両側で減衰する。平坦事前分布の曲線はゆっくり減衰し、汎化をデータから遠くまで広げる。指数事前分布の曲線はずっと速く減衰し、観測されたクラスターに密着する。指数事前分布の曲線が人間の汎化の仕方に一致するものである。](../../../images/intro2/generalization_gradient_1d.png)

そして実際の挙動に対する見返りがここにあります。指数事前分布を加えると、モデルの $d$-対-$r$ 曲線が折れ曲がり、**人間の曲線の上に着地します** ── 前の図の過剰拡張が消えました。

![以前と同じ人間とモデルの $d$-対-$r$ プロットですが、今回はモデルが区間サイズに対する指数事前分布を持ちます。以前は人間の曲線をはるかに超えて上昇していたモデルの曲線が、今や折れ曲がって飽和し、例の数 $n$ すべてについて人間の曲線のほぼ真上に座っています。尤度（サイズ原理）と事前分布が一緒になって人間の汎化を再現します。どちらか一方だけでは不十分です。](../../../images/intro2/tg_results_prior.png)

このフィットはテネンバウム（1999）が報告したものです。サイズ原理が*順序付け*を供給し（例が少ない → より遠くまで汎化）、指数事前分布が*飽和*を供給します（無限に拡張しない）。どちらの要素も単独では人間に一致せず、一緒にすれば一致します。

{{% notice style="success" title="長方形ゲームが加えるもの" %}}
連続的概念に対して*フレームワーク*は何も変わりませんでした ── 同じ事後確率加重投票、同じ強サンプリングのサイズ原理を、リストの代わりにグリッド上の列挙で計算します。しかし2つの新しく重要な成果があります。第一に、**シェパードの指数法則はモデルから仮定なしに現れます** ── コードが生成する勾配はデータから各ステップ離れるごとにほぼ一定の係数で減衰し、指数関数的減衰の特徴です（ここでは計算的に示され、理想化されたケースでは解析的に証明可能です）。第二に、**事前分布が重要です**。平坦な事前分布は過剰拡張し、サイズに対する指数事前分布は汎化を引き戻して人間に一致させます。その思いを胸に ── *事前分布は実際の働きをしている* ── 章の最後の問いへと進みます。仮説空間（とその事前分布）はどこから来るのか、そしてそれを誤って選んだらどうなるのか？ それがノーフリーランチです。
{{% /notice %}}
