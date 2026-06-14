+++
date = "2026-06-14"
title = "連続体：連続確率"
weight = 2
+++

## 数えることから測ることへ

Chibanyはヒストグラムをじっと見つめています。期待値は理解できました。455gという平均は、500gのとんかつ弁当と350gのハンバーグ弁当の混合として意味をなします。

しかし、まだ気になることがあります。

最初の週の実際の計測値を見てみましょう：
```
Monday:    520g  (tonkatsu)
Tuesday:   348g  (hamburger)
Wednesday: 505g  (tonkatsu)
Thursday:   362g  (hamburger)
Friday:    488g  (tonkatsu)
```

重さはちょうど500gや350gではありません！ばらつきがあります。

そして、より深い問いがあります：**弁当がちょうど505.000000...グラムである確率はいくらか？**

Chibanyは気づきます：チュートリアル1では、離散的な結果を**数える**ことで確率を学びました。しかし、重さは離散的ではありません。**連続**です。340gと520gの間には無限に多くの可能な値があります。

可能性が無限にある場合、どのように確率を割り当てるのでしょうか？

## 離散確率の問題

離散的なアプローチがなぜうまくいかないかを見てみましょう。

**チュートリアル1では**、Chibanyはこの公式を使いました：
$$P(\text{event}) = \frac{\text{# of outcomes in event}}{\text{# of total outcomes}}$$

これは結果が有限個だから成り立ちました：
- 結果空間：{とんかつ、ハンバーグ}
- $P(\text{tonkatsu}) = \frac{1}{2}$（ランダムに選ぶ場合）

**しかし連続的な重さでは**、これは成り立ちません：
- 結果空間：（例えば）340gから520gの間のすべての実数
- $P(\text{weight} = 505g \text{ exactly}) = \frac{1}{\infty} = 0$

**すべての特定の重さの確率がゼロです！**

これはおかしく思えます。Chibanyは確かに505gを観測しました。起きたことがゼロの確率であるはずがないのでしょうか？

{{% notice style="info" title="📘 基礎概念：数えることから測ることへ" %}}
**チュートリアル1から思い出してください**。確率は**数えること**から始まりました：

$$P(A) = \frac{|A|}{|\Omega|} = \frac{\text{outcomes in event}}{\text{total outcomes}}$$

これは{ハンバーグ、とんかつ}のような**離散的な**結果に対して完璧に機能しました。なぜなら：
- 結果を**数える**ことができた（|Ω| = 2）
- 各結果が確率の等しい「割り当て」を受けた（それぞれ1/2）
- 公式が直感的に意味をなした

**しかし重さのような連続的な変数では**、結果を数えることができません：
- 340gと520gの間に無限に多くの可能な値がある
- 無限大で割れない（|Ω| = ∞）
- 数える公式が破綻する

**重要な移行**：**離散的な結果を数える**のではなく、**連続的な面積を測る**ことになります。ロジック自体は同じ（有利なもの / 全体）ですが：
- **離散**：結果を数える → 総数で割る
- **連続**：面積を測る → 全体の面積で割る

**Chibanyの気づき**：「この新しい種類の問題には新しいツールが必要だが、確率の核心的な考え方は変わっていない！」

[← チュートリアル1第3章で数え方のアプローチを復習する](../../intro/03_prob_count/)
{{% /notice %}}

## 解決策：確率密度

解決策は、**正確な値**について問うのをやめ、**範囲**について問うことです。

次の代わりに：
- ❌ 「P(重さ = 505g)はいくらか？」（答え：0）

次のように問います：
- ✅ 「P(500g ≤ 重さ ≤ 510g)はいくらか？」（答え：正の数）

**重要な洞察：** 連続確率では、**数**ではなく**面積**を測ります。

### 確率密度関数（PDF）

**確率密度関数**（PDF）は、異なる値の**相対的な尤度**を表す関数$p(x)$です。

**重要な性質：**
1. $p(x) \geq 0$（すべての$x$で密度は非負）
2. $\int_{-\infty}^{\infty} p(x)   dx = 1$（確率の合計は1）
3. $P(a \leq X \leq b) = \int_a^b p(x)   dx$（確率は**曲線の下の面積**）

**重要：** $p(x)$自体は確率では**ありません**！それは**密度**です。
- $p(x)$は1より大きくなることがある
- $p(x)$の下の面積だけが確率である

{{% notice style="success" title="積分が分からなくても大丈夫！" %}}

**積分（∫）を見たことがなくても心配しないでください！**

こう考えてみましょう：
- **離散**：確率 = 数えること + 割ること
- **連続**：確率 = 曲線の下の面積を測ること

$$\int_a^b p(x)   dx \quad \text{means} \quad \text{"area under } p(x) \text{ from } a \text{ to } b\text{"}$$

GenJAXがこれらの面積を計算してくれます。手で微積分をする必要はありません！

{{% /notice %}}

### PDFと確率の可視化

簡単な例で見てみましょう：0から1の一様分布。

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

# PDF: uniform from 0 to 1
# p(x) = 1 for 0 ≤ x ≤ 1, and 0 otherwise
x = jnp.linspace(-0.5, 1.5, 1000)
pdf = jnp.where((x >= 0) & (x <= 1), 1.0, 0.0)
```

<details>
<summary>可視化コードを表示する</summary>

```python
plt.figure(figsize=(12, 5))

# Plot 1: The PDF itself
plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, 0, pdf, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('p(x)', fontsize=12)
plt.title('Probability Density Function', fontsize=14, fontweight='bold')
plt.ylim(-0.1, 1.3)
plt.grid(alpha=0.3)

# Plot 2: Probability of a range
plt.subplot(1, 2, 2)
plt.plot(x, pdf, 'b-', linewidth=2, alpha=0.3)

# Highlight the region 0.3 ≤ x ≤ 0.7
region_x = x[(x >= 0.3) & (x <= 0.7)]
region_pdf = pdf[(x >= 0.3) & (x <= 0.7)]
plt.fill_between(region_x, 0, region_pdf, color='orange', alpha=0.7,
                 label=f'P(0.3 ≤ X ≤ 0.7) = {0.7-0.3:.1f}')

plt.xlabel('x', fontsize=12)
plt.ylabel('p(x)', fontsize=12)
plt.title('Probability = Area Under Curve', fontsize=14, fontweight='bold')
plt.ylim(-0.1, 1.3)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualization_1.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![コーヒー温度の分布](../../images/intro2/coffee_temperature_histogram.png)


**重要な観察：**
- PDFは高さ1.0で平坦（一様な密度）
- $P(0.3 \leq X \leq 0.7) = \text{area} = \text{width} \times \text{height} = 0.4 \times 1.0 = 0.4$
- $P(X = 0.5 \text{ exactly}) = \text{area of vertical line} = 0$

## 一様分布

最も単純な連続分布は**一様**分布です。

**定義：** 確率変数$X$が$[a, b]$上で一様分布するとは、その範囲内のすべての値が等しく起こりやすいことを意味します。

**PDF：**
$$p(x) = \begin{cases}
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}$$

**直感：** PDFは許容範囲全体で平坦（一定）です。全体の面積が1になるように、高さは$\frac{1}{b-a}$です。

**記法：** $X \sim \text{Uniform}(a, b)$

### 例：一様なコーヒー温度

Chibanyの職場のコーヒーメーカーは信頼できません。朝のコーヒーの温度は60°Cから80°Cの間で一様分布しています。

```python
from genjax import gen, uniform
import jax.numpy as jnp
import jax.random as random

@gen
def coffee_temperature():
    """Model: coffee temperature uniformly between 60 and 80 degrees C"""
    temp = uniform(60.0, 80.0) @ "temp"
    return temp

# Simulate 10,000 cups
key = random.PRNGKey(42)
temps = []

for _ in range(10000):
    key, subkey = random.split(key)
    trace = coffee_temperature.simulate(subkey, ())
    temps.append(trace.get_retval())

temps = jnp.array(temps)
```

<details>
<summary>可視化コードを表示する</summary>

```python
# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(temps, bins=50, density=True, alpha=0.7, color='brown', edgecolor='black')
plt.axhline(1/(80-60), color='red', linestyle='--', linewidth=2,
            label=f'Theoretical PDF: p(x) = 1/20 = {1/20:.3f}')
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title("Chibany's Coffee Temperature (Uniform Distribution)", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.savefig('coffee_temperature_histogram.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![一様分布のPDFとCDF](../../images/intro2/pdf_vs_cdf.png)

<!-- validate: tol=0.02 -->
```python
# Continued from previous code block (requires temps array from above)
# Calculate probabilities for ranges
import jax.numpy as jnp

prob_too_cold = jnp.mean(temps < 65)  # Below 65°C
prob_just_right = jnp.mean((temps >= 70) & (temps <= 75))  # 70-75°C
prob_too_hot = jnp.mean(temps > 75)  # Above 75°C

print(f"P(temp < 65°C) = {prob_too_cold:.3f}")
print(f"P(70°C ≤ temp ≤ 75°C) = {prob_just_right:.3f}")
print(f"P(temp > 75°C) = {prob_too_hot:.3f}")
```

**出力：**
```
P(temp < 65°C) = 0.250
P(70°C ≤ temp ≤ 75°C) = 0.250
P(temp > 75°C) = 0.250
```

**理論的な計算：**
- $P(\text{temp} < 65) = \frac{65-60}{80-60} = \frac{5}{20} = 0.25$ ✓
- $P(70 \leq \text{temp} \leq 75) = \frac{75-70}{80-60} = \frac{5}{20} = 0.25$ ✓
- $P(\text{temp} > 75) = \frac{80-75}{80-60} = \frac{5}{20} = 0.25$ ✓

完璧に一致しています！GenJAXのシミュレーションは理論的な確率を近似します。

## 累積分布関数（CDF）

連続分布を扱う別の方法は、**累積分布関数**（CDF）を通じてです。

**定義：** 確率変数$X$のCDFは：
$$F_X(x) = P(X \leq x) = \int_{-\infty}^x p(t)   dt$$

これは「Xがx以下である確率はいくらか？」を教えてくれます。

**性質：**
1. $F_X(-\infty) = 0$（負の無限大以下である確率は0）
2. $F_X(\infty) = 1$（無限大以下である確率は1）
3. $F_X$は単調非減少（決して下がらない）
4. $P(a \leq X \leq b) = F_X(b) - F_X(a)$（CDF同士の差で確率を求める）

### 一様分布のCDF

$X \sim \text{Uniform}(a, b)$の場合：

$$F_X(x) = \begin{cases}
0 & \text{if } x < a \\
\frac{x-a}{b-a} & \text{if } a \leq x \leq b \\
1 & \text{if } x > b
\end{cases}$$

Chibanyのコーヒーでこれを可視化しましょう：

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Coffee temperature: Uniform(60, 80)
x = jnp.linspace(55, 85, 1000)

# PDF
pdf = jnp.where((x >= 60) & (x <= 80), 1/20, 0.0)

# CDF
cdf = jnp.where(x < 60, 0.0,
        jnp.where(x > 80, 1.0,
                  (x - 60) / 20))
```

<details>
<summary>可視化コードを表示する</summary>

```python
plt.figure(figsize=(12, 5))

# Plot 1: PDF
plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, 0, pdf, alpha=0.3)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('p(x)', fontsize=12)
plt.title('PDF: Probability Density Function', fontsize=14, fontweight='bold')
plt.ylim(-0.01, 0.07)
plt.grid(alpha=0.3)

# Plot 2: CDF
plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'r-', linewidth=2)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('F(x) = P(X ≤ x)', fontsize=12)
plt.title('CDF: Cumulative Distribution Function', fontsize=14, fontweight='bold')
plt.ylim(-0.1, 1.1)
plt.grid(alpha=0.3)

# Mark special points
plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
plt.axvline(70, color='gray', linestyle=':', alpha=0.5)
plt.plot(70, 0.5, 'ro', markersize=8)
plt.text(72, 0.52, 'F(70) = 0.5\nMedian', fontsize=10)

plt.tight_layout()
plt.savefig('pdf_vs_cdf.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![PDF曲線の下の面積としての確率](../../images/intro2/area_under_pdf.png)

**CDFの読み方：**
- x = 70°Cのとき、F(70) = 0.5：「コーヒーの50%は70°C以下」
- x = 65°Cのとき、F(65) = 0.25：「コーヒーの25%は65°C以下」
- x = 75°Cのとき、F(75) = 0.75：「コーヒーの75%は75°C以下」

{{% notice style="info" title="PDFとCDF" %}}

**どちらをいつ使うか？**

**PDF**（$p(x)$）：
- 値の**相対的な尤度**を示す
- 用途：可視化、形状の理解
- $P(a \leq X \leq b) = \int_a^b p(x) dx$（曲線の下の面積）

**CDF**（$F_X(x)$）：
- xまでの**累積確率**を示す
- 用途：計算、パーセンタイル
- $P(a \leq X \leq b) = F_X(b) - F_X(a)$（値の差）

**関係式：** $p(x) = \frac{d}{dx} F_X(x)$（PDFはCDFの微分）

両者は同じ分布を、ただし異なる視点から記述しています！

{{% /notice %}}

## Chibanyの弁当へのつながり

Chibanyの観察を思い出しましょう：弁当の重さはちょうど500gや350gではなく、ばらつきがあります！

今では、このばらつきをモデル化するツールがあります：

1. **とんかつ弁当**：重さは500g周辺で連続的
2. **ハンバーグ弁当**：重さは350g周辺で連続的

しかし、**一様**分布は当てはまりません。なぜでしょうか？

- 一様分布はある範囲内のすべての値が等しく起こりやすいと言う
- しかし、実際には重さが500gや350gの**近く に集中している**
- 中心から離れた値は**起こりにくい**

私たちが必要とする分布は：
- 中心に**ピーク**（最頻値）を持つ
- 中心から離れるにつれて**起こりにくくなる**
- **制御されたばらつき**を持つ（ある弁当は他より変動が大きい）

その分布が**ガウス分布**（正規分布）—あの有名な鐘型曲線です！

それが次の章で学ぶ内容です。

## まとめ

{{% notice style="success" title="第2章まとめ：重要なポイント" %}}

**課題：**
- 重さは**連続的**であり、離散的ではない
- 任意の2点の間に無限に多くの可能な値がある
- すべての特定の値の確率はゼロ！

**解決策：確率密度**
- **PDF** $p(x)$：各点での確率**密度**
- $P(a \leq X \leq b) = \int_a^b p(x) dx$：確率は曲線の下の**面積**
- $p(x)$自体は確率ではない（1より大きくなることがある！）

**一様分布：**
- 最も単純な連続分布
- ある範囲$[a, b]$内のすべての値が等しく起こりやすい
- PDF：$p(x) = \frac{1}{b-a}$（$a \leq x \leq b$の場合）
- CDF：$F_X(x) = \frac{x-a}{b-a}$（$a \leq x \leq b$の場合）

**GenJAXツール：**
- `jnp.uniform(a, b) @ "addr"`：一様分布からサンプリング
- シミュレーションは確率を近似する：$P(\text{event}) \approx \frac{\text{# times event occurs}}{\text{# simulations}}$

**今後の展開：**
- **ピーク**と**制御されたばらつき**を持つ分布が必要
- **ガウス分布**（正規分布）の登場
- 自然界のばらつきをモデル化する鐘型曲線！

{{% /notice %}}

## 練習問題

### 問題1：待ち時間

Chibanyのバスは午前8時から8時20分の間に一様に到着します。バスが到着する確率はいくらですか：
- a) 8時5分より前？
- b) 8時10分から8時15分の間？
- c) 8時18分より後？

{{% expand "答え" %}}

モデル：$X \sim \text{Uniform}(0, 20)$（$X$ = 午前8時からの分数）

**a) P(X < 5)：**
$$P(X < 5) = \frac{5-0}{20-0} = \frac{5}{20} = 0.25$$
バスが8時5分より前に到着する確率は**25%**。

**b) P(10 ≤ X ≤ 15)：**
$$P(10 \leq X \leq 15) = \frac{15-10}{20-0} = \frac{5}{20} = 0.25$$
バスが8時10分から8時15分の間に到着する確率は**25%**。

**c) P(X > 18)：**
$$P(X > 18) = \frac{20-18}{20-0} = \frac{2}{20} = 0.10$$
バスが8時18分より後に到着する確率は**10%**。

{{% /expand %}}

### 問題2：GenJAXシミュレーション

問題1のGenJAX生成関数を書き、10,000回のバス到着をシミュレーションしてください。経験的な確率が理論値と一致することを確認してください。

{{% expand "答え" %}}

<!-- validate: tol=0.01 -->
```python
from genjax import gen, uniform
import jax.numpy as jnp
import jax.random as random

@gen
def bus_arrival():
    """Bus arrives uniformly between 0 and 20 minutes after 8:00 AM"""
    arrival_time = uniform(0.0, 20.0) @ "arrival"
    return arrival_time

# Simulate 10,000 arrivals
key = random.PRNGKey(123)
arrivals = []

for _ in range(10000):
    key, subkey = random.split(key)
    trace = bus_arrival.simulate(subkey, ())
    arrivals.append(trace.get_retval())

arrivals = jnp.array(arrivals)

# Calculate empirical probabilities
prob_before_5 = jnp.mean(arrivals < 5)
prob_10_to_15 = jnp.mean((arrivals >= 10) & (arrivals <= 15))
prob_after_18 = jnp.mean(arrivals > 18)

print(f"a) P(before 8:05): {prob_before_5:.3f} (theoretical: 0.250)")
print(f"b) P(8:10 to 8:15): {prob_10_to_15:.3f} (theoretical: 0.250)")
print(f"c) P(after 8:18): {prob_after_18:.3f} (theoretical: 0.100)")
```

**出力：**
```
a) P(before 8:05): 0.248 (theoretical: 0.250)
b) P(8:10 to 8:15): 0.252 (theoretical: 0.250)
c) P(after 8:18): 0.099 (theoretical: 0.100)
```

ほぼ一致しています！小さな差異はランダムサンプリングによるものです。

{{% /expand %}}

### 問題3：ゼロ確率が不可能を意味しない理由

$P(X = 505.0 \text{ exactly}) = 0$であるなら、Chibanyがちょうど505.0gの弁当を観測したことはどうして可能なのでしょうか？

{{% expand "答え" %}}

**重要な洞察：** 「確率ゼロ」は連続分布において「不可能」を意味しません！

**説明：**
1. 理論的には、重さは無限の精度を持つ**実数**です
2. $P(X = 505.00000...)= 0$なのは、無限に多くの値の中の1点だからです
3. 実際には、Chibanyの秤は**有限の精度**を持ちます（例：±0.1g）
4. 実際に観測したのは：$P(504.95 \leq X \leq 505.05) > 0$（小さな範囲！）

**アナロジー：** ダーツボードにダーツを投げる
- $P(\text{hit exact point (x,y)}) = 0$（無限の精度）
- しかしどこかに当たる（ある小さな領域）
- その領域は正の面積を持ち、したがって正の確率を持つ

**数学的な区別：**
- **確率ゼロ** ≠ 不可能（ただし無限に起こりにくい）
- **不可能** = 分布のサポートにない

例：Uniform(60, 80)の場合、$P(X = 90)$は単にゼロなのではなく—90はその範囲にすら入らないので不可能です！

{{% /expand %}}

---

**次へ：** [第3章 - ガウス分布 →](./03_gaussian.md)

いよいよ鐘型曲線と出会い、それが自然界のあちこちに見られる理由を理解します！
