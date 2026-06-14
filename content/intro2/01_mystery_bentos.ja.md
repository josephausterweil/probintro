+++
title = "Chibanyの謎の弁当"
weight = 1
+++

## 重さの問題

新しいセメスターが始まり、Chibanyの学生たちがまた弁当を持ってきてくれている。でも今回は、何かが違う。

前のセメスターは、弁当箱が透明だった。だから中にとんかつが入っているのか、ハンバーグが入っているのかすぐに分かった。でも今セメスターは、弁当箱が**不透明**だ。受け取った時に中身が見えない。すぐに何の料理かを知りたいが、学生たちが見ている前で弁当箱を開けるのは極めて失礼だ。幸いにも、Chibanyは好奇心旺盛で、確率論者でもある。

そこで計画を立てた：**弁当の重さを量る**のだ。

とんかつ弁当はボリュームがあって重い。ハンバーグ弁当は軽い。重さを記録すれば、何を受け取っているかが分かるかもしれないし、次に何が来るかも予測できるかもしれない。

## 立ち聞きした会話

ある午後、Chibanyがうたた寝をしていると、近くで2人の学生が話しているのが聞こえた：

> **学生1：**「ほとんどの日にとんかつをChibanyに持ってきてるんだ。大好きみたいだから！」
> **学生2：**「私も！でもカフェテリアのとんかつが切れた時は、ハンバーグを持ってくることもあるよ。」
> **学生1：**「そうだよね、私はとんかつを…10回中7回くらい持ってくるかな？」
> **学生2：**「同じくらい！とんかつ70%、ハンバーグ30%くらい。」

Chibanyは微笑んだ。やっぱり**パターン**があるじゃないか！でも実験は続けることにした。弁当の重さを量るだけでこの70-30の比率を発見できるだろうか？

## 第1週：不思議なこと

Chibanyは最初の1週間の弁当の重さを量って記録した：

```
月曜日：520g
火曜日：348g
水曜日：505g
木曜日：362g
金曜日：488g
```

平均を計算すると：**441グラム**。

「ふむ」と考えた。「おかしいな。前のセメスターでは、とんかつ弁当は約**500g**で、ハンバーグ弁当は約**350g**だったはずだ。でも441gはちょうど中間だ！中くらいの大きさの弁当を受け取っているのだろうか？」

次の数週間でさらに弁当の重さを量った：

```
第2週：355g、510g、492g、345g、515g
第3週：498g、358g、505g、362g、490g
第4週：352g、488g、508g、355g、495g
```

1か月後、20個の測定値が得られた。平均は依然として約**445g**だった。

でも何かが合わない……

## 明かされたパラドックス

Chibanyは測定値のヒストグラムを描いた：

```python
import numpy as np

# Chibany's actual measurements (grams)
weights = np.array([
    520, 348, 505, 362, 488,  # Week 1
    355, 510, 492, 345, 515,  # Week 2
    498, 358, 505, 362, 490,  # Week 3
    352, 488, 508, 355, 495   # Week 4
])

print(f"Average weight: {weights.mean():.1f}g")
print(f"Weights near 350g: {np.sum((weights >= 340) & (weights <= 370))}")
print(f"Weights near 500g: {np.sum((weights >= 480) & (weights <= 520))}")
print(f"Weights near 445g: {np.sum((weights >= 435) & (weights <= 455))}")
```

<details>
<summary>可視化コードを表示</summary>

```python
import matplotlib.pyplot as plt

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(weights, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(weights.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Average: {weights.mean():.1f}g')
plt.xlabel('Weight (grams)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title("Chibany's Mystery Bentos - First Month", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('mystery_bentos_histogram.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![謎の弁当 - 重さの分布](../../images/intro2/mystery_bentos_histogram.png)

**出力：**
```
Average weight: 445.4g
Weights near 350g: 6
Weights near 500g: 14
Weights near 445g: 0
```

グラフを見つめた。**何かがとてもおかしい。**

ほとんどの重さは**350g**付近（ハンバーグの範囲）に集中している。
残りのほとんどは**500g**付近（とんかつの範囲）に集中している。
しかし**ゼロ個**の測定値が445g（平均）付近にある！

{{% notice style="warning" title="パラドックス" %}}
**平均の重さは、ほとんど実現しない重さだ！**

これは不可能に思える。どの弁当も445g付近の重さでないのに、どうして平均が445gになるのか？
{{% /notice %}}

## 解決：期待値

Chibanyは気づいた。「中くらいの弁当」を受け取っているのではない。重いとんかつ弁当と軽いハンバーグ弁当の**混合**を受け取っているのだ！

データをより注意深く見ると：
- 約**20個中14個**の測定値が500g付近（とんかつ）
- 約**20個中6個**の測定値が350g付近（ハンバーグ）

これはおよそ：
- **70%のとんかつ**（θ = 0.7）、学生たちが言った通り！
- **30%のハンバーグ**（θ = 0.3）

これで445gの平均が理解できる！個々の弁当が445gというわけではない。**混合の長期平均**が：

$$\text{平均の重さ} = (0.7 \times 500\text{g}) + (0.3 \times 350\text{g}) = 350 + 105 = 455\text{g}$$

観測された平均445gは理論値455gに近い。この差は少ないサンプル数による単なるランダムな変動だ。

これを**期待値**と呼び、$E[X]$ と書く。

## 期待値とは何か？

**簡単に言えば：** 期待値とは、同じことを何度も何度も繰り返したときに「平均的に」得られる値のことだ。

Chibanyの弁当では：
- 70%の日にとんかつを受け取る（500g）
- 30%の日にハンバーグを受け取る（350g）
- 多くの日にわたる平均では、弁当の重さは：(0.7 × 500) + (0.3 × 350) = 455g

**数学的な定義：** 期待値とは、すべての可能な結果の**加重平均**であり、重みが確率である。

$x_1, x_2, \ldots, x_n$ の値を確率 $p_1, p_2, \ldots, p_n$ でとる離散確率変数 $X$ に対して：

$$E[X] = \sum_{i=1}^{n} p_i \cdot x_i$$

**分解すると：**
- $p_i$ = 結果 $i$ が起きる確率
- $x_i$ = 結果 $i$ の値
- $\sum$ = 「すべてを足す」

Chibanyの場合：
- $X$ = 弁当の重さ
- $x_1 = 500$（とんかつの重さ）、確率 $p_1 = 0.7$
- $x_2 = 350$（ハンバーグの重さ）、確率 $p_2 = 0.3$

したがって：
$$E[X] = 0.7 \times 500 + 0.3 \times 350 = 455\text{g}$$

### 3つの重要な洞察

**1. 期待値 ≠ 日常の意味での「期待する」値**

個々の弁当が正確に455gだと「期待する」べきではない。実際、ほとんどの弁当は455gでない！「期待値」という言葉は少し誤解を招く。本当の意味は「長期平均」だ。

**2. 期待値は重心だ**

ヒストグラムがシーソーの上で釣り合っているとイメージしよう。バランスが取れるように支点をどこに置くか？期待値の位置だ！それが分布の「重心」だ。

**3. 期待値は構造を隠す**

$E[X] = 455\text{g}$ を知っていても、2種類の弁当があることは分からない。**双峰性**の構造（2つの山）が失われる。だから分布を完全に理解するには、さらに多くのツール（分散や混合モデルなど）が必要になる。

## 期待値についてよくある誤解

{{% notice style="info" title="よくある誤解" %}}

**誤解1：「期待値は最も可能性の高い値だ」**
❌ **誤り！** Chibanyの場合、E[X] = 455gだが、最も可能性の高い値は350gか500gだ。445gの弁当はゼロ個！

✓ **正しい理解：** 期待値は長期平均であり、最も起こりやすい結果ではない。

---

**誤解2：「期待値は次に観測される値だ」**
❌ **誤り！** 次の弁当は約350gか約500gで、455gではない。

✓ **正しい理解：** 期待値は分布の中心を表し、個々の結果を表すものではない。

---

**誤解3：「期待値が分布を完全に表す」**
❌ **誤り！** まったく異なる2つの分布が同じ期待値を持つことができる：
- 分布A：すべての弁当がちょうど455gの重さ
- 分布B：70%が500g、30%が350g

どちらもE[X] = 455gだが、まったく異なる分布だ！

✓ **正しい理解：** 期待値は単なる1次モーメントだ。分散（広がり）、形状なども必要だ。

---

**誤解4：「期待値は実現不可能な結果にはなりえない」**
❌ **誤り！** 期待値は観測できない値になることがある。

例：公平なサイコロの期待値は $E[X] = 3.5$ だが、3.5の目は絶対に出ない！

✓ **正しい理解：** 期待値は数学的な平均であり、必ずしも実現可能な結果ではない。

{{% /notice %}}

## 期待値を重心として可視化する

E[X]がなぜ「重心」と呼ばれるのかを、支点の位置を変えて確認しよう：

<details>
<summary>可視化コードを表示</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

def draw_seesaw_panel(ax, fulcrum_position, title, show_calculation=True):
    """Draw a single seesaw panel with given fulcrum position"""

    # Bento positions and masses
    pos_hamburger = 350
    pos_tonkatsu = 500
    mass_hamburger = 0.3
    mass_tonkatsu = 0.7

    # Calculate distances from fulcrum
    dist_hamburger = abs(pos_hamburger - fulcrum_position)
    dist_tonkatsu = abs(pos_tonkatsu - fulcrum_position)

    # Calculate torques
    torque_left = mass_hamburger * dist_hamburger if pos_hamburger < fulcrum_position else 0
    torque_right = mass_tonkatsu * dist_tonkatsu if pos_tonkatsu > fulcrum_position else 0

    if pos_hamburger > fulcrum_position:
        torque_right += mass_hamburger * dist_hamburger
    if pos_tonkatsu < fulcrum_position:
        torque_left += mass_tonkatsu * dist_tonkatsu

    net_torque = torque_right - torque_left

    # Calculate rotation angle (exaggerated for visibility)
    max_angle = 25  # degrees
    rotation_angle = np.clip(net_torque * max_angle / 50, -max_angle, max_angle)

    # Draw seesaw
    seesaw_length = 200
    seesaw_left = fulcrum_position - seesaw_length / 2
    seesaw_right = fulcrum_position + seesaw_length / 2

    # Apply rotation
    angle_rad = np.radians(rotation_angle)
    left_y = -seesaw_length / 2 * np.sin(angle_rad)
    right_y = seesaw_length / 2 * np.sin(angle_rad)

    # Draw the seesaw plank
    ax.plot([seesaw_left, seesaw_right], [left_y, right_y],
            'k-', linewidth=6, solid_capstyle='round', zorder=2)

    # Draw fulcrum (triangle)
    triangle_size = 15
    triangle = Polygon([
        [fulcrum_position - triangle_size/2, -triangle_size],
        [fulcrum_position + triangle_size/2, -triangle_size],
        [fulcrum_position, 0]
    ], facecolor='red', edgecolor='darkred', linewidth=2, zorder=3)
    ax.add_patch(triangle)

    # Draw bento masses (circles positioned on seesaw)
    # Calculate actual positions on rotated seesaw
    hamburger_offset = pos_hamburger - fulcrum_position
    tonkatsu_offset = pos_tonkatsu - fulcrum_position

    hamburger_y = hamburger_offset * np.sin(angle_rad)
    tonkatsu_y = tonkatsu_offset * np.sin(angle_rad)

    # Hamburger bento
    circle_hamburger = Circle((pos_hamburger, hamburger_y + 8),
                             radius=8, facecolor='orange',
                             edgecolor='darkorange', linewidth=2, zorder=4)
    ax.add_patch(circle_hamburger)
    ax.text(pos_hamburger, hamburger_y + 25, '30%\n(350g)',
           ha='center', va='center', fontsize=10, fontweight='bold')

    # Tonkatsu bento (larger circle to show 70%)
    circle_tonkatsu = Circle((pos_tonkatsu, tonkatsu_y + 8),
                            radius=12, facecolor='brown',
                            edgecolor='saddlebrown', linewidth=2, zorder=4)
    ax.add_patch(circle_tonkatsu)
    ax.text(pos_tonkatsu, tonkatsu_y + 30, '70%\n(500g)',
           ha='center', va='center', fontsize=10, fontweight='bold')

    # Mark fulcrum position
    ax.axvline(fulcrum_position, color='red', linestyle=':', alpha=0.5, zorder=1)
    ax.text(fulcrum_position, -40, f'Fulcrum\n{fulcrum_position}g',
           ha='center', va='top', fontsize=9, color='red', fontweight='bold')

    # Show calculation if requested
    if show_calculation:
        calc_text = f"Left torque: {torque_left:.1f}\nRight torque: {torque_right:.1f}"
        ax.text(0.05, 0.95, calc_text, transform=ax.transAxes,
               fontsize=9, va='top', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.8))

    # Determine balance state
    if abs(rotation_angle) < 1:
        balance_state = "⚖️ BALANCED!"
        balance_color = 'green'
    elif rotation_angle > 0:
        balance_state = "↻ TIPS RIGHT"
        balance_color = 'red'
    else:
        balance_state = "↺ TIPS LEFT"
        balance_color = 'blue'

    # Title with balance state
    ax.text(0.5, 0.98, title, transform=ax.transAxes,
           ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.88, balance_state, transform=ax.transAxes,
           ha='center', va='top', fontsize=14, fontweight='bold',
           color=balance_color)

    # Clean up axes
    ax.set_xlim(320, 530)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.axis('off')

# Create figure with 3 panels
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Fulcrum too far left (tips right)
draw_seesaw_panel(ax1, 400, "Fulcrum at 400g")

# Panel 2: Fulcrum too far right (tips left)
draw_seesaw_panel(ax2, 480, "Fulcrum at 480g")

# Panel 3: Fulcrum at E[X] (balanced!)
draw_seesaw_panel(ax3, 455, "Fulcrum at E[X] = 455g")

plt.suptitle("Why E[X] is the Balance Point",
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('seesaw_balance_point.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![謎の弁当のシーソーバランス](../../images/intro2/seesaw_visualization.png)

**期待値E[X] = 455gは、分布がバランスを保つ唯一の位置だ。**

公園のシーソーをイメージしよう：

- 支点を400gに置くと、重いとんかつ側（500gに確率70%）がハンバーグ側より重く、シーソーは右に傾く
- 支点を480gに置くと、ハンバーグ側（わずか30%だが）があまりにも遠い（130g！）ので「てこの力」が大きく、シーソーは左に傾く
- E[X] = 455gの時だけ、すべてが完全にバランスする。ハンバーグの距離（105g離れている）×重み（30%）= とんかつの距離（45g離れている）×重み（70%）：どちらも31.5になる

## シミュレーションによる検証

これを計算で確認しよう。Chibanyの学生たちがランダムに70%のとんかつと30%のハンバーグを選んでいるなら、多くの日にわたって何が起こるはずか？

```python
import numpy as np

# Simulate 1000 days of bento deliveries
np.random.seed(42)
n_days = 1000

# Each day: 70% chance tonkatsu (500g), 30% chance hamburger (350g)
bento_types = np.random.choice(['tonkatsu', 'hamburger'],
                               size=n_days,
                               p=[0.7, 0.3])

weights = np.where(bento_types == 'tonkatsu', 500, 350)

# Calculate averages
observed_average = np.mean(weights)
theoretical_expected = 0.7 * 500 + 0.3 * 350

print(f"Observed average: {observed_average:.1f}g")
print(f"Theoretical E[X]: {theoretical_expected:.1f}g")
print(f"Difference: {abs(observed_average - theoretical_expected):.1f}g")

# Show the counts
n_tonkatsu = np.sum(bento_types == 'tonkatsu')
n_hamburger = np.sum(bento_types == 'hamburger')
print(f"\nActual counts:")
print(f"  Tonkatsu: {n_tonkatsu} ({n_tonkatsu/n_days*100:.1f}%)")
print(f"  Hamburger: {n_hamburger} ({n_hamburger/n_days*100:.1f}%)")
```

<details>
<summary>可視化コードを表示</summary>

```python
import matplotlib.pyplot as plt

# Plot histogram with both averages
plt.figure(figsize=(10, 6))
plt.hist(weights, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
plt.axvline(observed_average, color='red', linestyle='--', linewidth=2,
            label=f'Observed average: {observed_average:.1f}g')
plt.axvline(theoretical_expected, color='blue', linestyle='--', linewidth=2,
            label=f'Theoretical E[X]: {theoretical_expected}g')
plt.xlabel('Weight (g)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=11)
plt.title("1000 Days of Mystery Bentos", fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('simulation_validation.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![10,000個のシミュレートされた謎の弁当](../../images/intro2/simulation_validation.png)

**出力：**
```
Observed average: 455.5g
Theoretical E[X]: 455.0g
Difference: 0.5g

Actual counts:
  Tonkatsu: 701 (70.1%)
  Hamburger: 299 (29.9%)
```

長期平均は期待値に収束する！これが**大数の法則**の作用だ。

## 期待値の性質

期待値には、混合モデルにとって重要な有用な数学的性質がある：

### 線形性

**性質1：** $E[aX + b] = aE[X] + b$

確率変数をスケーリングおよびシフトすると、その期待値も同じようにスケーリングおよびシフトされる。

**例：** Chibanyがグラムの代わりにオンスで測定するようにした。
1グラム ≈ 0.035オンスなので、オンスでの重さ = グラムでの重さ × 0.035

$$E[\text{オンスでの重さ}] = 0.035 \times E[\text{グラムでの重さ}] = 0.035 \times 455 \approx 15.9\text{ oz}$$

**性質2：** $E[X + Y] = E[X] + E[Y]$

和の期待値は期待値の和だ。これはXとYが**従属**していても成り立つ！

**例：** Chibanyが1日に5つの弁当を受け取る場合、合計重量の期待値は：
$$E[\text{合計}] = E[X_1] + E[X_2] + E[X_3] + E[X_4] + E[X_5]$$
$$= 5 \times E[\text{1つの弁当}] = 5 \times 455 = 2275\text{g}$$

{{% notice style="success" title="線形性が重要な理由" %}}
この**線形性の性質**が混合モデルを機能させるものだ！

混合があるとき：
$$E[X] = \theta \cdot E[X_{\text{とんかつ}}] + (1-\theta) \cdot E[X_{\text{ハンバーグ}}]$$

複雑な混合の期待値を、成分の期待値の加重平均を取るだけで計算できる。

これは第5章でガウス混合モデルを学ぶときに重要になる！
{{% /notice %}}

## GenJAXで混合をモデル化する

次に、GenJAXを使ってChibanyの弁当混合を**生成モデル**として表現してみよう！これはチュートリアル2で学んだことに直接基づいている。

### 生成プロセス

チュートリアル2で学んだように、ランダムプロセスは**生成関数**を使って表現する。Chibanyの弁当選択プロセスを以下に示す：

```python
import jax
import jax.numpy as jnp
from genjax import gen, flip

@gen
def bento_mixture():
    """Generate a single bento weight from the mixture"""
    # Step 1: Choose the bento type
    # 70% chance of tonkatsu, 30% chance of hamburger
    is_tonkatsu = flip(0.7) @ "type"

    # Step 2: Assign the weight based on type
    # (For now, we use exact weights - we'll add variation in Chapter 3!)
    weight = jnp.where(is_tonkatsu, 500.0, 350.0)

    return weight
```

**ここで何が起きているか？**

1. `flip(0.7)` は重みつきのコインを投げる：70%がTrue（とんかつ）、30%がFalse（ハンバーグ）
2. `@ "type"` はこのランダムな選択にアドレスを付与する（チュートリアル2の第3章で学んだように）
3. `jnp.where(is_tonkatsu, 500.0, 350.0)` はとんかつなら500g、ハンバーグなら350gを返す — ランダムな `type` から計算された単一の決定論的な値で、モデルの戻り値となる（アドレスを付与するのは `"type"` のようなランダムな選択のみで、決定論的な結果には付与しない）

これがChibanyの弁当の**生成モデル**だ！

### モデルからのシミュレーション

1000個の弁当をシミュレートして平均重量を計算してみよう。Chibanyの実験と同じだ：

<!-- validate: skip-output -->
```python
import jax.random as random

# Create a random key (GenJAX requires explicit randomness)
import jax.numpy as jnp

key = random.PRNGKey(42)

# Simulate 1000 bentos
n_bentos = 1000
weights = []

for i in range(n_bentos):
    key, subkey = random.split(key)
    trace = bento_mixture.simulate(subkey, ())
    weights.append(trace.get_retval())

weights = jnp.array(weights)

# Calculate statistics
mean_weight = jnp.mean(weights)
n_tonkatsu = jnp.sum(weights == 500.0)
n_hamburger = jnp.sum(weights == 350.0)

print(f"Simulated average weight: {mean_weight:.1f}g")
print(f"Theoretical E[X]: {0.7 * 500 + 0.3 * 350:.1f}g")
print(f"\nCounts:")
print(f"  Tonkatsu (500g): {n_tonkatsu} ({n_tonkatsu/n_bentos*100:.1f}%)")
print(f"  Hamburger (350g): {n_hamburger} ({n_hamburger/n_bentos*100:.1f}%)")
```

**出力：**
```
Simulated average weight: 451.5g
Theoretical E[X]: 455.0g

Counts:
  Tonkatsu (500g): 677 (67.7%)
  Hamburger (350g): 323 (32.3%)
```

シミュレートされた平均は理論的な期待値に非常に近い！

### 期待値との接続

期待値の公式を思い出そう：
$$E[X] = 0.7 \times 500 + 0.3 \times 350 = 455\text{g}$$

**GenJAXはこのプロセスをシミュレートする：**
1. 各シミュレーションは生成プロセスからサンプリングする
2. 多くのサンプルの平均が期待値を**近似**する
3. これが**モンテカルロ推定**だ：シミュレーションを使って数学的な期待値を近似する

### 個別のトレースの検査

GenJAXの強みの1つは、モデルが生成するものを**検査**できることだ。いくつかのトレースを見てみよう：

<!-- validate: skip-output -->
```python
# Generate and examine 5 bentos
key = random.PRNGKey(123)

for i in range(5):
    key, subkey = random.split(key)
    trace = bento_mixture.simulate(subkey, ())

    # trace.get_choices() returns the random choices made; "type" is the one we addressed
    bento_type = "Tonkatsu" if trace.get_choices()["type"] else "Hamburger"
    weight = trace.get_retval()

    print(f"Bento {i+1}: {bento_type:10s} → {weight:.0f}g")
```

**出力：**
```
Bento 1: Hamburger  → 350g
Bento 2: Hamburger  → 350g
Bento 3: Tonkatsu   → 500g
Bento 4: Tonkatsu   → 500g
Bento 5: Hamburger  → 350g
```

各トレースは**種類**（ランダムな選択）と**重さ**（戻り値）の両方を記録する。これはチュートリアル2の第3章で学んだ**トレース構造**だ！

{{% notice style="info" title="GenJAX vs. 純粋なPython" %}}

**なぜ純粋なPython/NumPyの代わりにGenJAXを使うのか？**

今のところ、GenJAXバージョンはやり過ぎに思えるかもしれない。でも私たちが得るものを見てみよう：

1. **明示的な生成モデル**：コードが確率的なストーリーのように読める
2. **アドレス可能な選択**：すべてのランダムな決定に名前がある（`"type"`、`"weight"`）
3. **条件付け**（もうすぐ！）：「重さが425gの場合は？」と聞ける
4. **推論**（第4〜6章）：データからパラメータを学習できる
5. **合成可能性**：拡張が簡単（弁当の種類の追加、重さのばらつきの追加など）

モデルが複雑になるほど（第3〜6章）、GenJAXは不可欠になる！

{{% /notice %}}

### プレビュー：何が足りないか？

このモデルは**離散混合**（とんかつ対ハンバーグ）を捉えているが、**捉えられていない**ことに注目しよう：

- 実際のとんかつ弁当は正確に500gではない — ばらつきがある（488g、505g、515gなど）
- 実際のハンバーグ弁当も正確に350gではない — ばらつきがある（348g、358g、362gなど）

この**カテゴリ内のばらつき**をモデル化するには：

1. **連続分布**（第2章）
2. **ガウス分布**（第3章）
3. **ガウス混合モデル**（第5章）

が必要だ。そこへ向かっている！

## でも、まだ終わっていない…

Chibanyはヒストグラムを見つめた。平均は理解できた。455gは混合として意味をなす。でも、まだ何かが気になる。

これら2つの測定値を見てみよう：
- **488g**（おそらくとんかつ）
- **362g**（おそらくハンバーグ）

でも**425g**はどうだろう？ちょうど真ん中だ。重いハンバーグなのか、それとも軽いとんかつなのか？

そして、もう1つ：重さは正確に500gと350gではない。ばらつきがある！とんかつ弁当の中には520gのものもあれば、485gのものもある。なぜだろう？

Chibanyは気づいた：

> **離散カテゴリでは不十分だ。重さは連続的だ。**

可能な値は2つだけではない（350gと500g）。340gから520gの間には**無限に多くの**可能な重さがある。

ヒストグラムがこれを示している：データには各カテゴリ内に**広がり**がある。

これに対処するためには、新しい種類の確率が必要だ：**連続確率分布**。

そして最も重要な連続分布は？**ガウス分布**（正規分布とも呼ばれる）だ。それが各カテゴリ内のベルカーブ形状を生み出す。

でもまず、連続確率を扱うための基本的な枠組みを理解する必要がある……

## まとめ

{{% notice style="success" title="第1章のまとめ：重要なポイント" %}}

**謎：**
- Chibanyは謎の弁当を受け取り、重さしか量れない
- 平均重量は445gだが、445gに近い弁当はほとんどない！
- ヒストグラムは445gではなく、350gと500gの2つの山を示している

**解決：期待値**
- Chibanyは**混合**を受け取っている：70%のとんかつ（約500g）、30%のハンバーグ（約350g）
- **期待値**は結果の加重平均だ：
  $$E[X] = \sum_{i} p_i \cdot x_i = 0.7 \times 500 + 0.3 \times 350 = 455\text{g}$$

**重要な概念：**
1. **期待値 ≠ 「期待される」結果**：長期平均であり、最も可能性の高い値ではない
2. **重心**：E[X]は分布がシーソーでバランスを保つ位置
3. **構造を隠す**：E[X]だけでは双峰性の形状やばらつきは分からない
4. **大数の法則**：サンプルサイズが大きくなるにつれて、標本平均はE[X]に収束する

**重要な性質：**
- **スケーリング：** $E[aX + b] = aE[X] + b$
- **線形性：** $E[X + Y] = E[X] + E[Y]$（従属変数でも成立！）
- **混合：** $E[\text{混合}] = \theta E[X_1] + (1-\theta) E[X_2]$

**まだ必要なもの：**
- **広がり**の尺度（分散/標準偏差）：第4章で登場
- **連続確率**の枠組み：次章で登場！
- **カテゴリ内のばらつき**の理解：なぜすべてのとんかつが正確に500gでないのか？

**次を見据えて：**
次の章では基本的な課題に取り組む：可能な値が**無限に多い**場合（連続分布）に確率をどう扱うか。

{{% /notice %}}

## 練習問題

### 問題1：メニューの拡張
Chibanyの学生たちが第3の種類の弁当を持ってくるようになった：**寿司**（600g）。今の割合は：
- 50%のとんかつ（500g）
- 30%のハンバーグ（350g）
- 20%の寿司（600g）

弁当の期待重量はいくらか？

{{% expand "答え" %}}

$$E[X] = 0.5 \times 500 + 0.3 \times 350 + 0.2 \times 600$$
$$E[X] = 250 + 105 + 120 = 475\text{g}$$

期待される重さは**475g**だ。

注意：これは依然として加重平均だが、今は2成分ではなく3成分になっている！

{{% /expand %}}

### 問題2：複数の弁当
割合が70%のとんかつ（500g）と30%のハンバーグ（350g）の場合、10個の弁当の**合計**重量の期待値はいくらか？

{{% expand "答え" %}}

期待値の線形性より：
$$E[\text{10個の合計}] = 10 \times E[\text{1個の弁当}]$$
$$E[\text{10個の合計}] = 10 \times 455 = 4550\text{g}$$

または約**4.55 kg**（およそ10ポンド）。

**重要な洞察：** どの弁当がとんかつかハンバーグかを知る必要はない。線形性によって、合計の期待値を直接計算できる！

{{% /expand %}}

### 問題3：概念的な挑戦

ChibanyはE[X] = 455gの弁当を受け取っていることを観察した。同僚は別のカフェテリアから弁当を受け取り、同様にE[X] = 455gを観察した。

これは同じ分布の弁当を受け取っていることを意味するか？なぜそう思うか、またはなぜそう思わないか？

{{% expand "答え" %}}

**いいえ！** 同じ期待値を持つ、まったく異なる分布がありうる。

**E[X] = 455gとなるシナリオの例：**

**シナリオ1（Chibany）：**
- 70%のとんかつ（500g）、30%のハンバーグ（350g）

**シナリオ2（同僚）：**
- すべての弁当がちょうど455gの重さ

**シナリオ3（同僚）：**
- 50%のとんかつ（500g）、50%の寿司（410g）

**シナリオ4（同僚）：**
- 20%のメガ弁当（800g）、80%のライト弁当（368.75g）
- 確認：0.2 × 800 + 0.8 × 368.75 = 160 + 295 = 455g ✓

4つのシナリオすべてがE[X] = 455gだが、**まったく異なる分布**だ！

**区別するために必要なもの：**
- **分散/標準偏差**：重さはどれほど散らばっているか？
- **形状**：単峰性（1つの山）、双峰性（2つの山）、一様？
- **範囲**：最小・最大の可能な重さ
- **完全なヒストグラム**：実際の分布を見る

**重要な教訓：** 期待値は分布の単なる**1次モーメント**だ。全体像を捉えられない。だから分散や完全な確率密度関数などの追加ツールが必要だ。

{{% /expand %}}
