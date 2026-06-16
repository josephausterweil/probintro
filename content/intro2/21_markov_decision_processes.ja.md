+++
date = "2026-06-16"
title = "マルコフ決定過程：世界を知っているときの計画立案"
weight = 21
+++

## 1つの決定から決定の列へ

[第20章](../20_statistical_decision_theory/)はChibanyに*1つ*の良い決定を下す方法を教えた：損失を比較考量し、信念について平均を取り、行動する。しかし人生は1つの決定ではない。弁当を食べれば翌日どれだけ空腹かが変わる；今日ジムをサボれば明日行くのがより難しくなる。行動は*次*の行動が直面する世界を作り変える。選択が前方へと波及する帰結を持った瞬間、「最良の行動を選ぶ」だけではもはや不十分だ——最良の*列*を選ばなければならない。

> **Jamal：**「わかった、じゃあ一週間まるごとを一気に計画すればいい——月曜は軽めに食べて、火曜はジム、水曜は…」
>
> **Alyssa：**「*あらゆる*偶発事態を計画する？毎日たとえ数個の選択肢しかなくても、一週間分の計画の数は爆発する。それに世界はノイジーだよ——水曜は台本どおりにいかないかもしれない。」

Alyssaは本当の障害に名前を付けた——しかし正確であることには価値がある。なぜなら*ノイズ*は実は第2の問題ではないからだ。[第20章](../20_statistical_decision_theory/)の決定理論的解法を真剣に受け止めれば、ノイジーな世界が計画を壊すことはない；それは単に、正しい対象が初めから固定された台本ではなかったということを意味するだけだ。本来の解法は、*実際に何が起こるか*の関数として何をすべきかを述べる**決定規則**だ——月曜は軽めに食べ、*もし*水曜がうまくいかなければ調整する。そのような規則はすでにノイズを吸収している；台本が脆いのは、観測を捨ててしまったからにすぎない。

そうして残るのは、本当に痛手となる障害だ：**コスト**である。$|A|^T$ 個の素の行動列があり——四択を一ヶ月行う場合 $4^{30} \approx 10^{18}$ ——、そして*偶発対応的な*規則は、これまでに観測された全てを行動に写像しなければならないため、はるかに巨大な空間に存在する；その1つを採点するだけでも、過去のあらゆる選択を踏まえて未来が展開しうるあらゆる道筋についてリスクを平均する必要がある。逐次的意思決定は完璧に well-*defined*（明確に定義されている）——ただ正面から最適化するのが絶望的なだけだ。

救いとなるのは[第13章](../13_markov_chains/)の**マルコフ性**だ。もし未来が過去に依存するのが*現在の状態を通じてのみ*——履歴全体ではなく——であるなら、最良の偶発対応規則は、今あなたがいる状態以外には何も必要としない。これにより「あらゆる履歴から行動への写像」が、*状態*から行動への規則へと崩れ落ちる。これを**方策（ポリシー）**と呼び、その最良のものを見つける仕組みが**マルコフ決定過程**（MDP）である。本章はMDPを組み立て、それからルールを完全に知っている世界について——厳密に——それを解く。

始める前にもう1つ材料を。今日の報酬は、千年後の同じ報酬よりも価値がある。だから私たちは単純に未来の報酬を足し合わせるのではない——それらを**割引**する。割引率 $\gamma$ には後ほど正式に出会う；今のところは「遠い未来ほど価値が小さい」という直観だけを抱いておこう。

---

## MDP ＝ マルコフ連鎖 ＋ 決定 ＋ 報酬

MDPを理解する最もすっきりした方法は、すでに手にしているものから出発して、それを*組み立てる*ことだ。[第13章](../13_markov_chains/)ではマルコフ連鎖は状態の集合と単一の遷移行列 $P$ だった——Chibanyの気分が何の発言権もなく日々漂っていくものだった。MDPはその連鎖に2つのものを、1つずつ加える：

![マルコフ連鎖からMDPへの段階的構築を示す3パネルの図。最初のパネルは素のマルコフ連鎖：sとs'の2つの状態の円が1本の遷移矢印で結ばれ、「1つの行列 P」とラベル付けされている。2番目のパネルは状態の下に報酬ラベル R of s を加え、「報酬を加える＝1行動MDP」とラベル付けされている。3番目のパネルは2番目の行動のための別の色の2本目の遷移矢印を加え、「行列の選択を加える＝行動付きMDP」とラベル付けされている。](../../images/intro2/chibany_chain_to_mdp.png)

1. **報酬を加える。** 各状態に数 $R(s)$ を付与する——そこにいることがどれだけ良いか。報酬付きのマルコフ連鎖は、可能な限り最も単純なMDPだ：*1行動*MDPであり、選択肢は一切なく、連鎖がさまよう中で報酬を集めるだけだ。
2. **遷移行列の選択を加える。** いまエージェントに**行動**を与える。各行動はそれ*自身*の遷移行列だ：行動 $a$ を選ぶとは「明日の状態は、あの行列ではなく*この*行列から引かれる」ということを意味する。行動は次の状態を直接定めるのではない——次の状態が引かれる*分布*を定めるのだ。これがMDPの核心にある唯一の考え方である：**行動は、どの遷移行列が明日を支配するかを選ぶ。**

**マルコフ決定過程**とは、これによって残される5つの構成要素である。それぞれに記号とともに名前を付ける：

- **状態** $S$ ——エージェントが取りうる状況。
- **行動** $A$ ——利用可能な選択肢。
- **遷移関数** $T(s' \mid s, a) = P(s_{t+1} = s' \mid s_t = s, a_t = a)$ ——*行動ごとに*1つの遷移行列。（素のマルコフ連鎖は、行動が1つだけの特殊ケースだ。）
- **報酬** $R(s)$ ——各状態での見返り。（一般には報酬は行動にも依存しうる、$R(s, a)$；Chibanyのものは状態だけに依存する。）
- **割引** $\gamma \in [0, 1)$ ——未来が今と比べてどれだけの価値を持つか。

そして**方策（ポリシー）** $\pi(a \mid s) = P(a_t = a \mid s_t = s)$ はエージェントの規則だ：各状態でどの行動を取るか。世界がマルコフ的であるため、方策は*現在の*状態だけを必要とする——履歴ではない——これこそが $|A|^T$ の爆発を手なずけるものだ。

### ChibanyのウェルビーイングMDP

本章を通じて持ち運ぶ例で具体化しよう。Chibanyのウェルビーイングには**3つの状態**がある——$0 = $ **ジャンク沼**、$1 = $ **頑張り中**、$2 = $ **健康で幸せ**——そして**2つの行動**がある：**ふける**（出前を頼む、安らぐ）か **投資する**（料理する、運動する）。報酬は状態のみに依存する：$R = [\,+1,\,-2,\,+5\,]$。ジャンクはほどほどに心地よい（$+1$）；健康は素晴らしい（$+5$）；**頑張り中は谷だ**（$-2$）——*まだ*見返りのない努力である。

世界全体は、行動ごとに1つの $3 \times 3$ 行列が2つ——「行動＝行列を選ぶ」という考え方を文字どおりにしたものだ：

![ふけるとふける投資すると投資するというラベルの付いた2つの3かける3の遷移行列を並べた図。行と列はジャンク、頑張り中、健康の状態でインデックスされている。ふける行列は確率質量をジャンクの近くに保つ；投資する行列は、頑張り中を通過する代償を払って、確率質量を健康へと上に押し上げる。](../../images/intro2/chibany_matrices.png)

この例を研究する価値あるものにしているからくりはこうだ。ジャンクから健康へ至る*唯一*の道は、$-2$ の頑張り中の谷を**通って**走る——投資するの下では、ジャンクは確率 $0.6$ で頑張り中へ行く。近視眼的なエージェントはジャンク沼に留まり $+1$ を永遠に集める；遠視的なエージェントは $-2$ を飲み込んで $+5$ に到達する。投資するに値するかどうかは、*Chibanyがどれだけ先まで見通すか*に完全に依存する——それこそが $\gamma$ が制御するものだ。その考えを抱いておこう；それが本章の収穫となる。

コードでは、MDP全体は3つの配列だ：

<!-- validate: skip-output -->
```python
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

# states: 0 = Junk rut, 1 = Trying, 2 = Healthy & happy
# actions: 0 = Indulge, 1 = Invest
T = jnp.array([[[.9, .1, 0.], [.7, .3, 0.], [.2, .5, .3]],     # Indulge: T[0, s, s']
               [[.4, .6, 0.], [.1, .4, .5], [0., .1, .9]]])    # Invest:  T[1, s, s']
R = jnp.array([1., -2., 5.])      # reward of being in each state
gamma = 0.9                       # discount factor
states  = ["Junk", "Trying", "Healthy"]
actions = ["Indulge", "Invest"]
```

---

## 生成モデルとしての遷移

遷移関数が*何である*かに注目しよう：状態と行動が与えられると、それは次の状態についての分布だ。これはまさに、あなたが[チュートリアル2](../../genjax/)以来書いてきた種類の**生成モデル**であり——それをGenJAXで書くと「行動が分布を選ぶ」という考え方が実行可能になる。行動は、どの行列のどの行からサンプリングするかをインデックスする：

<!-- validate: tol=0.05 -->
```python
from genjax import gen, categorical

@gen
def transition(s, a):
    # categorical takes log-probabilities; row T[a, s] is the next-state
    # distribution chosen by action a from state s.
    return categorical(jnp.log(T[a, s])) @ "s_next"

# sample 10,000 next-states from Junk (s=0) under Invest (a=1); should match T[1,0]
draws = vmap(lambda k: transition.simulate(k, (0, 1)).get_retval())(jr.split(jr.key(0), 10000))
freqs = [round(float((draws == j).mean()), 2) for j in range(3)]
print("Junk + Invest -> next-state frequencies:", freqs)
print("the model row T[Invest, Junk]          :", [float(x) for x in T[1, 0]])
```

**出力：**
```
Junk + Invest -> next-state frequencies: [0.41, 0.59, 0.0]
the model row T[Invest, Junk]          : [0.4000000059604645, 0.6000000238418579, 0.0]
```

シミュレートされた頻度は、当然そうあるべきように、行列の行と一致する。この `transition` モデルは本章全体の背骨だ：価値反復法はその確率を*読んで*厳密に計画を立て、最後には私たちはそこから*サンプリング*してシミュレーションによって計画を立てる——同じ生成関数を、2通りに使うのだ。

---

## 価値、そしてベルマン方程式

行動を選ぶには、それらを採点する必要があり、その採点は**長期的な割引報酬**だ。3つの定義を、各記号を登場するたびに名付けながら示す：

- 時刻 $t$ からの**リターン**は、未来の全報酬の割引和、$G_t = R_{t} + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots = \sum_{k \ge 0} \gamma^k R_{t+k}$ であり、ここで $R_t = R(s_t)$ は時刻 $t$ にいる状態の報酬だ。割引 $\gamma$ はこの和を有限にし、近い報酬をより重く数える。
- 方策 $\pi$ の下での**状態価値**は、状態 $s$ から $\pi$ に従うことで期待されるリターンだ：$v_\pi(s) = \mathbb{E}_\pi[\,G_t \mid s_t = s\,]$。（ホライズンが無限であり、ダイナミクスが時間とともに変化しないため、これは*どの*状態 $s$ にいるかだけに依存し、*いつ*そこにいるかには依存しない。）
- **行動価値**（あるいは**Q値**）は、今行動 $a$ を取り、それから $\pi$ に従う場合のリターンだ：$q_\pi(s, a) = \mathbb{E}_\pi[\,G_t \mid s_t = s,\, a_t = a\,]$。

**最適方策** $\pi^*$ は、すべての状態で最も高い価値を持つものであり——ある事実がこれを単純にする：最適方策は**決定論的**だ。（なぜか？最良のものより悪いどんな行動を混ぜても平均を*下げる*ことしかできないので、各状態で唯一の最良の行動に全ての重みを置く方策は、どんな確率的な方策と比べても少なくとも同じくらい良い。）だから各状態では単に最良の行動を取ればよく、方策についての平均 $\mathbb{E}_\pi$ は行動についての $\max$ へと崩れ落ちる。これが**ベルマン方程式**を与える——ある状態の価値は、*この*ステップの報酬に、次に着地する先の割引価値を加えたものだ：

$$v^*(s) = \max_a \underbrace{\Big[\, R(s) + \gamma \sum_{s'} T(s' \mid s, a)\, v^*(s') \,\Big]}_{=\; q^*(s,\,a)}.$$

括弧内の量は、まさに私たちが定義したばかりの**行動価値** $q^*(s, a)$ なので、ベルマン方程式は単に $v^*(s) = \max_a q^*(s, a)$ だ——*ある状態の価値とは、その最良の行動の価値である。*言葉で読めば：*$s$ からなしうる最良のことは、即時報酬に割引された次状態価値を加えたものが最大となる行動だ。*バックアップを一枚の絵で：

```mermaid
graph LR
    S["state s"] -->|"each action a"| Q["q*(s,a) = R(s) + γ · expected v* of next state"]
    Q -->|"keep the best"| V["v*(s) = max over a"]
    classDef node fill:none,stroke:#9bbcff,stroke-width:2px,color:#fff
    class S,Q,V node
    linkStyle default stroke:#9bbcff,stroke-width:2px,color:#fff
```

この方程式は再帰的だ——$v^*$ が両辺に座っている——これは循環的に見えるが、次節ではその自己参照を、$v^*$ を無から積み上げる**反復**へと変える。その再帰こそが、$|A|^T$ 個の計画を列挙するのではなく、**動的計画法**によって問題を解けるようにするものだ。

---

## 価値反復法

ベルマン方程式は不動点だ：正しい $v^*$ を右辺に代入すると、同じ $v^*$ が出てくる。**価値反復法**は、その不動点を当たり前の方法で見つける——推測（すべてゼロ）から始め、ベルマン更新を何度も何度も適用し、それが収束するのを見守る：

$$v_{k+1}(s) = \max_a \Big[\, R(s) + \gamma \sum_{s'} T(s' \mid s, a)\, v_k(s') \,\Big].$$

各スイープは価値を未来から一歩さらに「バックアップ」する。JAXでは、アルゴリズム全体がベルマン作用素と `scan` だ：

```python
def bellman(V, g=gamma):
    Q = R[None, :] + g * (T @ V)            # Q[a, s] is exactly q(s, a) = R(s) + g * sum_s' T(s'|s,a) V(s')
    return jnp.max(Q, axis=0), jnp.argmax(Q, axis=0)   # max over a -> v(s); argmax over a -> best action

def value_iteration(n_sweeps=300, g=gamma):
    V, _ = lax.scan(lambda V, _: (bellman(V, g)[0], None), jnp.zeros(3), None, length=n_sweeps)
    return V, bellman(V, g)[1]              # converged values, and the greedy policy

Vstar, pistar = value_iteration()
print("V* =", [round(float(v), 1) for v in Vstar])
print("optimal policy:", [actions[int(a)] for a in pistar])
```

**出力：**
```
V* = [25.6, 28.4, 39.8]
optimal policy: ['Invest', 'Invest', 'Invest']
```

$\gamma = 0.9$ では、最適方策は**すべての状態で投資する**ことだ——ジャンク沼でさえ、そこでは投資するとは $-2$ の谷へ真っすぐ歩いて入ることを意味するのに。価値がその賭けを読み解けるものにする：$v^*(\text{Junk}) = 25.6$ は、Chibanyが永遠にふけることで稼ぐであろう $+1$ ずつより*はるかに*多い。モデルを知っているので、価値反復法は $+5$ までの全てを見通し、谷は値打ちがあると判断する。

方策が*考えを変える*様子を見る価値がある。序盤のスイープは数歩先しか見ないので、ジャンクはまだふけるの安全な $+1$ を好む；健康から十分な価値がバックアップされて初めて、ジャンクは投資するへ反転する：

```python
V = jnp.zeros(3)
print("sweep   V(Junk)  V(Trying)  V(Healthy)   Junk's best action")
for k in range(1, 6):
    Vn, pol = bellman(V)
    print(f"  {k}     {float(Vn[0]):6.2f}   {float(Vn[1]):6.2f}    {float(Vn[2]):6.2f}      {actions[int(pol[0])]}")
    V = Vn
```

**出力：**
```
sweep   V(Junk)  V(Trying)  V(Healthy)   Junk's best action
  1       1.00    -2.00      5.00      Indulge
  2       1.63    -0.38      8.87      Indulge
  3       2.29     2.00     12.15      Indulge
  4       3.03     4.39     15.02      Indulge
  5       4.46     6.61     17.56      Invest
```

健康の $+5$ が、1スイープあたり1回のバックアップで、表を**左へ進軍する**様子を見よう——これこそが「価値を一歩さらにバックアップする」が意味することを、具体的にしたものだ。それは頑張り中を $-2$（スイープ1）から $+2.00$（スイープ3）へと押し上げる；そしてその上昇した価値がジャンクに到達して初めて、谷がついに元を取り、スイープ5でジャンクは投資するにコミットする。価値はさらに数百スイープのあいだ上り続け、$[25.6, 28.4, 39.8]$ で落ち着く：

{{% expand "価値反復法はなぜ収束するのか？（任意）" %}}
ベルマン更新は**縮小写像**だ：2つの異なる価値の推測に適用すると、それらを $\gamma$ 倍だけ*互いに近づける*。ベルマン作用素を $B$ と書けば、$\max_s |B(U)(s) - B(V)(s)| \le \gamma \, \max_s |U(s) - V(s)|$ となる。$\gamma < 1$ なので、繰り返しの適用は任意の誤差を幾何級数的に縮める——$k$ スイープの後、真の $v^*$ までの差は、開始時の差の高々 $\gamma^k$ 倍だ。だから価値反復法は、どの開始推測からでも、*唯一*の不動点へ、常に収束する。（これはまた、まさに私たちが $\gamma \in [0, 1)$ を要求する理由でもある：$\gamma = 1$ では縮小性が失われ、無限ホライズンのリターンは有限でさえないかもしれない。）
{{% /expand %}}

![価値反復法のスイープにわたる3つの状態価値の折れ線グラフ。3つすべてがゼロから始まる；V(Healthy)が最も速く最も高く上昇し、V(Trying)とV(Junk)はよりゆっくり上り、3つすべてが数十スイープまでにおよそ25.6、28.4、39.8で平らになる。注釈は、最適方策がすべての状態で投資することだと記している。](../../images/intro2/value_iteration.png)

下で自分で価値反復法をステップ実行してみよう——一度に1スイープずつ進め、価値バーが埋まり、方策の矢印が落ち着くのを見よう：

<iframe src="../../widgets/mdp-value-iteration.html"
        width="100%" height="560"
        frameborder="0"
        style="background:#111111; border-radius:6px; margin:1rem 0;"
        title="Interactive value-iteration explorer for the Chibany MDP, with a discount-factor slider and per-sweep value bars and policy arrows">
</iframe>

---

## どれだけ先まで？割引率

上記のすべては $\gamma = 0.9$ を使った——遠視的なChibanyだ。しかし頑張り中の谷のドラマ全体は*忍耐*についてのものだ：$-2$ は、後にやってくる $+5$ を価値あるものと見なす場合にのみ元が取れる。では $\gamma$ を下げていくと——Chibanyが未来をどんどん気にしなくなると——何が起こるのか？

```python
gammas = jnp.linspace(0.0, 0.95, 200)
junk_action = jnp.array([value_iteration(300, float(g))[1][0] for g in gammas])   # Junk's best action
flip = float(gammas[int(jnp.argmax(junk_action == 1))])                           # first gamma that picks Invest
print(f"Junk's action flips Indulge -> Invest at gamma ~ {flip:.2f}")
for g in [0.5, 0.6, 0.64, 0.7, 0.9]:
    print(f"  gamma = {g}:  Junk -> {actions[int(value_iteration(300, g)[1][0])]}")
```

**出力：**
```
Junk's action flips Indulge -> Invest at gamma ~ 0.64
  gamma = 0.5:  Junk -> Indulge
  gamma = 0.6:  Junk -> Indulge
  gamma = 0.64:  Junk -> Invest
  gamma = 0.7:  Junk -> Invest
  gamma = 0.9:  Junk -> Invest
```

$\gamma \approx 0.64$ に鋭い閾値がある。それより下では、Chibanyは性急すぎる——割引された $+5$ は今日の $-2$ に値しないので、彼は沼に留まりふける。それより上では、未来が十分な価値を持つので、彼は谷に立ち向かい投資する。同じMDP、同じ報酬——*彼がどれだけ先まで見通すか*だけが——彼が抜け出すかどうかを決める：

![割引率ガンマを0から1まで変えたときのジャンクの最適行動のプロット。ガンマが約0.64より下では最適行動はふける（沼に留まる）；0.64より上では投資する（健康へ登る）に反転する。破線の垂直線が0.64での反転を示し、2つの領域は異なる陰影で塗られている。](../../images/intro2/gamma_sweep.png)

ステッププロットは方策が反転する*こと*は示すが、*なぜそこで*かは示さない。その仕組みは、ジャンクにおける2つの行動価値の競争だ——まさに先ほど定義した $q^*(\text{Junk}, a)$ である。**ふけるに対する投資するの優位性**、$q^*(\text{Junk}, \text{Invest}) - q^*(\text{Junk}, \text{Indulge})$ を $\gamma$ に対してプロットしよう：それは性急なエージェントには負（ふけるの方が価値がある）で、まさに $\gamma \approx 0.64$ でゼロを横切って正（投資するの方が価値がある）になる。反転と*は*、その横切りなのだ。

![ジャンク状態でのふけるに対する投資するの優位性——2つの行動価値の差——を割引率ガンマに対してプロットした図。曲線はガンマ0.64より下では負（紫の陰影、「ふけるが勝つ、沼に留まる」とラベル付け）、ガンマ約0.64でゼロを横切り、それより上では正（青の陰影、「投資するが勝つ、抜け出す」とラベル付け）に転じる。](../../images/intro2/gamma_qcross.png)

（上の価値反復法ウィジェットには $\gamma$ スライダーがある——$0.64$ をまたいでドラッグし、ジャンクの方策の矢印が反転するのを見よう。）これが割引の収穫だ：エージェントがどれだけ忍耐強いかを符号化する1つの数 $\gamma$ が、ジャンク沼で立ち往生する人生と、そこから抜け出す人生との違いになりうるのだ。

---

## モデルを知っている → シミュレートする

価値反復法は、遷移確率を読むことで $v^*$ を*厳密に*計算した。しかし私たちは遷移を**生成モデル**として持っている——だからベルマンの代数を一切必要としない、状態の価値を求める2つ目の方法がある：**シミュレートする**ことだ。**軌道**を**ロールアウト**するとは、ありうる未来の1つを前向きに再生することだ：ある状態から始め、方策に従って行動を選び、モデルから次の状態をサンプリングし、繰り返す——通過する各状態の報酬を集めながら。（「ロールアウト」とは、単にこの前向きシミュレーションを指すRLの用語だ；それ以上のものではない。）ある価値を推定するには、$s$ から多数の軌道をロールアウトし、それぞれの割引報酬を合計し、平均を取る。手法については何も変わっていない——これは[第16章](../16_monte_carlo/)の**モンテカルロ**推定量とまさに**同じ**だ。あそこではサンプル $X$ を引いて $f(X)$ を平均することで期待値 $\mathbb{E}_P[f(X)]$ を推定した；ここでは、ランダムなサンプル $X$ が**軌道**まるごとであり、関数 $f$ がその**割引リターン** $G$ だ。価値とは単に期待リターン*である*のだから、それを推定することは以前と同じ「サンプリングして平均する」だ：

$$v_\pi(s) = \mathbb{E}_\pi[\,G_t \mid s_t = s\,] \approx \frac{1}{N} \sum_{i=1}^{N} G^{(i)}, \qquad G^{(i)} = \sum_{k} \gamma^k R(s^{(i)}_k).$$

各軌道は `transition.simulate` 呼び出しの連鎖だ——同じ `@gen` モデルを、今度は*読む*のではなく*サンプリングする*。Chibany MDPには終端状態がないので、リターンは原理的には*無限*の割引和だ；しかし $\gamma^{80} \approx 0.0002$ なので、各ロールアウトを $80$ ステップの `horizon` で打ち切っても、無視できる裾を落とすだけだ。（ステップ $0$ の報酬は割引されない——`disc` は $1$ から始まる——そして各ステップでさらに $\gamma$ の因子が掛けられる。）

<!-- validate: tol=0.4 -->
```python
def mc_value(s0, policy, key, horizon=80, n_traj=5000):
    def one_trajectory(key):
        def step(carry, _):
            s, disc, total, key = carry
            key, k = jr.split(key)
            s_next = transition.simulate(k, (s, policy[s])).get_retval()   # sample the model
            # credit R for the state we're IN, weighted by disc = gamma^k; then discount and move on
            return (s_next, disc * gamma, total + disc * R[s], key), None
        (_, _, total, _), _ = lax.scan(step, (s0, 1.0, 0.0, key), None, length=horizon)
        return total
    return vmap(one_trajectory)(jr.split(key, n_traj)).mean()              # average over rollouts

vhat = mc_value(0, pistar, jr.key(1))      # value of Junk under the optimal policy, by simulation
print(f"Monte-Carlo V(Junk) = {float(vhat):.1f}   vs exact V*(Junk) = {float(Vstar[0]):.1f}")
```

**出力：**
```
Monte-Carlo V(Junk) = 25.7   vs exact V*(Junk) = 25.6
```

Chibanyのありうる未来を五千通りシミュレートし、その割引リターンを平均すると $25.7$ になる——厳密な $25.6$ とほんのわずかの差だ。

{{% notice style="info" title="再利用されたアドレスはいつ問題になるのか？サンプリング対推論" %}}
すべてのステップが同じアドレス `"s_next"` に書き込む——長い連鎖は衝突しないのか？ここでは、**しない**。なぜなら私たちは**前向きにサンプリングしている**だけだからだ：各ステップは `transition.simulate` を*独立に*呼び出し、その戻り値（`.get_retval()`）だけを保持し、トレースを捨てる。トレースが1つに繋ぎ合わされることは決してないので、再利用されたアドレスは見えない——そしてこれは*任意の*長さの連鎖で機能する。欲しいものがありうる未来（ロールアウト、モンテカルロ価値）だけのとき、これが最も単純で最も速いパターンだ；既定でこれに手を伸ばそう。

アドレスが問題になり始めるのは、連鎖に対して**推論**を行うとき——変数を*名前で*参照できる*1つの*トレースが必要なときだ。**Chibanyが3日目に頑張り中だった**ことを観測し、残りの軌道についての事後分布が欲しいとしよう。「3日目の状態」を固定するには、その*特定の*確率変数を指し示さなければならないので、それは独自のアドレスを必要とする——`s_2`、`s_4`、その他と区別される `s_3` だ。それら全てに `"s_next"` を再利用すると、GenJAXはモデルを構築することすらできない：`AddressReuse` を投げる。

任意の長さのアドレス付き連鎖には、正しいツールはGenJAXの **`Scan` コンビネータ**だ——アドレスをあなたの代わりにインデックスし、ループを*巻いたまま*に保つ `lax.scan` の `@gen` 版だ。`@ f"s_{t}"` 付きのPythonの `for t in range(n)` も別々のアドレスを与えるが、それは小さな*静的*な `n` に対してだけだ：それは連鎖全体を計算グラフへと**展開する**（大きな `n` ではコンパイルが遅く、不格好だ）し、動的な `n` はトレースすらできない。だから経験則はこうだ：**サンプリングする → `lax.scan` ＋ `simulate`；推論をする → `Scan` コンビネータ。**
{{% /notice %}}

ロールアウトが積み重なり、移動平均が $v^*$ に収束していく様子を見よう：

<iframe src="../../widgets/mdp-rollout-simulator.html"
        width="100%" height="560"
        frameborder="0"
        style="background:#111111; border-radius:6px; margin:1rem 0;"
        title="Interactive rollout simulator for the Chibany MDP: sample trajectories under the optimal policy and watch the Monte-Carlo value estimate converge to the exact value">
</iframe>

これがオチだ：*ダイナミクスを知っているなら、座って最適方策を**シミュレート**できる——学習は不要だ。*計画立案とは、既知の世界では、ただのシミュレーションだ。それがこの先に続くすべてへの架け橋となる。

{{% notice style="tip" title="なぜこれが2026年でもなお重要なのか" %}}
「モデルをシミュレートして方策を評価する」はおもちゃの考えではない——それは最強の現代エージェントの内部にあるエンジンだ。AlphaZero（Silver et al., 2018）とMuZero（Schrittwieser et al., 2020）は、ゲームの（学習された）モデルを通じて何百万回ものロールアウトをシミュレートすることで計画を立てる；Dreamer（Hafner et al., 2020）のようなモデルベースRLエージェントは世界モデルを学習し、それから自分を改善するためにその中で軌道を*想像する*。上の `transition.simulate` ループは、同じ動きの3状態の戯画だ。私たちは[第22章](../22_q_learning/)で、それに再び——名前付きで、**シミュレーションベースRL**として——出会う。
{{% /notice %}}

{{% notice style="success" title="今できるようになったこと" %}}
あなたは**マルコフ決定過程**をその5つの構成要素から組み立てられる——状態 $S$、行動 $A$、遷移 $T(s' \mid s, a)$、報酬 $R$、割引 $\gamma$ ——行動を*遷移行列の選択*として、そして**方策** $\pi$ を状態から行動への規則として理解しながら。遷移をGenJAXの `@gen` 生成モデルとして書ける。*既知*のMDPを**価値反復法**で厳密に解いて（**ベルマン**バックアップをその不動点まで反復して）、最適価値 $v^*$ と方策 $\pi^*$ を得られる。そして**割引 $\gamma$** がどのようにエージェントのホライズンを定めるかを理解しており——その方策が反転する閾値を見つけられる。さらに、生成モデルを通じてロールアウトを**シミュレート**することで、代数を一切使わずに、モンテカルロ流に価値を推定できる。

次に、[第22章](../22_q_learning/)は、本章が仮定した唯一のもの——モデル——を取り除く。$T$ と $R$ を*知らない*とき、ベルマンバックアップを実行することはできない——経験だけから行動することを**学習**しなければならないのだ。

*用語集：* [マルコフ決定過程](../../glossary/#markov-decision-process-)、[方策](../../glossary/#policy-)、[報酬](../../glossary/#reward-)、[割引率](../../glossary/#discount-factor-)、[リターン](../../glossary/#return-)、[軌道](../../glossary/#trajectory-)、[ロールアウト](../../glossary/#rollout-)、[価値関数](../../glossary/#value-function-)、[ベルマン方程式](../../glossary/#bellman-equation-)、[価値反復法](../../glossary/#value-iteration-)。
{{% /notice %}}

---

## 演習

{{% notice style="info" title="自分で試してみよう" %}}
1. **谷をもっと深くする。** $R(\text{Trying})$ を $-2$ から $-6$ に変え、$\gamma = 0.9$ で価値反復法を再実行する。ジャンクはまだ投資するを選ぶか？それから再び $\gamma$ をスイープする——反転の閾値はどれだけ動くか？
2. **より怠惰な割引。** $\gamma = 0.5$ に設定して価値反復法を実行する。各状態で最適方策を読み取り、谷の観点から、なぜジャンクは今やふけるを好むのに健康はまだ投資するのかを説明せよ。
3. **悪い方策を評価する。** `mc_value` を使って、ジャンクからの*常にふける*方策（`policy = jnp.array([0, 0, 0])`）の価値を推定する。それは $1/(1-\gamma) = 10$ ——$+1$ の状態に永遠に座っている価値——であるべきだと*推測*するかもしれないが、シミュレーションはそれより低く（約 $6.7$）出る。なぜか？（ヒント：ふけるの下でさえ、ジャンクは留まらない——頑張り中の $-2$ へと漏れ出していく。）両方を最適な $v^*(\text{Junk}) = 25.6$ と比較せよ。
{{% /notice %}}

これらすべてをインタラクティブに通しで扱う付属ノートブックがある：

**📓 [Colab で開く: `21_markov_decision_processes.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/21_markov_decision_processes.ipynb)**

---

## 参考文献

- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to control: Learning behaviors by latent imagination. *International Conference on Learning Representations (ICLR)*. <https://arxiv.org/abs/1912.01603>
- Schrittwieser, J., Antonoglou, I., Hubert, T., et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model. *Nature, 588*(7839), 604–609. <https://doi.org/10.1038/s41586-020-03051-4>
- Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science, 362*(6419), 1140–1144. <https://doi.org/10.1126/science.aar6404>

---

Special thanks to [JPPCA](https://jpcca.org/) for their generous support of this tutorial series.
