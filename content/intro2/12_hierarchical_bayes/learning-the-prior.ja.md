+++
date = "2026-06-14"
title = "事前分布の学習"
weight = 3
+++

## 母集団の事前分布はどこから来るのか？

これまで $(a, b) = (6, 4)$ を手動で*固定*してきた。しかし、この章が約束していたのは事前分布を**学習する**ことだった。階層モデルはすでにその答えを内包している：$(a, b)$ 自体が潜在変数であり、それ自身の分布を持つため、このチュートリアルで他のすべての未知数を推論したのと同様に、**学生のデータから推論する**ことができる。

$(a, b)$ に広く弱情報的な**ハイパー事前分布**（「母集団の事前分布に対する事前分布」、つまり「それなりにもっともらしい母集団の形状の範囲を表すが、特定の形にはコミットしない」もの）を設定する。以下では $0.5 \le a, b \le 20$ の一様なボックス分布を使用する（範囲を広げても、境界が極端にならない限り推定値はほとんど変わらない）。そして全学生のカウントを観測し、各候補 $(a, b)$ の値をデータへの当てはまりの良さで重み付けする。これはまさに**重点サンプリング**——[第5章](../../05_mixture_models/)とGenJAXチュートリアルで使った手法そのものを、今回はハイパーパラメータというひとつ上のレベルに向けて適用したものだ。

候補 $(a, b)$ をスコアリングするには、それが学生のカウント $k_i$ に割り当てる確率が必要だ——しかし $(a, b)$ はその学生の率 $\theta_i$ の*分布*を教えてくれるだけで、その値は教えてくれない。そこで**すべての可能な $\theta_i$ について平均をとる**：これは §2 で $\text{Beta}(a,b) \to \text{Beta}(a+k, b+n-k)$ という更新を可能にしたのと同じベータ・二項共役性を、逆方向に使うものだ。この平均はきれいな閉形式（**ベータ・二項**周辺分布）を持つ：

$$p(k_i \mid n_i, a, b) = \binom{n_i}{k_i}  \frac{B(a + k_i,  b + n_i - k_i)}{B(a, b)},$$

ここで $\binom{n_i}{k_i}$ は二項係数（「$n_i$ 個から $k_i$ 個を選ぶ」）、$B(\cdot,\cdot)$ はベータ関数——ベータ分布の正規化定数であり、JAX では `betaln` がその対数を計算する。これを暗記する必要はない；母集団をスコアリングするために学生全体の対数をただ合計するだけだ：

<!-- validate: tol=1.5 -->
```python
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import betaln, gammaln

k = jnp.array([70, 28, 6, 3, 2, 0])
n = jnp.array([100, 40, 10, 5, 2, 1])

def log_binom_coeff(n, k):
    # gammaln = log of the Gamma function (a continuous factorial); this is log of "n choose k".
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def population_loglik(a, b):
    """log p(all students' counts | a, b), theta integrated out (Beta-Binomial)."""
    per_student = (log_binom_coeff(n, k)
                   + betaln(a + k, b + n - k)
                   - betaln(a, b))
    return per_student.sum()

# Importance sampling over (a, b): draw many candidate populations from a broad
# hyperprior, weight each by how well it explains the data, report the weighted mean.
key = jr.PRNGKey(0)
ka, kb = jr.split(key)
N = 20000
a_samples = jr.uniform(ka, (N,), minval=0.5, maxval=20.0)   # broad hyperprior on a
b_samples = jr.uniform(kb, (N,), minval=0.5, maxval=20.0)   # broad hyperprior on b

log_w = jax.vmap(population_loglik)(a_samples, b_samples)
w = jnp.exp(log_w - log_w.max())
w = w / w.sum()

a_post = jnp.sum(w * a_samples)
b_post = jnp.sum(w * b_samples)
print(f"inferred a ~= {float(a_post):.2f}")
print(f"inferred b ~= {float(b_post):.2f}")
print(f"implied population tonkatsu rate ~= {float(a_post / (a_post + b_post)):.3f}")
```

**出力：**
```
inferred a ~= 14.57
inferred b ~= 8.12
implied population tonkatsu rate ~= 0.642
```

データだけで母集団の率が約 **0.64** に絞り込まれた——手動で設定した 0.60 と近い値だが、今回は仮定ではなく6人の学生から*学習した*ものだ（多く持参する Alyssa と Ben（0.70）がエビデンスのほとんどを担うため、少し高くなる）。モデルには母集団の平均を教えていない；モデルがそれを推論し、その推論された事前分布が各学生の推定値を縮小する。

{{% notice style="warning" title="正直な注意：ここでの重点サンプリングはノイズが多い" %}}
[第5章](../../05_mixture_models/)の混合推論と同様、広いハイパー事前分布にわたる重点サンプリングは*荒削りな*手法だ——サンプルされた $(a, b)$ のほとんどはデータをうまく説明できないため、実質的に重みを持つものはわずかであり、実行ごとに推定値が揺れる。これは想定内であり、その手法の正直な姿だ。より精密な推論（MCMC、変分法）は後の話題である；ここでの要点は概念的なものだ：**「事前分布を学習する」とは、ひとつ上のレベルの推論にすぎない。**
{{% /notice %}}

「事前分布には事前分布がある」というのは無限後退ではない——コミットできる弱情報的なハイパー事前分布で底を打ち、あとはデータが担う。それが階層ベイズのトリックの全てだ。

---

## モデルが*変動性*について何を学ぶか——そしてなぜそれが Farid の推定に影響するのか

上の推論は母集団について一つの数値を学習した——平均率、約 $0.64$ だ。しかし濃度の議論に戻ると、平均は話の半分に過ぎない；**集中度** $a + b$ が*学生間の差異の大きさ*を決め、それが各学生がどれだけ縮小されるかを制御する。そこで、この二つを明示的に分離し、モデルが**両方**を学習できるようにしよう。

### 再パラメータ化：平均と集中度

ハイパー事前分布を $(a, b)$ に直接設定するのは扱いにくい。なぜなら、このペアは「平均率は何か？」と「学生はどれだけ似ているか？」を絡み合わせているからだ。階層モデルにおける標準的な手法（例：Kemp, Perfors, & Tenenbaum, 2007）に従い、この二つの独立した問いへと**再パラメータ化**する：

$$\mu = \frac{a}{a+b} \quad (\text{母集団の}\textbf{平均}), \qquad \lambda = a + b \quad (\textbf{集中度}),$$

そして $a = \mu\lambda,  b = (1-\mu)\lambda$ で逆変換する。これで各パラメータに*別々の*ハイパー事前分布を設定できる——そして重要なのは $\lambda$ に対するものだ。

{{% notice style="info" title="同じ分布、二組のダイヤル：$(a, b)$ vs. $(\mu, \lambda)$" %}}
これは**新しい分布ではない**——まったく同じベータ分布を、二つの異なるダイヤルの組み合わせで記述したものだ。数学は何も変わらない；単にラベルを付け替えているだけだ。

| | 標準的なパラメータ化 | 再パラメータ化 |
|---|---|---|
| **パラメータ** | $a,\ b$（二つの「ソフトカウント」） | $\mu = \frac{a}{a+b}$、$\ \lambda = a + b$ |
| **各ダイヤルの働き** | $a$ と $b$ はそれぞれ平均と散らばりの*両方*を引っ張る | $\mu$ は**平均**のみを設定；$\lambda$ は**集中度**のみを設定 |
| **変換** | — | $a = \mu\lambda,\quad b = (1-\mu)\lambda$ |
| **例** | $\text{Beta}(6, 4)$ | $\mu = 0.6,\ \lambda = 10$ |

$\text{Beta}(6,4)$ と「$\mu = 0.6,\ \lambda = 10$」は**同じ分布を二通りの書き方で表したもの**だ——$\mu\lambda = 6$ および $(1-\mu)\lambda = 4$ を代入して確かめてほしい。$(\mu, \lambda)$ に切り替える理由は一つ：「平均率は何か？」と「学生はどれだけ似ているか？」について、*別々に*推論し、独立した事前分布を設定できるようになるからだ。これがまさにこのセクションで論じる区別だ。モデルが $a$ と $b$ を必要とするところ（例：`beta(a, b)`）では、引き続き $a = \mu\lambda$ および $b = (1-\mu)\lambda$ を渡す。
{{% /notice %}} モデルがU字型の母集団（各学生がほぼ決定論的、$\lambda < 1$）**または**集中した母集団（学生が全員似ている、$\lambda \gg 1$）を発見できるようにするため、$\lambda$ のハイパー事前分布は**両方の体制をカバーする**必要がある——$1$ の上下で数桁にわたるオーダー。**対数一様**事前分布がまさにそれを実現する：

$$\mu \sim \text{Uniform}(0, 1), \qquad \log \lambda \sim \text{Uniform}(\log 0.1,  \log 100).$$

{{% notice style="warning" title="$\lambda$ のハイパー事前分布が 1 以下に届かなければならない理由" %}}
ハイパー事前分布が $\lambda < 1$ を生成できなければ、モデルはほぼ決定論的な学生の母集団を*表現できない*——データが何を言っていても、U字形には構造的に盲目になる。「$a, b \in [0.5, 20]$」のような素朴なボックスは $\lambda = a + b \ge 1$ を強制し、異質性を静かに排除する。$\lambda$ を何桁もの範囲にわたってスパンする（ここでは対数一様を使って）ことが、**データ**に体制を選ばせることを可能にする。
{{% /notice %}}

### 二つの教室、同じ Emi と Farid

ここが核心であり、ずっと気になっていたかもしれない問いへの答えだ：*Farid（0/1）を平均方向に縮小するのは常に意味があるのか？* **いや——それは Farid が誰と同じ環境にいるかによる。** 二つの異なる教室を考えよう。どちらも**同じ二人のデータ不足の学生**——Emi（2/2）と Farid（0/1）——を含むが、*よく観測された持参者*が異なる。（$k_i/n_i$ は**学生 $i$ が Chibany にとんかつを持参した回数**——トンカツを*持参する*頻度を表し、学生が自分で何を食べるかではない。）

- **教室 A——混合型持参者。** Alyssa 70/100、Ben 28/40、Carmen 6/10、Diego 3/5。よく観測された学生は中程度の率でとんかつを持参している（$0.6$–$0.7$）：誰も一種類だけを確実に持参するわけではない。このデータはまさに**集中した**母集団の典型だ。
- **教室 B——習慣の動物。** Alyssa 97/100、Ben 2/40、Carmen 19/20、Diego 0/20。よく観測された学生は*ほぼ常に同じものを持参する*——Alyssa と Carmen はほぼ毎回とんかつ、Ben と Diego はほぼ毎回ハンバーガー。このデータはまさに**U字型**母集団の典型だ。

各教室を同じ推論に入力し、$\mu$ と $\lambda$ を学習させ、それが見つけた母集団で Emi と Farid を縮小する：

<!-- validate: tol=0.06 -->
```python
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import betaln, gammaln

def log_beta_binom(k, n, a, b):
    # log p(k | n, a, b): the Beta-Binomial marginal (theta integrated out).
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) \
        + betaln(a + k, b + n - k) - betaln(a, b)

def infer_population(k, n, seed=0, n_samples=60000):
    """Infer (mu, lambda) by importance sampling, a = mu*lam, b = (1-mu)*lam.
    Hyperpriors: mu ~ Uniform(0,1); lambda log-uniform over [0.1, 100] so it can
    land BELOW 1 (students differ / U-shaped) or ABOVE 1 (students are alike)."""
    km, kl = jr.split(jr.key(seed))
    mu = jr.uniform(km, (n_samples,), minval=0.01, maxval=0.99)
    lam = jnp.exp(jr.uniform(kl, (n_samples,), minval=jnp.log(0.1), maxval=jnp.log(100.0)))
    a, b = mu * lam, (1 - mu) * lam
    log_w = jax.vmap(lambda a, b: log_beta_binom(k, n, a, b).sum())(a, b)
    w = jnp.exp(log_w - log_w.max()); w = w / w.sum()
    return float(jnp.sum(w * mu)), float(jnp.sum(w * lam))

# Both classrooms share the SAME two data-light students: Emi 2/2, Farid 0/1.
classrooms = {
    "A (mixed bringers)":    (jnp.array([70, 28, 6, 3, 2, 0]),  jnp.array([100, 40, 10, 5, 2, 1])),
    "B (creatures of habit)":(jnp.array([97, 2, 19, 0, 2, 0]),  jnp.array([100, 40, 20, 20, 2, 1])),
}

for label, (k, n) in classrooms.items():
    mu, lam = infer_population(k, n)
    a, b = mu * lam, (1 - mu) * lam
    print(f"Classroom {label}:  inferred mean mu={mu:.2f}, concentration lambda={lam:.1f}")
    for name, ki, ni in [("Emi", 2, 2), ("Farid", 0, 1)]:
        print(f"    {name} {ki}/{ni}: raw {ki/ni:.2f} -> shrunk {(a + ki) / (a + b + ni):.2f}")
```

**出力：**
```
Classroom A (mixed bringers):  inferred mean mu=0.66, concentration lambda=41.3
    Emi 2/2: raw 1.00 -> shrunk 0.68
    Farid 0/1: raw 0.00 -> shrunk 0.65
Classroom B (creatures of habit):  inferred mean mu=0.47, concentration lambda=0.6
    Emi 2/2: raw 1.00 -> shrunk 0.89
    Farid 0/1: raw 0.00 -> shrunk 0.17
```

二つの教室は正反対の集中度を学習した——教室 A では $\lambda \approx 41$（学生が似ている）、教室 B では $\lambda \approx 0.6$（学生が異なる、U字型）——**よく観測された持参者のデータのみから**、そしてそれが*同一の*データ不足の学生に対する判定を逆転させる：

| 学生（同じデータ） | 教室 A（$\lambda \approx 41$） | 教室 B（$\lambda \approx 0.6$） |
|---|---:|---:|
| **Emi**（2/2） | $1.00 \to 0.68$（平均方向へ引き寄せられる） | $1.00 \to 0.89$（1 に近いと判断される） |
| **Farid**（0/1） | $0.00 \to 0.65$（平均方向へ引き寄せられる） | $0.00 \to 0.17$（0 に近いと判断される） |

教室 A では他の全員が中程度のため、Farid が持参したハンバーガーはほぼ確実にフロックだ——グループに向けて強く縮小する。教室 B では他の全員が極端なため、Farid のハンバーガー一回はほぼ額面通りに受け取られる——*これはおそらく常にハンバーガーを持参する学生だ*。**Farid についての観測は同じなのに、結論は正反対——母集団がモデルに「一つのデータ点をどれだけ信頼すべきか」を教えたからだ。** それが——薄いデータをどれだけ信頼すべきかを、他の全員の構造から学習すること——階層ベイズが行う最も深いことだ。

---
