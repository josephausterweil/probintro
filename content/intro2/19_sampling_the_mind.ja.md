+++
date = "2026-06-16"
title = "心をサンプリングする：人とKemp階層モデル"
weight = 19
+++

{{% notice style="info" title="この章と課題についての注意" %}}
この章では*完全な*手法、すなわち階層モデルに対するGibbs＋Metropolisサンプラーの実装を教えます。ただし、**異なるアプリケーション**（Chibanyが弁当屋を評価する例）を用い、モンテカルロ課題とは**異なるデータおよび異なる導出**に基づいています。これは意図的な設計です。この章を読み終えればそのようなサンプラーを*構築*できるようになり、課題では問題の新しいバリアントに取り組みます。そうすることで、答えをそのままコピーするのではなく、スキルを実際に使うことになります。課題がこの章では扱わない内容に踏み込む箇所は、その都度明示します。
{{% /notice %}}

## 人間が受理ステップになるとき

[第18章](../18_markov_chain_monte_carlo/)では、Metropolis–Hastingsをひとつの操作として構築しました。*変化を提案し、新しい状態と古い状態を比較する確率でそれを受理する*というものです。ここで、奇妙で美しい問いが生まれます。

> **Alyssa:**「微妙に異なる2つのアニメ動物を見せて、『どちらが猫らしいですか？』と聞きます。相手が一方を選ぶ。次にそれを少し調整してまた聞く。相手はまた選ぶ。彼らはステップごとに何を*しているのか*？」
>
> **Jamal:**「変化を提案して、よりネコらしく見えればそれを受理している……それはMetropolisの受理ステップです。*人間*が受理ステップなんです。」
>
> **Alyssa:**「そして、各選択肢が自分の『猫』のイメージにどれだけ合うかに比例して受理するなら、選択の系列はマルコフ連鎖です。その定常分布は何でしょう？」

学習者が自分自身の事後分布 $P(h \mid \text{data})$ に比例する確率で提案を受理するならば、第18章の論理により、仮説 $h$ 上の連鎖はその事後分布を定常分布として収束します。そしてこれを*測定*ツールにする捩りがあります。データなしで「どちらが猫らしいか」だけを使って手続きを実行すると、事後分布が*そのまま*その人の**事前分布**になります。つまり、誰かの頭の中にある「猫」という概念の形そのものです。連鎖は誰かの頭の中にあるアイデアの形に収束します。**MCMCを実行することで、人の事前分布を読み出すことができます。**

これが**マルコフ連鎖モンテカルロ with People**（Sanborn & Griffiths, 2007）です。アニメ動物（比率が数値で表現された棒人間風のキリン、馬、猫、犬）に適用すると、各カテゴリの精神的プロトタイプを人々の選択の定常分布として回復し、回復された空間で4つのカテゴリがきれいに分離します。前章のサンプラーを人に向けると、認知のための測定器になるのです。

この章の残りでは、*同じ*機構を異なるターゲットに向けます。人の事前分布ではなく、[第12章](../12_hierarchical_bayes/)が設定したが粗く近似するしかなかった階層モデルの事後分布です。今ではそれを鋭くサンプリングできます。

---

## 弁当屋階層モデルのサンプラー

Chibanyは弁当屋を評価してきました。各店舗 $i$ について、$n_i$ 回の訪問のうち $k_i$ 回のとんかつ評価が良かった記録があります。各店舗には固有の真の**とんかつ品質率** $\theta_i$ があります。しかし、店舗は無関係ではありません。すべて同じ都市にあり、Chibanyは*集団パターン*—典型的な品質とその周囲の典型的なばらつき—があり、店舗はそこから引かれていると考えています。

これは[第12章](../12_hierarchical_bayes/)の2段階**ベータ・二項分布階層モデル**そのものです。

$$\theta_i \sim \text{Beta}(a, b), \qquad k_i \sim \text{Binomial}(n_i, \theta_i),$$

ここで**集団事前分布** $(a, b)$ は全店舗から*共に学習される*（第12章の超仮説のアイデア——事前分布は仮定ではなく獲得される）。第12章では重点サンプリングでこれを推定し、「粗いツール」と呼びました。事前分布が提案として不適切なためノイズが多かったのです。今度は代わりにMCMCで事後分布をサンプリングします。モデル全体を図で示します。

![弁当屋階層モデルのプレート図。上部に集団平均φと集中度κが、ベータパラメータaとbに向けて矢印が伸び、それが各店舗の品質率θ_iに向けて矢印が伸び、さらに観測された良い評価数k_iへと繋がる。θとkのノードは「店舗1からM」とラベルされたプレートの中に収まり、kのノードは観測済みを示すシェーディングがされている。](../../images/intro2/kemp_plate.png)

**再パラメータ化。** 自然なベータパラメータ $(a, b)$ はサンプラーにとって扱いにくいです。スケールが大きく異なる場合があり（例えば $a = 2$、$b = 30$）、2つの異なる概念が絡み合っています。そこで、それらを分離するペアに切り替えます。

- $\varphi = \dfrac{a}{a+b} \in (0, 1)$ ——集団の**平均**品質（集団が中心を置く場所）、
- $\kappa = a + b > 0$ ——**集中度**（店舗がその平均の周りにどれだけ密集しているか：大きな $\kappa$ = 店舗が全て似ている、小さな $\kappa$ = 店舗がバラバラ）。

集中度が自動的に正に保たれ、対称的なランダムウォークステップが適切に動作するよう、$\ell = \log \kappa$ でさらに一段階下でサンプリングします。3つの数——$(\varphi, \ell)$ と各店舗の $\theta_i$——同じモデルですが、今やそれぞれのつまみが一つのクリーンなものを意味します。

サンプリングする前に2つのつまみに慣れておきましょう。以下のエクスプローラーでは、$\varphi$ と $\kappa$ をドラッグすると集団事前分布 $\text{Beta}(\kappa\varphi, \kappa(1-\varphi))$ が描かれます——*平均*がバンプをスライドさせ、*集中度*がその全体的な性格を変える様子を観察してください。「全店舗が極端」（U字型、$\kappa < 1$）から「全店舗が同じ」（スパイク）まで変化します。

<iframe src="../../widgets/beta-explorer.html"
        width="100%" height="440"
        frameborder="0"
        style="background:#111111; border-radius:6px; margin:1rem 0;"
        title="Interactive Beta distribution explorer in the mean/concentration parametrization">
</iframe>

（$\varphi$ を0.6に保ったまま $\kappa$ を一方の極端からもう一方へスライドしてみてください——そのたった一つのつまみが、Metropolisステップが店舗から*学習*するものです。）

計画は**ハイブリッドサンプラー**です。[第18章](../18_markov_chain_monte_carlo/)が指し示したMHとGibbsの組み合わせそのものです。共役で扱いやすい部分にはGibbs、そうでない部分にはMetropolisを使います。

```mermaid
graph LR
    A["Gibbs: 各θ_iを共役ベータ分布から再描画"] --> B["Metropolis: (φ', ℓ')を提案し<br/>周辺尤度比で受理"]
    B --> A
    classDef node fill:none,stroke:#9bbcff,stroke-width:2px,color:#fff
    class A,B node
    linkStyle default stroke:#9bbcff,stroke-width:2px,color:#fff
```

---

## ステップ1 — θᵢのGibbsサンプリング（共役）

各店舗の率は簡単な部分です。集団 $(a, b)$ を固定すると、各 $\theta_i$ は自分の店舗のデータにのみ依存し、[第12章](../12_hierarchical_bayes/)のベータ・二項分布の共役性により、その完全条件付き分布が閉形式で得られます。

$$\theta_i \mid a, b, k_i, n_i \sim \text{Beta}(a + k_i,\ b + n_i - k_i).$$

これは正確な条件付き分布からの*直接描画*——Gibbsステップです。[第18章](../18_markov_chain_monte_carlo/)と同様、受理・棄却はありません。真の条件付き分布からのサンプリングは常に受理されます。

<!-- validate: skip-output -->
```python
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import betaln, gammaln

# Chibany's 12 bento shops: k_i good tonkatsu ratings out of n_i visits.
# The shops genuinely vary -- some great, some mediocre.
K = jnp.array([9., 3., 7., 5., 8., 2., 6., 9., 4., 7., 1., 8.])    # good ratings
N = jnp.array([10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.])
M = K.shape[0]

def gibbs_theta(key, phi, kappa):
    a = kappa * phi
    b = kappa * (1.0 - phi)
    keys = jr.split(key, M)
    # theta_i | a, b, k_i, n_i ~ Beta(a + k_i, b + n_i - k_i)  -- conjugate, always accept
    return jax.vmap(lambda kk, ki, ni: jr.beta(kk, a + ki, b + ni - ki))(keys, K, N)

theta = gibbs_theta(jr.key(0), phi=0.6, kappa=5.0)
print("one Gibbs draw of theta_i:", [round(float(t), 2) for t in theta])
```

**出力:**
```
one Gibbs draw of theta_i: [0.9, 0.47, 0.54, 0.33, 0.68, 0.17, 0.52, 0.88, 0.43, 0.62, 0.27, 0.9]
```

各店舗の率は自分のデータに引き寄せられます（9/10の店舗1は高く、1/10の店舗11は低く）が、集団にも縮小されます——第12章の部分プーリングそのものが、今度はGibbsの1ステップずつ発生しています。

---

## θᵢの積分消去：ベータ・二項分布の周辺

集団 $(\varphi, \kappa)$ は難しい部分です。その条件付き分布が整った名前付き分布ではないためです。しかし、まずクリーンな簡略化があります。候補となる集団が店舗のデータをどれだけうまく説明するかをスコアリングするには、その店舗の $\theta_i$ は実際には必要ありません——**積分消去**できます。$\theta_i$ は仲介者です。データ $k_i$ は集団に $\theta_i$ を通じてのみ依存しているため、すべての可能な $\theta_i$ にわたって平均をとり、$k_i$ の周辺確率を直接扱えます。

$$p(k_i \mid n_i, a, b) = \int_0^1 \underbrace{p(k_i \mid n_i, \theta_i)}_{\text{二項分布}} \underbrace{p(\theta_i \mid a, b)}_{\text{ベータ分布}} \ d\theta_i = \text{BetaBin}(k_i \mid n_i, a, b).$$

これは通常の**周辺化**です——[第12章](../12_hierarchical_bayes/)の和・積分則と同じで、今回は引用するだけでなく実際に実行します。ベータ・二項分布の共役性により積分は閉形式になります。

$$\text{BetaBin}(k \mid n, a, b) = \binom{n}{k} \frac{B(a+k,\ b+n-k)}{B(a, b)},$$

ここで $B$ はベータ関数です。数値的な安定性のために対数空間で計算します。$\log B(x, y) = \ln\Gamma(x) + \ln\Gamma(y) - \ln\Gamma(x+y)$——これが `betaln` と `gammaln` が与えるものです。

<!-- validate: tol=0.001 -->
```python
def log_betabin(k, n, a, b):
    log_choose = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)   # log C(n, k)
    return log_choose + betaln(a + k, b + n - k) - betaln(a, b)         # log BetaBin

# sanity check: BetaBin is a real distribution, so it must sum to 1 over k = 0..n
ks = jnp.arange(11) * 1.0
total = float(jnp.sum(jnp.exp(jax.vmap(lambda kk: log_betabin(kk, 10.0, 3.0, 2.0))(ks))))
print(f"sum over k of BetaBin(k | n=10, a=3, b=2) = {total:.4f}  (must be 1)")
```

**出力:**
```
sum over k of BetaBin(k | n=10, a=3, b=2) = 1.0000  (must be 1)
```

$\theta_i$ を積分消去することを、統計学者は**コラプシング**（またはRao-Blackwell化）と呼びます。その恩恵は次の通りです。集団に対するMetropolisステップは一切 $\theta_i$ に言及する必要がなくなります——候補の $(\varphi, \kappa)$ を全店舗の*周辺分布*がデータにどれだけ合っているかで一度にスコアリングします。

---

## ステップ2 — (φ, ℓ)のMetropolisサンプリング

次に集団の更新です。$(\varphi, \ell)$ に小さな対称的な移動を提案します——それぞれにガウスのナッジを加えて——Metropolisルールで受理します。$\theta_i$ をコラプスしたので、ターゲットは全店舗にわたるベータ・二項周辺分布の積になり、受理比は提案された集団と現在の集団でその積を比較します。

$$A = \min\left(1,\ \frac{\prod_{i} \text{BetaBin}(k_i \mid n_i, a', b')}{\prod_{i} \text{BetaBin}(k_i \mid n_i, a, b)}\right),$$

ここで $a' = \kappa'\varphi'$、$b' = \kappa'(1 - \varphi')$ は提案された集団です。この比がきれいになる3つの理由があり、それぞれ名前を付ける価値があります。

- **提案補正なし**。ランダムウォーク提案が*対称*であるため（[第18章](../18_markov_chain_monte_carlo/)のMetropolis特殊ケース）。
- **事前分布比なし**。$\varphi$（$(0,1)$ 上一様）と $\ell$ に**平坦事前分布**を置いたため——事前分布の項が等しくなりキャンセルします。
- **ヤコビアンなし**。サンプリングする $(\varphi, \ell)$ 座標で直接提案しているためです。$\theta_i$ はこのステップから完全に外れています（コラプスした）。変数変換を追う必要はありません。

{{% notice style="warning" title="課題がさらに踏み込む箇所" %}}
$\ell = \log\kappa$ に平坦事前分布を置くことは*モデリングの選択*であり、最も単純なものです。代わりに集中度 $\kappa$ に**適切な事前分布**を置く場合——例えば対数正規分布——事前分布の項はキャンセルしなくなり、受理基準に事前分布比の因子が残ります。モンテカルロ課題ではまさにそのバリアントを探索し、コラプスするのではなく $\theta_i$ を陽に保ったまま異なる方法でスコアリングします。同じ連鎖、同じターゲット、異なる簿記——機構を最もクリアに*見る*ことができる、コラプス＋平坦事前分布の形式をここでは使います。他の形式を導出するのは課題の仕事であり、この章の仕事ではありません。
{{% /notice %}}

---

## 完全なサンプラー

2つのステップを組み合わせます。各**スウィープ**では、$\theta_i$ のGibbsアップデートと $(\varphi, \ell)$ のMetropolisアップデートを1回ずつ行います。なぜ2つの*異なる*カーネルをこのように交互にすることが正当なのでしょうか？それは、各カーネルが単独で結合事後分布を不変に保つためです——したがって、それらを順に適用しても同様です。（第18章の詳細釣り合いのような名前付きの事実です。証明はしません。）数千スウィープ実行し、バーンインを捨て、残りを収集します。

<!-- validate: tol=0.6 -->
```python
def log_marg_all(phi, ell):
    kappa = jnp.exp(ell)
    a = kappa * phi
    b = kappa * (1.0 - phi)
    return jnp.sum(jax.vmap(lambda ki, ni: log_betabin(ki, ni, a, b))(K, N))   # product, in logs

def run_sampler(key, n_sweeps, s_phi=0.04, s_ell=0.25, burn=1000):
    def sweep(carry, k):
        phi, ell = carry
        kg, kp, kq, ka = jr.split(k, 4)
        _ = gibbs_theta(kg, phi, jnp.exp(ell))                  # Gibbs the thetas (conjugate)
        phi_p = phi + s_phi * jr.normal(kp)                     # symmetric proposals on (phi, ell)
        ell_p = ell + s_ell * jr.normal(kq)
        outside = (phi_p <= 0.0) | (phi_p >= 1.0)               # flat prior on phi over (0, 1)
        log_ratio = log_marg_all(phi_p, ell_p) - log_marg_all(phi, ell)  # flat priors: marginal ratio only
        accept = (~outside) & (jnp.log(jr.uniform(ka)) < log_ratio)
        phi = jnp.where(accept, phi_p, phi)
        ell = jnp.where(accept, ell_p, ell)
        return (phi, ell), (phi, jnp.exp(ell), accept)
    init = (0.5, jnp.log(5.0))
    _, (phis, kappas, accs) = jax.lax.scan(sweep, init, jr.split(key, n_sweeps))
    return phis[burn:], kappas[burn:], float(jnp.mean(accs))    # drop burn-in

phis, kappas, acc = run_sampler(jr.key(1), 6000)
print(f"MH acceptance rate:            {acc:.2f}")
print(f"posterior mean phi:            {float(jnp.mean(phis)):.3f}")
print(f"posterior median kappa:        {float(jnp.median(kappas)):.1f}")
print(f"predictive P(next rating good) = mean phi = {float(jnp.mean(phis)):.3f}")
```

**出力:**
```
MH acceptance rate:            0.76
posterior mean phi:            0.564
posterior median kappa:        4.6
predictive P(next rating good) = mean phi = 0.564
```

サンプラーは店舗から*集団*を学習します。典型的なとんかつ品質率は $\varphi \approx 0.56$ 前後で、適度な集中度（$\kappa \approx 5$）は店舗が実際に異なることを反映しています。収集されたサンプルの事後ヒストグラムを示します（典型的な1回の実行）。

![サンプラーが収集した描画の2つのヒストグラム。左側は集団平均φの事後分布で、0.56付近を中心とした山を形成している。右側は集中度κの事後分布で右に歪んだ形をしており、中央値が4〜5付近——店舗が実際に異なることと一致する適度な集中度を示している。](../../images/intro2/kemp_posteriors.png)

*初めて訪問する*店舗の次の評価が良い確率は、集団平均 $\varphi$ そのものです——モデルは一度も訪問したことのない店舗に適用できる事前分布を学習しており、これが階層モデルの要点です。そして階層モデルはもう一つの恩恵をもたらします。学習した集団を各店舗の共役事後分布に代入すると、すべての店舗の推定値が集団平均に向けて*縮小*されます——[第12章](../12_hierarchical_bayes/)の部分プーリングが、今度は私たちが仮定したのではなくサンプラーが発見した集団で行われます。

![12店舗の縮小プロット。各店舗の生の評価比率（左）がグレーの線で事後平均（右）に繋がれており、0.1や0.9付近の極端な店舗は学習済み集団平均（約0.56）のオレンジの破線に向けて明らかに引き寄せられているが、中間の店舗はほとんど動かない。](../../images/intro2/kemp_shrinkage.png)

{{% notice style="tip" title="収束しているか？（実践的な補足）" %}}
最初の1000スウィープをバーンインとして捨て、残りを信頼しました。実際には、MCMCの出力を信頼する前に収束を*確認*します——最も単純には**トレースプロット**（[第18章](../18_markov_chain_monte_carlo/)のように、パラメータをスウィープ番号に対してプロットしたもの）を見ることです。バーンイン後は、固定レベルの周りをふらつく定常ノイズのように見えるはずで、ドリフトもなく、一箇所に長くスタックしたプラトーもありません。このサンプラーの $\varphi$ のトレースを、捨てたバーンインをシェーディングして示します。

![集団平均φの6000スウィープにわたるトレース。最初の1000スウィープはグレーにシェーディングされてバーンインとラベルされている。その後、トレースはドリフトもプラトーもなく0.56付近のバンドで安定してふらついており——収束した連鎖のシグネチャである定常ノイズを示している。](../../images/intro2/kemp_phi_trace.png)

いくつかの異なる初期値からサンプラーを実行し、結果が一致することを確認するのが、同じアイデアのマルチチェーン版です。ここでは非形式的に保ちます（「長く実行して最初の部分を捨てる」）。形式的な診断はそれ自体でトピックです。
{{% /notice %}}

---

## GenJAXによる実装

ハイブリッドサンプラーはGenJAXのプリミティブにきれいにマッピングされます。分割を見ておく価値があります。**Gibbs**ステップは `beta` 生成モデルからの直接描画であり、**Metropolis**ステップはコラプスされた周辺分布をスコアリングします——ここには `assess` する結合トレースはありません。$\theta_i$ を積分消去することで集団更新から取り除いたためです。

<!-- validate: skip-output -->
```python
from genjax import gen, beta as gbeta

@gen
def theta_post(a, b):
    return gbeta(a, b) @ "theta"        # the conjugate Beta posterior, as a generative draw

def gibbs_theta_genjax(key, phi, kappa):
    a = kappa * phi
    b = kappa * (1.0 - phi)
    keys = jr.split(key, M)
    return jax.vmap(lambda kk, ki, ni: theta_post.simulate(kk, (a + ki, b + ni - ki)).get_retval())(keys, K, N)

draw = gibbs_theta_genjax(jr.key(0), 0.6, 5.0)
print("GenJAX Gibbs draw of theta_i:", [round(float(t), 2) for t in draw])
# the Metropolis step reuses log_marg_all / log_betabin from above -- it scores the closed-form marginal
print(f"marginal log-score at (phi=0.56, kappa=5): {log_marg_all(0.56, jnp.log(5.0)):.2f}")
```

（モンテカルロ課題では、この*同じ*連鎖を少し異なる方法で組み立てるよう求められます——$\theta_i$ をスコア内で陽に保ち、$\kappa$ に事前分布を追加します。ここで構築した機構がテストされているものです。追加されるのは簿記です。）

---

## ループを閉じる

全体の弧を振り返ってみましょう。[第13章](../13_markov_chains/)ではマルコフ連鎖を渡され、それがどこに落ち着くかを求めました。[第15章](../15_memory_search/)では連鎖を記憶のさまよいをモデル化するために使いました。次に[第16章](../16_monte_carlo/)でサンプリングによる推定を教え、[第17章](../17_particle_filtering/)でサンプルが動くターゲットを追うようにし、[第18章](../18_markov_chain_monte_carlo/)では第13章を*逆方向に*実行しました——選んだターゲットに当たる連鎖を設計したのです。この章ではそれを二重に活用しました。マルコフ連鎖として実行された人間は、頭の中の概念の形を明らかにします。そして、ハイブリッドGibbs-Metropolis連鎖は弁当評価の山の背後にある隠れた集団を学習します。最初に心をモデル化するために使ったウォークは、同じツールとして、事後分布が手で計算するには難しすぎる場所であればどこでも向けることができます——それはほぼあらゆる場所です。

このツールキット全体はGenJAXのプリミティブにも、手法対手法でマッピングされます。チュートリアル2以来収集してきたものです——各古典的アルゴリズムはそれらの短い組み合わせです。

| 古典的手法 | GenJAX サーフェス |
|---|---|
| 基本モンテカルロ | `model.simulate(key, args)` + キーへの `vmap` |
| 棄却サンプリング | `simulate` し、条件に一致する描画を保持 |
| 重点サンプリング | `model.importance(key, constraint, args)` → (トレース, 対数重み) |
| 粒子フィルタ | 重み付け (`importance`) → リサンプリング (`categorical`) → 伝播 (`simulate`) を各観測ごとに |
| MCMC (Metropolis–Hastings) | `model.assess` から組み立て——スコア、比、受理/棄却 |
| Gibbsステップ（共役） | 直接的な `beta(...)` 生成描画 |

{{% notice style="tip" title="2026年においてもこれが重要な理由" %}}
受理比を手書きすることはほぼなくなりました——しかし、これらの章で構築した*サンプル → スコア → 再重み付けまたは受理*のループは消えていません。**学習された提案**を獲得したのです。**拡散モデル**は本質的に、学習された逆MCMCです。ノイズからデータに歩き戻るように訓練された連鎖です。**RLHF**はポリシーからサンプリングして報酬で再重み付けします——学習された尤度による尤度重み付けです。言語モデルからの**Best-of-$N$**サンプリングは、検証器を重みとした重点サンプリングです。これらの4章の機構は、今日のモデルがどのように訓練され、操縦され、整列されるかの概念的なコアです。
{{% /notice %}}

{{% notice style="success" title="今あなたにできること" %}}
**MCMC with People**を理解しました——選択肢を選ぶ人間は*Metropolisの受理ステップ*そのものであり、その選択の連鎖は頭の中の事前分布に収束します。階層ベータ・二項分布モデルの**ハイブリッドGibbs–Metropolisサンプラー**を構築できます。**Gibbs**でユニットごとの率を共役ベータ条件付き分布からサンプリングし、**ベータ・二項周辺**を通じて率を**コラプス**し、集団 $(\varphi, \kappa)$——平均と（対数）集中度に再パラメータ化された——を周辺尤度比で**Metropolis**サンプリングします。各ステップが有効な理由（共役性、対称提案、平坦事前分布、カーネル合成）と、学習した集団から予測値を読み取る方法を知っています。

これでチュートリアル3のサンプリングアークが閉じます。ここでのツール——モンテカルロ、重点サンプリング、MCMC——は、ほぼすべての現代的ベイズモデリングの計算エンジンです。

*用語集:* [マルコフ連鎖モンテカルロ](../../glossary/#markov-chain-monte-carlo-mcmc-), [Gibbsサンプリング](../../glossary/#gibbs-sampling-), [Metropolis–Hastings](../../glossary/#metropolishastings-), [MCMC with People](../../glossary/#mcmc-with-people-), [共役事前分布](../../glossary/#conjugate-prior-), [集中度パラメータ](../../glossary/#concentration-parameter-α-).
{{% /notice %}}

---

## 演習

{{% notice style="info" title="自分で試してみよう" %}}
1. **全店舗が同じ場合。** 全店舗のデータを6/10に置き換えてください。サンプラーを再実行します。$\kappa$ の事後分布はどうなりますか——集中度は*上がる*のか*下がる*のか、そしてなぜですか？（「全店舗が同じ」ことがばらつきについて何を意味するか考えてみましょう。）データが非常に均質なとき、$\kappa$ に*事前分布*が有用かもしれない理由は何ですか？
2. **予測を読み取る。** 収集した `phis` を使って、$\varphi$（5パーセンタイルと95パーセンタイル）の事後平均と大まかな90%区間を報告してください。Chibanyに初めて訪れる未評価の店舗についてどう伝えますか？1文で答えてください。
3. **連鎖を観察する。** スウィープにわたる $\varphi$ のトレースをプロット（または200スウィープごとに値を出力）してください。バーンイン後に定常ノイズのように見えますか？次に、$\varphi = 0.5$ ではなく $\varphi = 0.05$ から連鎖を開始してください——その悪い開始を忘れるまでに何スウィープかかりますか？
{{% /notice %}}

コンパニオンノートブックでこれらすべてをインタラクティブに実行できます。

**📓 [Colabで開く: `19_sampling_the_mind.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/19_sampling_the_mind.ipynb)**

---

このチュートリアルシリーズへの寛大なご支援に、[JPPCA](https://jpcca.org/) に特別の感謝を申し上げます。

---

## 参考文献

- Kemp, C., Perfors, A., & Tenenbaum, J. B. (2007). Learning overhypotheses with hierarchical Bayesian models. *Developmental Science, 10*(3), 307–321. <https://doi.org/10.1111/j.1467-7687.2007.00585.x>
- Sanborn, A. N., & Griffiths, T. L. (2007). Markov chain Monte Carlo with people. In *Advances in Neural Information Processing Systems, 20* (pp. 1265–1272). <https://papers.nips.cc/paper_files/paper/2007/hash/89d4402dc03d3b7318bbac10203034ab-Abstract.html>
