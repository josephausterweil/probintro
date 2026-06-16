+++
date = "2026-06-16"
title = "ディリクレ過程混合モデル"
weight = 6
+++

## 固定Kの問題点

第5章では、K=2成分のガウス混合モデル（GMM）を使ってChibanyの弁当の謎を解きました。しかし、**Kを事前に指定する**必要があり、BICを用いてその選択を検証しなければなりませんでした。

もし以下のような状況だったら、どうすればよいでしょうか：
- 何種類存在するかわからない場合
- 種類の数が時間とともに変化する場合
- モデルにクラスター数を自動的に発見させたい場合

**ディリクレ過程混合モデル（DPMM）の登場**：データから成分数を学習するベイズノンパラメトリックなアプローチです。

---

## 直感：無限のクラスター

Chibanyの仕入れ先が時間とともに新しい弁当の種類を増やし続けていると想像してください。固定KのGMMでは、次のことが必要です：
1. 新しい種類が登場したことに気づく
2. モデル選択（BIC）を再実行して新しいKを選ぶ
3. モデル全体を再フィットする

DPMMを使えば、モデルはKを事前に指定しなくても、データが到着するにつれて新しいクラスターを**自動的に**発見します。

**重要な洞察**：DPMMは**無限**に多くの潜在クラスターに対して事前分布を設定しますが、実際に「アクティブ」（観測値が割り当てられた）になるクラスターは有限個です。

---

## 中華料理店過程のアナロジー

DPMMを理解する最も直感的な方法は、**中華料理店過程（CRP）**によるものです。

### 設定

無限に多くのテーブルがあるレストランを想像してください（各テーブルはクラスターを表します）。お客さん（観測値）が一人ずつ入店し、座る場所を選びます：

**ルール**：n+1番目のお客さんが座る場所：
- **既に占有されているテーブルk**：そこにいるお客さんの人数に比例した確率： $\frac{n_k}{n + \alpha}$
- **新しいテーブル**：確率： $\frac{\alpha}{n + \alpha}$

ここで：
- nₖ = テーブルkにいるお客さんの人数
- α =「集中パラメータ」（新しいテーブルを作る傾向を制御する）
- n = これまでのお客さんの総数

### 富める者はさらに富む

これにより**富める者はさらに富む**というダイナミクスが生まれます：
- 人気のあるテーブルはさらに多くのお客さんを引き寄せる（クラスタリング）
- しかし、常に新しいテーブルを始める可能性がある（柔軟性）
- αはトレードオフを制御する：αが大きいほど → 新しいテーブルが増える

### 弁当への接続

- **お客さん** = 弁当の観測値
- **テーブル** = クラスター（弁当の種類）
- **着席の選択** = クラスターの割り当て
- α = 新しい弁当の種類が出現する可能性

---

## 数学：スティック折り畳み構成

DPMMは**スティック折り畳み（stick-breaking）**構成を用いて、無限に多くの成分の混合比率を定義します。

### プロセス

長さ1のスティックを想像してください。それを小片に折っていきます：

**k = 1, 2, 3, ..., ∞ について：**
1. βₖ ~ Beta(1, α) をサンプリングする
2. πₖ = βₖ × (1 - π₁ - π₂ - ... - πₖ₋₁) と設定する

**平易な言葉で言うと**：
- β₁ = 成分1のために取るスティックの割合
- 残りのスティック：1 - β₁
- β₂ = 残りのスティックのうち成分2のために取る割合
- π₂ = β₂ × (1 - π₁)
- 以下同様...

**結果**：π₁, π₂, π₃, ... の合計は1になります（有効な混合比率）、後の成分は指数的に小さなシェアを得ます。

### ベータ分布

βₖ ~ Beta(1, α) は残りのスティックをどれだけ取るかを決定します：

- **αが大きい場合**（例：α=10）：折り目がより均等になる → 多くの成分が同様の重みを持つ
- **αが小さい場合**（例：α=0.5）：最初の数回の折り目でスティックの大部分を取ってしまう → 少数の支配的な成分

---

## ガウス混合のためのDPMM：完全なモデル

### モデルの仕様

**スティック折り畳み（無限成分）**：
- k = 1, 2, ..., K_max について：
  - βₖ ~ Beta(1, α)
  - π₁ = β₁
  - πₖ = βₖ × (1 - Σⱼ₌₁ᵏ⁻¹ πⱼ)　（k > 1の場合）

**成分パラメータ**：
- μₖ ~ N(μ₀, σ₀²)　[平均の事前分布]

**観測値**（スティック折り畳みの重みを直接使用）：
- i = 1, ..., N について：
  - zᵢ ~ Categorical(π)　[スティック折り畳みの重みを使ったクラスター割り当て]
  - xᵢ ~ N(μ_zᵢ, σₓ²)　[割り当てられたクラスターからの観測値]

**重要**：クラスター割り当てにはスティック折り畳みの重みπを直接使います。ディリクレ分布による余分なドローを追加すると、「二重ランダム化」が生じ、推論が大幅に遅くなり精度も低下します！

### なぜK_maxが必要なのか？

実際には、無限モデルをある大きなK_max（例えば10や20）で打ち切ります。K_maxが真のクラスター数より大きい限り、この近似は正確です。

---

## GenJAXでDPMMを実装する

修正されたアプローチを使ってChibanyの弁当のDPMMを実装しましょう：

<!-- validate: skip-output -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, beta, normal, categorical, Target, ChoiceMap
import jax.random as random

# Hyperparameters
ALPHA = 2.0      # Concentration parameter
MU0 = 0.0        # Prior mean for cluster means
SIG0 = 4.0       # Prior std dev for cluster means
SIGX = 0.05      # Observation std dev (tight clusters)
KMAX = 10        # Maximum number of components

def make_dpmm_model(K, N):
    """
    Factory function creates DPMM model with fixed K and N

    This avoids TracerIntegerConversionError by making K and N
    closures rather than traced parameters.

    Args:
        K: Maximum number of clusters (truncation level)
        N: Number of observations
    """
    @gen
    def dpmm_model(alpha, mu0, sig0, sigx):
        """
        Dirichlet Process Mixture Model with Gaussian components

        Args:
            alpha: Concentration parameter
            mu0: Prior mean for cluster means
            sig0: Prior std dev for cluster means
            sigx: Observation std dev
        """
        # Step 1: Stick-breaking construction
        betas = []
        for k in range(K):
            beta_k = beta(1.0, alpha) @ f"beta_{k}"
            betas.append(beta_k)

        # Convert betas to pis (mixing weights)
        pis = []
        remaining = 1.0
        for k in range(K):
            pi_k = betas[k] * remaining
            pis.append(pi_k)
            remaining *= (1.0 - betas[k])

        pis_array = jnp.array(pis)
        pis_array = jnp.maximum(pis_array, 1e-6)  # Numerical stability
        pis_array = pis_array / jnp.sum(pis_array)  # Normalize

        # Step 2: Sample cluster means
        mus = []
        for k in range(K):
            mu_k = normal(mu0, sig0) @ f"mu_{k}"
            mus.append(mu_k)
        mus_array = jnp.array(mus)

        # Step 3: Generate observations
        # IMPORTANT: Use pis directly (no extra Dirichlet draw!)
        zs = []
        xs = []
        for i in range(N):
            # Cluster assignment using stick-breaking weights directly
            z_i = categorical(pis_array) @ f"z_{i}"
            zs.append(z_i)

            # Observation from assigned cluster
            x_i = normal(mus_array[z_i], sigx) @ f"x_{i}"
            xs.append(x_i)

        return {
            'mus': mus_array,
            'pis': pis_array,
            'zs': jnp.array(zs),
            'xs': jnp.array(xs),
            'betas': jnp.array(betas)
        }

    return dpmm_model

# Example: Generate synthetic data from DPMM
key = random.PRNGKey(42)

# Create model with K=10 clusters, N=20 observations
model = make_dpmm_model(K=10, N=20)

# Simulate (using default hyperparameters)
trace = model.simulate(key, (ALPHA, MU0, SIG0, SIGX))
result = trace.get_retval()

print(f"Generated data: {result['xs']}")
print(f"Cluster assignments: {result['zs']}")
print(f"Active mixing weights: {result['pis'][result['pis'] > 0.01]}")
```

**出力：**
```
Generated data: [-10.4  -9.9 -10.1   0.1   9.9  10.2 ...]
Cluster assignments: [0, 0, 0, 5, 3, 3, 3, ...]
```

注目：このモデルはアクティブなクラスター（この実行では0、3、5）を自動的に発見し、残りを無視しました！

---

## 推論：DPMMのスライスサンプラー

次に、Chibanyの実際の弁当の重みを条件として与え、クラスターを**推論**してみましょう。これは順方向よりも難しく、推論アルゴリズムの選択が非常に重要です。

### なぜ単純な重点サンプリングではいけないのか？

最初に思いつくアイデアは、事前分布からDPMMをまとめてサンプリングし、データに一致するものを保持する（重点サンプリング/棄却サンプリング）方法です。しかし**ここでは大きく失敗します**：ランダムな10成分スティック折り畳みのドローが、$-10, 0, +10$ の三つの密集したクラスター付近に平均を配置することはほとんどないため、本質的にすべてのサンプルの重みは限りなく小さくなります。盲目的に推測するのではなく、データの方向に*動いて*いくアルゴリズムが必要です。

### スライスサンプリングのアイデア

古典的な解決策は、Walker（2007）の**スライスサンプラー**です。そのトリックは、観測値ごとに補助的な「スライス」変数を1つ導入することです：

$$u_i \sim \text{Uniform}(0,\ \pi_{z_i})$$

ここで $\pi_{z_i}$ は観測値 $i$ が現在属するクラスターの混合重みであり、$\text{Uniform}(a,b)$ は区間 $[a,b]$ 上の一様分布です。

これがなぜ有用なのでしょうか？スライス値が与えられたとき、成分 $k$ が観測値 $i$ の**候補**となるのは、その重みがスライスを超える場合、つまり $\pi_k > u_i$ の場合だけです。スティック折り畳みの重みは幾何的に縮小するため、スライスを超える成分は**有限個**しかありません。つまり、モデルが無限に多くの潜在クラスターを持っていても、各スイープでは有限かつ*適応的な*セットのみを考慮すればよいのです。アクティブなクラスター数 $K$ はデータの要求に応じてスイープごとに増減でき、これはまさにノンパラメトリックモデルが持つべき動作です。（ストレージ上限として十分大きな打ち切り `KMAX` を確保しますが、クラスターが生きているかどうかを決めるのは打ち切りではなくスライスです。）

### ギブスのスイープ

各スイープは4つの条件付き更新を循環し、他の量の現在値を与えた上でそれぞれの量をサンプリングします：

1. **スライス変数** $u_i \sim \text{Uniform}(0, \pi_{z_i})$ — 観測値ごとの閾値を設定する。
2. **割り当て** $z_i$ — スライスで許可されたクラスターの中から、$x_i$ をどれだけうまく説明できるかで重み付けして選ぶ： $ P(z_i = k) \propto \mathbb{1}[\pi_k > u_i]  \mathcal{N}(x_i \mid \mu_k, \sigma_x)$、ここで $\mathbb{1}[\cdot]$ は指示関数（真なら1、偽なら0）。
3. **スティック重み** $\beta_k \sim \text{Beta}(1 + n_k,\ \alpha + \sum_{j>k} n_j)$、ここで $n_k$ は現在クラスター $k$ にある観測値の数 — 標準的なスティック折り畳みの事後分布。
4. **クラスター平均** $\mu_k$ — クラスター $k$ に割り当てられた点からの共役正規-正規更新（空のクラスターは事前分布にフォールバック）。

各ステップが読みやすいよう、スイープに対する明示的な `for` ループを維持します；後の章では `scan` を使ったベクトル化の方法を示します。

<!-- validate: tol=0.6 -->
```python
import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial

# Observed bento weights (three clear clusters)
observed_weights = jnp.array([
    -10.4, -10.0, -9.4, -10.1, -9.9,   # cluster around -10
    0.0,                                 # cluster around 0
    9.5, 9.9, 10.0, 10.1, 10.5,          # cluster around +10
])
N = observed_weights.shape[0]

# Hyperparameters
ALPHA = 1.0    # concentration parameter
MU0   = 0.0    # prior mean for cluster means
SIG0  = 10.0   # prior std for cluster means
SIGX  = 1.0    # observation noise std
KMAX  = 20     # truncation / storage bound (the slice decides how many are live)

def stick_break(betas):
    """betas (K,) in (0,1) -> mixing weights pis (K,): pi_k = beta_k * prod_{j<k}(1-beta_j)."""
    log1m = jnp.log1p(-betas)
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(log1m)[:-1]])
    return betas * jnp.exp(cum)

def normal_logpdf(x, mu, sig):
    return -0.5 * jnp.log(2 * jnp.pi * sig**2) - 0.5 * ((x - mu) / sig)**2

def sample_betas(key, z, alpha, K):
    """Stick-breaking posterior: beta_k ~ Beta(1 + n_k, alpha + sum_{j>k} n_j)."""
    counts = jnp.bincount(z, length=K)                 # n_k
    after = jnp.cumsum(counts[::-1])[::-1]
    after = jnp.concatenate([after[1:], jnp.zeros(1)])  # sum_{j>k} n_j
    keys = random.split(key, K)
    betas = jax.vmap(lambda k, a, b: jax.random.beta(k, a, b))(keys, 1.0 + counts, alpha + after)
    return jnp.clip(betas, 1e-6, 1 - 1e-6)

def sample_mus(key, x, z, K, mu0, sig0, sigx):
    """Conjugate Normal-Normal posterior for each cluster mean (empty -> prior)."""
    counts = jnp.bincount(z, length=K)
    sums = jnp.zeros(K).at[z].add(x)
    prec0, precx = 1.0 / sig0**2, 1.0 / sigx**2
    post_prec = prec0 + counts * precx
    post_mean = (prec0 * mu0 + precx * sums) / post_prec
    post_std = jnp.sqrt(1.0 / post_prec)
    keys = random.split(key, K)
    eps = jax.vmap(lambda k: jax.random.normal(k))(keys)
    return post_mean + post_std * eps

@partial(jax.jit, static_argnums=(2,))
def gibbs_sweep(key, state, K, x, alpha, mu0, sig0, sigx):
    z, betas, mus = state
    k1, k2, k3, k4 = random.split(key, 4)
    pis = stick_break(betas)

    # 1. slice variables u_i ~ Uniform(0, pi_{z_i})
    u = jax.random.uniform(k1, (x.shape[0],)) * pis[z]

    # 2. assignments: P(z_i=k) propto 1[pi_k > u_i] * N(x_i | mu_k, sigx)
    loglik = normal_logpdf(x[:, None], mus[None, :], sigx)       # (N, K)
    logp = jnp.where(pis[None, :] > u[:, None], loglik, -jnp.inf)  # slice indicator
    keys = random.split(k2, x.shape[0])
    z = jax.vmap(lambda k, lp: jax.random.categorical(k, lp))(keys, logp)

    # 3. stick weights, 4. cluster means
    betas = sample_betas(k3, z, alpha, K)
    mus = sample_mus(k4, x, z, K, mu0, sig0, sigx)
    return (z, betas, mus)

# Run the sampler
key = random.PRNGKey(0)
z = jnp.zeros(N, dtype=jnp.int32)                # init: everyone in cluster 0
key, kb, km = random.split(key, 3)
betas = jnp.clip(jax.random.beta(kb, 1.0, ALPHA, (KMAX,)), 1e-6, 1 - 1e-6)
mus = MU0 + SIG0 * jax.random.normal(km, (KMAX,))
state = (z, betas, mus)

n_sweeps, burn = 300, 100
z_history = []
for t in range(n_sweeps):
    key, sk = random.split(key)
    state = gibbs_sweep(sk, state, KMAX, observed_weights, ALPHA, MU0, SIG0, SIGX)
    if t >= burn:
        z_history.append(state[0])

z_history = jnp.stack(z_history)                 # (n_samples, N)
z_final, betas_final, mus_final = state

# Report: relabel active clusters left-to-right by their mean for readability
active = jnp.unique(z_final)
order = sorted(active.tolist(), key=lambda k: float(mus_final[k]))
print("=== DPMM slice sampler (300 sweeps, 100 burn-in, seed 0) ===")
print(f"Discovered {len(active)} active clusters")
for rank, k in enumerate(order):
    n_k = int(jnp.sum(z_final == k))
    print(f"  Cluster {rank}: mu = {float(mus_final[k]):6.2f}   (n = {n_k})")

# Posterior over the number of occupied clusters
Ks = jnp.array([jnp.unique(z).shape[0] for z in z_history])
vals, counts = jnp.unique(Ks, return_counts=True)
print("\nPosterior over number of clusters K:")
for v, c in zip(vals, counts):
    print(f"  P(K = {int(v)}) = {float(c) / Ks.shape[0]:.2f}")
```

**出力：**
```
=== DPMM slice sampler (300 sweeps, 100 burn-in, seed 0) ===
Discovered 3 active clusters
  Cluster 0: mu = -10.03   (n = 5)
  Cluster 1: mu =   0.29   (n = 1)
  Cluster 2: mu =  10.25   (n = 5)

Posterior over number of clusters K:
  P(K = 3) = 0.58
  P(K = 4) = 0.35
  P(K = 5) = 0.07
```

サンプラーは**三つのクラスター全てを復元します** — $\approx -10$ の5つの弁当、$\approx 0$ の1つの弁当、$\approx +10$ の5つの弁当 — そしてそれらの平均を正確に学習します。$K$ に関する事後分布も、クラスター数に本物の*不確かさ*を反映しています：$K=3$ が最も確率が高いですが、モデルは偽の4番目や5番目のクラスターに対しても実質的な重みを与えています — これは固定$K$のGMMでは全く表現できないことです。

{{% notice style="warning" title="注意点：$K$ に関する事後分布は扱いにくい対象" %}}
「$P(K = 3) = 0.58$」をモデルが*実際にいくつのクラスターが存在するか*についての較正された信念として読むのは魅力的です。しかし注意が必要です — **クラスター数に関する周辺事後分布は深く微妙な対象であり、DPMMではあなたが期待するような動作をしません。**

[Miller & Harrison (2014)](https://www.jmlr.org/papers/v15/miller14a.html) は、DPMMのクラスター数に関する事後分布が**一致しない（inconsistent）**ことを証明しました：たとえデータが固定成分数の有限混合から本当に生成されていても、データをどんどん収集するにつれて $K$ に関する周辺事後分布は*余分なクラスターを生み続け、正しい数に落ち着くことがない*のです。驚くべきことに、これはモデルが**密度推定を完全にうまくやっている**間でも起こります — 予測分布は問題なく、結合分布もうまく推定されています；*特定的にカウント $K$* だけが誤動作します。つまり、DPMMは優れた密度推定器であり、クラスターカウンターとしては扱いにくいのです。

良いニュースは、これは修正可能であり、注意深い実践者がよく使う修正があるということです。[Ascolani, Lijoi, Rebaudo & Zanella (2022)](https://projecteuclid.org/journals/bayesian-analysis/volume-18/issue-4/Clustering-Consistency-with-Dirichlet-Process-Mixtures/10.1214/22-BA1357.full) は、集中パラメータ $\alpha$ に対して**事前分布を設定する**こと — 上記で `ALPHA = 1.0` として固定したのではなく — を行うと、データが有限混合から生成されている場合にクラスター数の*一致性を回復*できることを示しました。$\alpha$ 自体を学習させることは（[第12章](../12_hierarchical_bayes/)で見る「事前分布への超事前分布」という同じ手）が、まさにエレガントな解決策です。実用的な結論：DPMMの*予測的*フィットとデータの*クラスタリング*は信頼してください。ただし $\alpha$ に事前分布を設定していない限り、「いくつのクラスターがあるか」という一つの数値は疑いを持って扱ってください。
{{% /notice %}}

{{% notice style="note" title="スライス値が打ち切りを担う" %}}
`KMAX = 20` のストレージスロットを確保しましたが、20クラスターを仮定しているわけではありません：任意のスイープで、ある観測値のスライスを超える重みを持つ成分（$\pi_k > u_i$）だけがアクティブです。データがスライスを通じていくつのクラスターが存在するかを決定します — これがノンパラメトリックにする目的そのものです。
{{% /notice %}}

---

## 事後分布の分析

サンプラーは単一の答えではなく、クラスタリングの*コレクション*（バーンイン後の各スイープごとに1つ）を返します。**ラベルの入れ替え（label switching）**のため、それらをまとめるには少し注意が必要です：あるスイープで「0」と呼んでいたクラスターが次のスイープでは「2」と呼ばれることがあります。ラベルは任意だからです。したがって、スイープをまたいで単純に `mu_0` を平均することはできません — その平均は異なる物理クラスターを混ぜてしまい、意味を失います。

意味のある二つのまとめ方：

**(1) 単一の代表的クラスタリング** — 最終スイープを取り、クラスターを平均によって左から右へ再ラベリングし、番号付けが解釈可能になるようにします：

<!-- validate: tol=0.6 -->
```python
# Relabel the final sweep's clusters 0..K-1 by increasing mean
active = jnp.unique(z_final)
order = sorted(active.tolist(), key=lambda k: float(mus_final[k]))
relabel = {k: r for r, k in enumerate(order)}
mode_assignments = jnp.array([relabel[int(z)] for z in z_final])

print("Cluster assignment per bento:", [int(v) for v in mode_assignments])
print("\nCluster means:")
for r, k in enumerate(order):
    n_k = int(jnp.sum(z_final == k))
    print(f"  Cluster {r}: μ = {float(mus_final[k]):6.2f}   (n = {n_k})")
```

**出力：**
```
Cluster assignment per bento: [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2]

Cluster means:
  Cluster 0: μ = -10.03   (n = 5)
  Cluster 1: μ =   0.29   (n = 1)
  Cluster 2: μ =  10.25   (n = 5)
```

**(2) ラベル不変なまとめ** — 全てのサンプルを平均した、2つの弁当が*同じ*クラスターに属する**共クラスタリング確率**。これはラベルの入れ替えを完全に回避します。なぜなら、「同じクラスター？」というのはクラスターが何と呼ばれるかに依存しないからです：

<!-- validate: tol=0.15 -->
```python
# P(bento i and bento j share a cluster), averaged over posterior samples
same_cluster = jnp.mean(
    (z_history[:, :, None] == z_history[:, None, :]).astype(jnp.float32),
    axis=0,
)

print("Co-clustering probability matrix P(i ~ j):")
for row in jnp.round(same_cluster, 2):
    print("  [" + " ".join(f"{float(v):.2f}" for v in row) + "]")
```

**出力：**
```
Co-clustering probability matrix P(i ~ j):
  [1.00 0.92 0.90 0.88 0.90 0.00 0.00 0.00 0.00 0.00 0.00]
  [0.92 1.00 0.88 0.87 0.93 0.00 0.00 0.00 0.00 0.00 0.00]
  [0.90 0.88 1.00 0.88 0.88 0.00 0.00 0.00 0.00 0.00 0.00]
  [0.88 0.87 0.88 1.00 0.90 0.00 0.00 0.00 0.00 0.00 0.00]
  [0.90 0.93 0.88 0.90 1.00 0.00 0.00 0.00 0.00 0.00 0.00]
  [0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00]
  [0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.90 0.93 0.90 0.86]
  [0.00 0.00 0.00 0.00 0.00 0.00 0.90 1.00 0.88 0.86 0.90]
  [0.00 0.00 0.00 0.00 0.00 0.00 0.93 0.88 1.00 0.88 0.90]
  [0.00 0.00 0.00 0.00 0.00 0.00 0.90 0.86 0.88 1.00 0.86]
  [0.00 0.00 0.00 0.00 0.00 0.00 0.86 0.90 0.90 0.86 1.00]
```

ブロック構造は明確です：$\approx -10$ の5つの弁当（行0-4）はほぼ常に互いと同じクラスターを共有し、残りとは**決して**共有しません；$\approx 0$ の1つの弁当（行5）は単独に座ります；$\approx +10$ の5つの弁当（行6-10）が3番目のブロックを形成します。モデルは**3つのグループが存在することを一切告げられることなく**3つのグループを回復しました — そして、ブロック内の確率が1.0をやや下回っているのは、$K$ の事後分布で見られたように、グループが時々分割される小さな可能性を正直に反映しています。

---

## 落とし穴：ラベルの入れ替え

上記で微妙だが重要な問題を回避しました。DPMMだけでなく*あらゆる*混合モデルに噛みつく問題なので、明示的にする価値があります。

**クラスターラベルは任意です。** モデルの中に「クラスター0」と「クラスター2」を区別するものは何もありません — 尤度
$$p(x \mid z, \mu) = \prod_i \mathcal{N}(x_i \mid \mu_{z_i}, \sigma_x)$$
は、2つのクラスターの名前を入れ替えてそれに合わせて平均を入れ替えても**全く変化しません**。モデルには組み込みの対称性があります：$K$ 個の占有クラスターがある場合、*同じ*クラスタリングの $K!$ 個の同等のラベリングがあり、全て同一の事後確率を持ちます。

**なぜこれが単純なまとめを壊すのか。** 良いサンプラーは、多くのスイープにわたってこれらの同等なラベリングの間を移動します — $-10$ に座っているグループはあるスイープではクラスター0と呼ばれ、次のスイープではクラスター2と呼ばれることがあります。したがって、次のようなラベルごとの平均を計算すると
$$\bar\mu_0 = \frac{1}{S}\sum_{s} \mu_0^{(s)},$$
あるスイープでは $-10$ グループの平均を、他のスイープでは $+10$ グループの平均を平均してしまいます。結果は混乱したものになります — 通常、巨大な標準偏差を持つ全体的なデータ平均付近の数値が得られ、これはサンプラーが完璧に機能していても*推論が失敗した*ように*見えます*。（試してみてください：スイープをまたいで `mu_0` を平均すると $\mu \approx 0 \pm 9$ のようなものが得られます — これは意味をなしません — サンプラーは問題ありません；*まとめ方*が間違っているのです。）

**解決策** — これらはすべてここで使用した、または使用できるものです：

1. **ラベル不変な量を報告する。** 上記の共クラスタリング行列は「クラスター $k$ とは何か？」を決して尋ねず、ただ「$i$ と $j$ は一緒にいるか？」だけを尋ねます。したがって、ラベルの入れ替えはそれに影響を与えることができません。これは最も堅牢な選択肢で、最初に使うべきものです。クラスター数 $K$ に関する事後分布もラベル不変です。
2. **サンプル全体の平均ではなく、単一の代表的サンプルをまとめる** — 例えば最終スイープ（または最高事後確率のスイープ）を標準的な順序に再ラベリングする。それが `mode_assignments` の行ったことです：クラスターを平均で左から右へ並べ替えて「クラスター0」が常に最も軽いグループを示すようにしました。
3. **識別可能性制約を課す / 事後に再ラベリングする。** 順序を固定する（例：$\mu_0 < \mu_1 < \mu_2$）か、平均する前に各スイープのラベルを参照に最も合うように並べ替える再ラベリングアルゴリズム（Stephens, 2000）を実行する。すると、ラベルごとの平均が再び意味を持つようになります。

{{% notice style="warning" title="生のラベルごとパラメータを平均しないこと" %}}
混合モデルのMCMCサンプルに対して `jnp.mean(mu_k for each sweep)` を書こうとしている場合は止めてください。クラスタリングの*ラベル不変*関数をまとめるか、サンプルを標準的な順序に最初に再ラベリングしてください。生のラベルごとの平均は黙って異なるクラスターを混ぜ合わせ、健全なサンプラーを壊れているように見せます。
{{% /notice %}}

---

## 事後予測分布

**質問**：Chibanyは次の弁当にどんな重さを期待すればいいでしょうか？

次の弁当の重さを予測するために、回復した混合から引き出します：各クラスターが持つ弁当の数に比例してクラスターを選び、そのクラスターのガウス分布から重さをサンプリングします。代表的（最終スイープ）クラスタリングを使い、きれいで解釈しやすい予測を行います。

<!-- validate: tol=1.0 -->
```python
# Mixing weights from the representative clustering: proportion of bentos per cluster
counts = jnp.bincount(z_final, length=KMAX)
weights = counts / counts.sum()                  # zero for empty clusters

def draw_one(k):
    k1, k2 = random.split(k)
    z_new = jax.random.categorical(k1, jnp.log(weights + 1e-12))   # pick a cluster
    return mus_final[z_new] + SIGX * jax.random.normal(k2)         # sample its Gaussian

key, sk = random.split(key)
predictions = jax.vmap(draw_one)(random.split(sk, 5000))

print(f"Posterior predictive mean: {float(jnp.mean(predictions)):.2f}")
print(f"Posterior predictive std:  {float(jnp.std(predictions)):.2f}")
for label, lo, hi in [("≈ -10", -15, -5), ("≈  0", -5, 5), ("≈ +10", 5, 15)]:
    frac = float(jnp.mean((predictions >= lo) & (predictions < hi)))
    print(f"  P(next bento {label}) = {frac:.2f}")
```

**出力：**
```
Posterior predictive mean: 0.21
Posterior predictive std:  9.72
  P(next bento ≈ -10) = 0.45
  P(next bento ≈  0) = 0.09
  P(next bento ≈ +10) = 0.46
```

事後予測分布は**多峰性** — 3つのクラスターの混合 — であり、全体的な平均（$\approx 0$）は*合理的な予測ではありません*：実際に0付近の重さの弁当は存在しないからです。有用な記述は各モードの内訳です：次の弁当は軽い（$\approx -10$）タイプと重い（$\approx +10$）タイプのどちらかになる可能性がほぼ同等であり、稀な中間タイプになる小さな可能性があります。可視化してみましょう！

---

## 結果の可視化

```python
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
```

<details>
<summary>可視化コードを表示するにはクリック</summary>

```python
import jax.numpy as jnp

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Representative clustering (final sweep), relabeled left-to-right by mean
active = jnp.unique(z_final)
order = sorted(active.tolist(), key=lambda k: float(mus_final[k]))
colors = ['red', 'green', 'blue', 'purple', 'orange']

# Left: observed data colored by recovered cluster, with cluster centers
ax1.scatter(observed_weights, jnp.zeros_like(observed_weights),
            s=120, alpha=0.4, color='gray', label='Observed data')
for rank, k in enumerate(order):
    mu = float(mus_final[k])
    members = observed_weights[z_final == k]
    color = colors[rank % len(colors)]
    ax1.scatter(members, jnp.zeros_like(members) + 0.05, s=120, color=color)
    ax1.axvline(mu, color=color, linestyle='--', alpha=0.7,
                label=f'Cluster {rank}: μ={mu:.1f}')

ax1.set_xlabel('Weight')
ax1.set_yticks([])
ax1.set_title('Recovered Clusters')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: posterior predictive distribution with each cluster's contribution
ax2.hist(predictions, bins=50, density=True, alpha=0.5, color='gray',
         edgecolor='black', label='Posterior predictive')

counts = jnp.bincount(z_final, length=KMAX)
weights = counts / counts.sum()
x_range = jnp.linspace(-15, 15, 1000)
for rank, k in enumerate(order):
    mu = float(mus_final[k])
    w = float(weights[k])
    color = colors[rank % len(colors)]
    cluster_pdf = w * scipy_norm.pdf(x_range, mu, SIGX)
    ax2.plot(x_range, cluster_pdf, color=color, linewidth=2,
             label=f'Cluster {rank} (π≈{w:.2f})')

ax2.set_xlabel('Weight')
ax2.set_ylabel('Density')
ax2.set_title('Posterior Predictive Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dpmm_results.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![DPMM: 3つのクラスターを発見](../../images/intro2/dpmm_results.png)

**可視化が示すもの**：
- **左**：事後クラスター中心と不確かさを持つ観測データ点
- **右**：三峰性の事後予測（3つのガウス分布の混合）

---

## DPMMと固定KのGMMの比較

| 特徴 | 固定KのGMM | DPMM |
|---------|-------------|------|
| **Kの指定？** | 必要（Kを選択しなければならない） | 不要（データから学習） |
| **モデル選択** | BIC、交差検証 | 自動 |
| **新しいクラスター** | 再フィッティングが必要 | 自動的に発見 |
| **計算コスト** | 低い（固定K） | 高い（無限K、打ち切り） |
| **Kの不確かさ** | モデル化されない | 自然に捉えられる |

**DPMMを使うべき場合**：
- クラスター数が不明な場合
- 探索的データ分析
- データが順次到着する場合（オンライン学習）
- ベイズ的不確かさの定量化が必要な場合

**固定KのGMMを使うべき場合**：
- Kが既知または強く制約されている場合
- 計算効率が重要な場合
- よりシンプルな実装が望ましい場合

---

## α（集中パラメータ）の役割

αは新しいクラスターを作る傾向を制御します：

```python
# Try different alpha values
alphas = [0.1, 1.0, 5.0, 20.0]
```

<details>
<summary>可視化コードを表示するにはクリック</summary>

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, alpha in zip(axes, alphas):
    # Generate stick-breaking weights
    key = random.PRNGKey(42)
    betas = []
    pis = []

    for k in range(20):  # Show first 20 components
        key, subkey = random.split(key)
        beta_k = jax.random.beta(subkey, 1.0, alpha)
        betas.append(beta_k)

        if k == 0:
            pi_k = beta_k
        else:
            pi_k = beta_k * (1.0 - sum(pis))
        pis.append(pi_k)

    # Plot
    ax.bar(range(20), pis)
    ax.set_xlabel('Component')
    ax.set_ylabel('Mixing Proportion')
    ax.set_title(f'α = {alpha}')
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('stick_breaking_alpha.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![異なるα値によるスティック折り畳みプロセス](../../images/intro2/stick_breaking_alpha.png)

**解釈**：
- **α = 0.1**：最初の成分が支配的（少数のクラスター）
- **α = 1.0**：適度な広がり（バランスが取れている）
- **α = 5.0**：より多くの成分がアクティブ（多くのクラスター）
- **α = 20.0**：非常に均等な広がり（拡散的）

---

## 実世界のアプリケーション

### 異常検知
- 正常データがクラスターを形成する
- 外れ値がシングルトンクラスターを作る
- αが外れ値への感度を制御する

### トピックモデリング
- 文書がトピックの混合
- DPMMがトピック数を自動的に発見する
- 各トピックは単語上の分布

### ゲノミクス
- 発現パターンによる遺伝子のクラスタリング
- 機能グループの数は不明
- DPMMが異なる発現プロファイルを特定する

### 画像セグメンテーション
- ピクセルが色/テクスチャでクラスタリングされる
- DPMMが自然なセグメントを見つける
- セグメント数を指定する必要がない

---

## 練習問題

### 問題1：αの調整

先ほどの観測された弁当データを使い、α ∈ {0.5, 2.0, 10.0} で推論を実行してください。

**a)** アクティブなクラスター数はどのように変化しますか？

**b)** 事後不確かさはどのように変化しますか？

<details>
<summary>解答を表示</summary>

先ほどの `gibbs_sweep` を再利用し、各 $\alpha$ でサンプラーを再実行して**占有クラスターの平均数**を報告します（スイープにわたって平均することで、単一スイープのノイズを避けます）：

<!-- validate: tol=0.5 -->
```python
def run_sampler(alpha, seed=0, n_sweeps=300, burn=100):
    key = random.PRNGKey(seed)
    z = jnp.zeros(N, dtype=jnp.int32)
    key, kb, km = random.split(key, 3)
    betas = jnp.clip(jax.random.beta(kb, 1.0, alpha, (KMAX,)), 1e-6, 1 - 1e-6)
    mus = MU0 + SIG0 * jax.random.normal(km, (KMAX,))
    state = (z, betas, mus)
    z_hist = []
    for t in range(n_sweeps):
        key, sk = random.split(key)
        state = gibbs_sweep(sk, state, KMAX, observed_weights, alpha, MU0, SIG0, SIGX)
        if t >= burn:
            z_hist.append(state[0])
    return jnp.stack(z_hist)

for alpha in [0.5, 2.0, 10.0]:
    z_hist = run_sampler(alpha)
    Ks = jnp.array([jnp.unique(z).shape[0] for z in z_hist])
    print(f"α = {alpha:4.1f}:  E[K] = {float(jnp.mean(Ks)):.2f}")
```

**出力：**
```
α =  0.5:  E[K] = 3.21
α =  2.0:  E[K] = 3.60
α = 10.0:  E[K] = 4.76
```

傾向は理論の予測通りです：集中パラメータ $\alpha$ が大きいほど、モデルはより**多くの**クラスター（その一部は3つの実際のグループの偽の分割）を起動し、小さな $\alpha$ はモデルを節倹的に保ちます。$\alpha = 0.5$ でも、モデルは3つの本物のクラスターを見つけることに注意してください — データが十分に分離されており、尤度がより少ないクラスターへの事前分布の引力を上回っています。
</details>

---

### 問題2：逐次学習

Chibanyが弁当を1つずつ受け取っています。各弁当が到着するたびにモデルが更新される**オンライン学習**を実装してください。

**ヒント**：既に持っているスライスサンプラーを再利用する簡単なアプローチ — 新しい弁当が到着するたびに、*これまでに見た全データ*でサンプラーを再実行して占有クラスターを報告する。（より効率的なアプローチは、最初からやり直すのではなく、以前の事後分布から*ウォームスタート*することです；これが逐次モンテカルロの考え方です。）

<details>
<summary>解答（スケッチ）を表示</summary>

これは実装の練習として残します。以下の構造は**疑似コード**です — `run_sampler` は問題1の関数です；ポイントは新しい推論アルゴリズムではなく、データの増加するプレフィックスに対する外側のループです：

<!-- validate: skip -->
```python
def online_dpmm(data_stream):
    """Rerun the slice sampler on a growing prefix of the data."""
    for i in range(1, len(data_stream) + 1):
        prefix = data_stream[:i]                 # all bentos seen so far
        z_hist = run_sampler_on(prefix)          # adapt run_sampler to take the data
        K = average_num_clusters(z_hist)         # E[K] over sweeps, as in Problem 1
        print(f"After {i} bentos: E[K] ≈ {K:.1f}")

online_dpmm(observed_weights)
```

**予想される動作**：占有クラスター数は本当に新しい弁当の種類が初めて現れるときに増加し、各種類が見られた後は安定します — モデルはデータがそれを強制するときにのみ新しいクラスターにコミットします。
</details>

---

## 達成したこと

私たちは謎から始めました：どの個別の弁当の重さとも一致しない平均重量の弁当。このチュートリアルを通じて：

1. **第1章**：混合における期待値のパラドックスを理解した
2. **第2章**：連続確率（PDF、CDF）を学んだ
3. **第3章**：ガウス分布をマスターした
4. **第4章**：パラメータのベイズ学習を行った
5. **第5章**：EMを使ったガウス混合モデルを構築した
6. **第6章**：DPMMによる無限混合に拡張した

**今あなたが持つツール**：
- 複雑な多峰性データのモデル化
- 潜在構造の自動発見
- クラスタリングの不確かさの定量化
- GenJAXによるベイズ推論の実行

### 次に進む先

クラスタリングはデータの山の中の構造を見つけることについてでした。この先の章では、同じベイズの機械を新しい問いに向けます：

- **[第7章：ベイズ汎化](../07_generalization/)** は、少数の例から*概念*をどのように学ぶかを問います — 同じ仮説上の事後分布のアイデアですが、今度は仮説が*集合*（ルール）であり、ヒトがどのように汎化するかのモデルが成果です。
- **[第8〜11章：ベイズネットワークの骨格](../08_bayes_nets/)** は単一モデルからモデルの*構造*へと視野を広げます：グラフとして描くこと（ベイズネット）、どの変数がどれに情報を与えるかを読み取ること（条件付き独立性とd-分離）、*見ること*と*することの違い*を区別すること（因果ベイズネットとdo演算子）、そしてすべてをビット数で測ること（情報理論）。今構築したDPMMはそれ自体ベイズネットです — 第8章がそれを明示します。
- **[第12章：階層的ベイズ](../12_hierarchical_bayes/)** は、モデルが関連する問題から*自身の事前分布を学習*できるよう事前分布の上に事前分布を積み重ねます — そして上記で述べたように、これはDPMMのクラスター数の動作を飼い馴らす正確な手です。

謎の弁当は始まりに過ぎませんでした：チュートリアル3の残りはグラフ、因果、情報、そして事前分布自体を学ぶことについてです。

---

## さらなる読み物

### 理論的基礎
- Ferguson (1973): "A Bayesian Analysis of Some Nonparametric Problems"（オリジナルのDP論文）
- Teh et al. (2006): "Hierarchical Dirichlet Processes"（HDPへの拡張）
- Austerweil, Gershman, Tenenbaum, & Griffiths (2015): "Structure and Flexibility in Bayesian Models of Cognition"（The Oxford Handbook of Computational and Mathematical Psychology所収 - 認知モデリングへのベイズノンパラメトリックアプローチの包括的な概観）

### 実用的な実装
- Neal (2000): "Markov Chain Sampling Methods for Dirichlet Process Mixture Models"（MCMC推論）
- Walker (2007): "Sampling the Dirichlet Mixture Model with Slices"（この章で使用したスライスサンプラー）
- Kalli, Griffiths, & Walker (2011): "Slice sampling mixture models"（改良と明確な解説）
- Blei & Jordan (2006): "Variational Inference for Dirichlet Process Mixtures"（スケーラブルな推論）
- Stephens (2000): "Dealing with label switching in mixture models"（有効な成分ごとのまとめのための事後再ラベリング）

### GenJAXドキュメント
- [GenJAX GitHub](https://github.com/probcomp/genjax) - 公式リポジトリ
- [Probabilistic Programming Examples](https://www.gen.dev/) - Gen.jl（姉妹プロジェクト）

---

{{% notice style="tip" title="重要なポイント" %}}
1. **DPMM**：Kを自動的に学習するベイズノンパラメトリックモデル
2. **スティック折り畳み**：無限の成分の混合比率を定義する
3. **CRP**：直感的な「お客さんとテーブル」の解釈
4. **α**：クラスター傾向を制御する集中パラメータ
5. **スライスサンプラー**：補助スライス変数 $u_i$ が無限スティックを適応的に打ち切り、各ギブスのスイープで有限個のアクティブなクラスターのみを処理する
6. **ラベルの入れ替え**：クラスターラベルは任意 — ラベル不変な量（共クラスタリング、$K$ の事後分布）または単一の再ラベリングされたサンプルでまとめ、生のラベルごとの平均は決して使わない
{{% /notice %}}

---

## インタラクティブな探索

DPMMを自分で実験してみませんか？**インタラクティブJupyterノートブック**で以下のことができます：

- 集中パラメータαを調整してクラスタリングへの効果を確認する
- データ点を追加・削除してモデルの適応を見る
- 打ち切りレベルK_maxを変更する
- リアルタイムで事後分布を可視化する

{{% notice style="success" title="自分で試してみよう！" %}}
**📓 [Google ColabでインタラクティブDPMMノートブックを開く](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/dpmm_interactive.ipynb)**

インストール不要 - ブラウザで直接実行できます！
{{% /notice %}}

ノートブックには以下が含まれます：
- スティック折り畳みを使った完全なDPMM実装
- 全パラメータのインタラクティブウィジェット
- 事後分布のリアルタイム可視化
- 理解を深めるためのガイド付き練習問題

これはα、K_max、そしてデータ自体が事後分布の生成においてどのように相互作用するかの直感を構築する素晴らしい方法です。

---

## 参考文献

- Ascolani, F., Lijoi, A., Rebaudo, G., & Zanella, G. (2023). Clustering consistency with Dirichlet process mixtures. *Biometrika, 110*(2), 551–558. <https://doi.org/10.1093/biomet/asac051>
- Austerweil, J. L., Gershman, S. J., Tenenbaum, J. B., & Griffiths, T. L. (2015). Structure and flexibility in Bayesian models of cognition. In J. R. Busemeyer, Z. Wang, J. T. Townsend, & A. Eidels (Eds.), *The Oxford handbook of computational and mathematical psychology* (pp. 187–208). Oxford University Press.
- Blei, D. M., & Jordan, M. I. (2006). Variational inference for Dirichlet process mixtures. *Bayesian Analysis, 1*(1), 121–143. <https://doi.org/10.1214/06-BA104>
- Ferguson, T. S. (1973). A Bayesian analysis of some nonparametric problems. *The Annals of Statistics, 1*(2), 209–230. <https://doi.org/10.1214/aos/1176342360>
- Kalli, M., Griffin, J. E., & Walker, S. G. (2011). Slice sampling mixture models. *Statistics and Computing, 21*(1), 93–105. <https://doi.org/10.1007/s11222-009-9150-y>
- Miller, J. W., & Harrison, M. T. (2014). Inconsistency of Pitman–Yor process mixtures for the number of components. *Journal of Machine Learning Research, 15*(96), 3333–3370. <https://jmlr.org/papers/v15/miller14a.html>
- Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process mixture models. *Journal of Computational and Graphical Statistics, 9*(2), 249–265. <https://doi.org/10.1080/10618600.2000.10474879>
- Stephens, M. (2000). Dealing with label switching in mixture models. *Journal of the Royal Statistical Society: Series B (Statistical Methodology), 62*(4), 795–809. <https://doi.org/10.1111/1467-9868.00265>
- Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet processes. *Journal of the American Statistical Association, 101*(476), 1566–1581. <https://doi.org/10.1198/016214506000000302>
- Walker, S. G. (2007). Sampling the Dirichlet mixture model with slices. *Communications in Statistics — Simulation and Computation, 36*(1), 45–54. <https://doi.org/10.1080/03610910601096262>
