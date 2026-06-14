+++
date = "2026-06-14"
title = "インタラクティブノートブック — 全チュートリアル"
weight = 99
+++

## インタラクティブ Jupyter ノートブック

このページでは、チュートリアルシリーズ全体で利用可能なすべての Jupyter ノートブックの包括的な概要を提供します。各ノートブックは Google Colab で直接開き、すぐにインタラクティブな探索が可能です。

**ノートブックの使い方：**
- 📓 「Open in Colab」をクリックしてブラウザでノートブックを起動する
- ✏️ セルを実行し、コードを編集し、パラメータを試す
- 💾 変更を保持するために Google Drive にコピーを保存する
- 📚 詳細な説明は、リンク先のチュートリアル章に戻って確認する

---

## チュートリアル 1：離散確率

### 最初の GenJAX モデル
**ノートブック**: [📓 Open in Colab: `first_model.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/first_model.ipynb)

**扱う内容：**
- GenJAX による初めての確率的モデル
- Chibany のランチ選択（ハンバーガー vs とんかつ）のシミュレーション
- 離散的な結果に対する基本的な確率計算
- ランダムサンプリングと確率分布の理解

**関連チュートリアル章：**
- [チュートリアル 1、第 3 章：数え上げによる確率](../intro/03_prob_count/)

**トピック：**
- 離散確率分布
- ランダムサンプリング
- GenJAX の基礎
- 確率の可視化

---

### 条件付けとベイズの定理
**ノートブック**: [📓 Open in Colab: `conditioning.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/conditioning.ipynb)

**扱う内容：**
- 実践における条件付き確率
- GenJAX でのベイズの定理の実装
- インタラクティブな例を使ったタクシー問題
- 逐次的な信念更新

**関連チュートリアル章：**
- [チュートリアル 1、第 4 章：条件付き確率](../intro/04_conditional/)

**トピック：**
- 条件付き確率
- ベイズの定理
- 事後分布による信念更新
- 事前分布と尤度

---

### ベイズ学習
**ノートブック**: [📓 Open in Colab: `bayesian_learning.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/bayesian_learning.ipynb)

**扱う内容：**
- 可視化を用いたタクシー問題の完全な解法
- 複数の観測に基づく逐次ベイズ更新
- 異なるベースレートと精度を探索するインタラクティブスライダー
- 事前信念が事後的な結論に与える影響

**関連チュートリアル章：**
- [チュートリアル 1、第 5 章：ベイズの定理](../intro/05_bayes/)
- [チュートリアル 2（GenJAX）、第 4 章：条件付け](../genjax/04_conditioning/)

**トピック：**
- ベイズ推論
- 逐次更新
- ベースレートの効果
- 事前分布—事後分布の関係

---

## チュートリアル 2：GenJAX プログラミング

### 初めての GenJAX モデル（チュートリアル 2）
**ノートブック**: [📓 Open in Colab: `02_first_genjax_model.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/02_first_genjax_model.ipynb)

**扱う内容：**
- GenJAX による生成モデルの構築
- リアルタイムでパラメータを調整するインタラクティブウィジェット
- 確率分布の可視化
- パラメータの変化が結果に与える影響の理解

**関連チュートリアル章：**
- [チュートリアル 2（GenJAX）、第 0 章：はじめに](../genjax/00_getting_started/)
- [チュートリアル 2（GenJAX）、第 2 章：最初のモデル](../genjax/02_first_model/)

**トピック：**
- GenJAX の生成関数
- パラメータ探索
- インタラクティブな可視化
- モデルシミュレーション

---

## チュートリアル 3：連続確率とベイズ学習

### ガウス分布によるベイズ推論のインタラクティブ探索
**ノートブック**: [📓 Open in Colab: `gaussian_bayesian_interactive_exploration.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/gaussian_bayesian_interactive_exploration.ipynb)

**扱う内容：**
- **パート 1：ガウス—ガウスベイズ更新**
  - 尤度の分散を調整するインタラクティブスライダー
  - リアルタイムの事後分布更新による逐次的な観測追加
  - 事後分布と予測分布の比較
  - 測定ノイズが学習に与える影響

- **パート 2：ガウス混合によるカテゴリ分類**
  - 事前分布が決定境界に与える影響
  - 分散比がカテゴリ分類に与える影響
  - 周辺（混合）分布の可視化
  - 双峰分布と単峰分布の理解

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、メインページ](../intro2/)
- [チュートリアル 3（Intro2）、第 4 章：ベイズ学習](../intro2/04_bayesian_learning/)
- [チュートリアル 3（Intro2）、第 5 章：混合モデル](../intro2/05_mixture_models/)

**トピック：**
- ガウス分布
- ベイズパラメータ学習
- 共役事前分布
- 事後推論
- 混合モデル
- 決定境界

---

### 課題 1：ガウスベイズ更新
**ノートブック**: [📓 Open in Colab: `solution_1_gaussian_bayesian_update.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/solution_1_gaussian_bayesian_update.ipynb)

**扱う内容：**
- **パート (a)**：事前分布の可視化
- **パート (b)**：尤度の分散の影響（σ²_x = 0.25 vs. 4）
- **パート (c)**：観測数の影響（N=1 vs. N=5）
- **パート (d)**：精度加重平均の実践
- **パート (e)**：事後分布 vs. 予測分布
- **パート (f)**：解析的な公式の GenJAX による検証

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 4 章：ベイズ学習](../intro2/04_bayesian_learning/)

**トピック：**
- ガウス共役事前分布
- 尤度分散の効果
- 事後分布の集中
- 予測分布
- 精度による重み付け

---

### 課題 2：ガウスクラスター
**ノートブック**: [📓 Open in Colab: `solution_2_gaussian_clusters.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/solution_2_gaussian_clusters.ipynb)

**扱う内容：**
- **パート (a)**：ベイズの定理を用いた P(カテゴリ|観測) の導出
- **パート (b)**：事前分布が決定境界に与える影響
- **パート (c)**：分散比がカテゴリ分類に与える影響
- **パート (d)**：周辺分布の計算
- **パート (e)**：双峰混合 vs. 単峰混合の理解
- **パート (f)**：混合モデルの GenJAX シミュレーション

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 4 章：ベイズ学習 — 問題 2](../intro2/04_bayesian_learning/#problem-2-gaussian-clusters-preview-of-chapter-5)
- [チュートリアル 3（Intro2）、第 5 章：混合モデル](../intro2/05_mixture_models/)

**トピック：**
- 混合モデルによるカテゴリ分類
- 連続分布でのベイズの定理
- 決定境界
- 周辺確率
- 混合分布

---

### ディリクレ過程混合モデル（DPMM）
**ノートブック**: [📓 Open in Colab: `dpmm_interactive.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/dpmm_interactive.ipynb)

**扱う内容：**
- DPMM のインタラクティブな探索
- 自動クラスター発見
- 中華料理店過程の可視化
- 無限混合モデル
- ベイズノンパラメトリクス

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 6 章：DPMM](../intro2/06_dpmm/)

**トピック：**
- ディリクレ過程
- 無限混合モデル
- 中華料理店過程
- ベイズノンパラメトリクス
- 自動モデル選択

---

### ベイズ汎化
**ノートブック**: [📓 Open in Colab: `07_generalization.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/07_generalization.ipynb)

**扱う内容：**
- シールウォームアップ：*仮説の集合*としての概念
- 7 つの候補ルールを用いた 1〜30 の数当てゲーム
- 弱サンプリング vs. 強サンプリングにおける**サイズ原理**
- 汎化勾配 — どの新しい数が当てはまるかの予測

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 7 章：ベイズ汎化](../intro2/07_generalization/)

**トピック：**
- 集合としての仮説
- サイズ原理
- 弱サンプリング vs. 強サンプリング
- 汎化勾配

---

### ベイズネットワーク
**ノートブック**: [📓 Open in Colab: `08_bayes_nets.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/08_bayes_nets.ipynb)

**扱う内容：**
- 第 5 章の混合モデルをベイズネットとして明示的に再構築
- 階層バージョン（混合重みの事前分布を上位に追加）
- 条件付き確率表を用いた Chibany の複数親を持つ弁当ネットワーク
- 観測された重みから隠れクラスターへの推論

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 8 章：ベイズネットワーク](../intro2/08_bayes_nets/)

**トピック：**
- 有向非巡回グラフ（DAG）
- マルコフ因数分解
- 条件付き確率表
- 祖先サンプリングと推論

---

### 条件付き独立性と d 分離
**ノートブック**: [📓 Open in Colab: `09_conditional_independence.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/09_conditional_independence.ipynb)

**扱う内容：**
- 実行可能なモデルとしての雨 / スプリンクラー / 濡れた床のコライダー
- 証拠への条件付けと重要度サンプリングによる事後分布の復元
- **説明消去**が数値的に起こる様子の観察（0.30 → 0.59 → 0.30）

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 9 章：条件付き独立性と d 分離](../intro2/09_conditional_independence/)

**トピック：**
- チェーン、フォーク、コライダーパターン
- d 分離
- マルコフブランケット
- 説明消去

---

### 因果ベイズネットと do 演算子
**ノートブック**: [📓 Open in Colab: `10_causal_bayes_nets.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/10_causal_bayes_nets.ipynb)

**扱う内容：**
- 観測モデルと介入モデルとしての喫煙 / 歯 / がんネットワーク
- モンテカルロによる P(がん | 歯) vs. P(がん | do(歯)) の計算
- グラフ手術から生じる「見る/する」ギャップ（≈0.098 vs. 0.052）の観察

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 10 章：因果ベイズネットと do 演算子](../intro2/10_causal_bayes_nets/)

**トピック：**
- do 演算子とグラフ手術
- 交絡因子
- 観測分布 vs. 介入分布
- パールの因果推論の梯子

---

### 情報理論
**ノートブック**: [📓 Open in Colab: `11_information_theory.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/11_information_theory.ipynb)

**扱う内容：**
- モンテカルロによるエントロピーと相互情報量の推定
- 無から相互情報量を生み出すコライダー
- I(雨; お茶) = 0 が I(雨; お茶 | 看板) ≈ 0.46 ビットに跳ね上がる — ビット単位での説明消去

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 11 章：情報理論](../intro2/11_information_theory/)

**トピック：**
- 情報量とエントロピー
- 相互情報量
- 情報量単位での独立性
- ビット単位のコライダー

---

### マルコフ連鎖
**ノートブック**: [📓 Open in Colab: `13_markov_chains.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/13_markov_chains.ipynb)

**扱う内容：**
- 遷移行列としての Chibany の弁当チェーン；そこからの系列サンプリング
- 任意の初期状態から 70/30 定常分布へ収束するべき乗法
- 固有値 1 の固有ベクトルとしての定常分布
- 3 状態の計算例

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 13 章：マルコフ連鎖](../intro2/13_markov_chains/)

**トピック：**
- マルコフ性と遷移行列
- 定常分布とべき乗法
- エルゴード性
- GenJAX による系列サンプリングと `jax.lax.scan`

---

### ネットワーク上のランダムウォーク
**ノートブック**: [📓 Open in Colab: `14_random_walks_networks.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/14_random_walks_networks.ipynb)

**扱う内容：**
- 隣接行列として表した Chibany の動物ネットワークを行正規化して遷移行列に変換
- ランダムウォークの定常分布：π ∝ 次数（橋の Cat が勝つ）
- シミュレーションによる訪問頻度の確認と次数則との一致
- 小さな有向ウェブ上での ε テレポートを用いた手製 PageRank

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 14 章：ネットワーク上のランダムウォーク](../intro2/14_random_walks_networks/)

**トピック：**
- グラフ、隣接行列、次数
- ノード上のマルコフ連鎖としてのランダムウォーク
- π ∝ 次数とその破綻条件（有向グラフ）
- PageRank

---

### 記憶探索
**ノートブック**: [📓 Open in Colab: `15_memory_search.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/15_memory_search.ipynb)

**扱う内容：**
- 小さな意味ネットワーク上のセンサード付きランダムウォーク
- センサリング関数（各動物を初回訪問時に報告）とアイテム間反応時間
- スイッチルールなしで現れる位置 1 が最も遅い「スイッチコスト」シグネチャ
- 流暢性リストからブロック構造を復元するシミュレーションベース（ABC）スケッチ

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 15 章：ランダムウォークとしての記憶探索](../intro2/15_memory_search/)

**トピック：**
- 意味的流暢性とクラスタリング/スイッチング
- センサリング；初回到達時間；IRT
- 1 つの過程からヒトの最適採餌曲線を復元する
- ウォークの逆算（U-INVITE、SNAFU）とシミュレーションベース推論

---

### モンテカルロ
**ノートブック**: [📓 Open in Colab: `16_monte_carlo.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/16_monte_carlo.ipynb)

**扱う内容：**
- モンテカルロ推定量：サイコロの目を平均して 3.5 を求める、ダーツ投げで π を推定する
- 棄却サンプリングと指示関数；重み $w = p/q$ を用いた重要度サンプリング
- 自己正規化重要度サンプリング（非正規化事後分布）と有効サンプルサイズ
- `model.importance` を用いた GenJAX 重要度サンプリング

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 16 章：モンテカルロ](../intro2/16_monte_carlo/)

**トピック：**
- サンプリングによる期待値と確率の推定；$1/\sqrt{n}$ レート
- 棄却サンプリング、逆 CDF 法、重要度サンプリング
- 提案分布の品質診断としての有効サンプルサイズ

---

### 粒子フィルタ
**ノートブック**: [📓 Open in Colab: `17_particle_filtering.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/17_particle_filtering.ipynb)

**扱う内容：**
- ノイズの多いセンサーのピングから廊下を進む Chibany の追跡
- 粒子フィルタのループ — 重み付け、リサンプリング、伝播 — と各ステップの役割
- 重みの縮退とリサンプリングが解決策である理由（それなしでの ESS の崩壊）
- 伝播ステップとしての GenJAX 運動モデル

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 17 章：粒子フィルタ](../intro2/17_particle_filtering/)

**トピック：**
- 状態空間モデル；逐次重要度サンプリング
- 重み付け → リサンプリング → 伝播；縮退
- ヒト推論の過程モデルとしての粒子フィルタ

---

### マルコフ連鎖モンテカルロ
**ノートブック**: [📓 Open in Colab: `18_markov_chain_monte_carlo.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/18_markov_chain_monte_carlo.ipynb)

**扱う内容：**
- 双峰ターゲット分布上のメトロポリス—ヘイスティングス；正規化定数がキャンセルされる理由
- 相関した 2 次元ガウス分布上のギブスサンプリング（常に受理）
- 混合、バーンイン、多峰分布の罠（2 つの初期値からの閉じ込められたチェーン）
- `assess` スコアリングプリミティブを用いた GenJAX での MH ステップの組み立て

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 18 章：マルコフ連鎖モンテカルロ](../intro2/18_markov_chain_monte_carlo/)

**トピック：**
- ターゲットを命中するチェーンの設計；詳細釣り合い条件
- メトロポリス—ヘイスティングスとギブスサンプリング
- バーンイン、混合、多峰ターゲット

---

### 心のサンプリング
**ノートブック**: [📓 Open in Colab: `19_sampling_the_mind.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/19_sampling_the_mind.ipynb)

**扱う内容：**
- MCMC with People：メトロポリス受理ステップとしての人の選択
- 階層ベータ—二項分布（弁当店）のハイブリッド ギブス—メトロポリスサンプラー
- ベータ—二項周辺分布（θ を積分消去）と平均/集中度の再パラメータ化
- 未見の新しいユニットに対する予測分布の読み取り

**関連チュートリアル章：**
- [チュートリアル 3（Intro2）、第 19 章：心のサンプリング](../intro2/19_sampling_the_mind/)

**トピック：**
- MCMC with People；選択から事前分布を復元する
- 階層モデル上のハイブリッド ギブス + メトロポリス
- ベータ—二項共役性と積分消去サンプラー

---

## 推奨学習パス

### パス 1：完全な初心者向け
1. `first_model.ipynb` - GenJAX の基礎はここから始める
2. `conditioning.ipynb` - 条件付き確率を学ぶ
3. `bayesian_learning.ipynb` - ベイズ更新をマスターする
4. `02_first_genjax_model.ipynb` - 最初の完全なモデルを構築する
5. `gaussian_bayesian_interactive_exploration.ipynb` - 連続確率を探索する
6. `solution_1_gaussian_bayesian_update.ipynb` - ガウス推論を練習する
7. `solution_2_gaussian_clusters.ipynb` - 混合モデルを学ぶ
8. `dpmm_interactive.ipynb` - 発展：無限混合

### パス 2：ベイズ学習に焦点を当てる
1. `bayesian_learning.ipynb` - 離散ベイズの定理
2. `gaussian_bayesian_interactive_exploration.ipynb` - 連続ベイズ推論
3. `solution_1_gaussian_bayesian_update.ipynb` - ガウス共役事前分布
4. `solution_2_gaussian_clusters.ipynb` - 混合モデル推論
5. `dpmm_interactive.ipynb` - ベイズノンパラメトリクス

### パス 3：クイックインタラクティブツアー
1. `02_first_genjax_model.ipynb` - インタラクティブなパラメータ探索
2. `gaussian_bayesian_interactive_exploration.ipynb` - スライダー付きベイズ学習
3. `dpmm_interactive.ipynb` - 自動クラスタリング

### パス 4：グラフィカルモデルと因果推論
1. `08_bayes_nets.ipynb` - グラフとしてモデルを描く
2. `09_conditional_independence.ipynb` - d 分離と説明消去
3. `10_causal_bayes_nets.ipynb` - 見ることと行うこと（do 演算子）
4. `11_information_theory.ipynb` - ビット単位で依存性を測定する

### パス 5：マルコフ連鎖、ネットワーク、記憶
1. `13_markov_chains.ipynb` - 遷移行列と定常分布
2. `14_random_walks_networks.ipynb` - グラフ上のランダムウォーク、π ∝ 次数、PageRank
3. `15_memory_search.ipynb` - 意味ネットワーク上のセンサード付きランダムウォークとしての想起

### パス 6：サンプリングとモンテカルロ
1. `16_monte_carlo.ipynb` - サンプリングによる推定；重要度サンプリングと有効サンプルサイズ
2. `17_particle_filtering.ipynb` - ストリーミング推論；重み付け → リサンプリング → 伝播
3. `18_markov_chain_monte_carlo.ipynb` - ターゲットを命中するチェーンの設計；メトロポリス—ヘイスティングスとギブス
4. `19_sampling_the_mind.ipynb` - MCMC with People と階層ベータ—二項分布のサンプラー

---

## ノートブックの使い方のヒント

**はじめに：**
- 「Open in Colab」をクリックして任意のノートブックを起動する
- セルを順番に実行する（Shift+Enter）
- パラメータ値を変えて実験する

**インタラクティブウィジェット：**
- 多くのノートブックにはスライダーとコントロールが含まれている
- パラメータを調整してリアルタイムで結果の更新を確認する
- エッジケースを理解するために極端な値を試す

**作業の保存：**
- ファイル → ドライブにコピーを保存（Google Drive に保存される）
- 実験やメモが保存される
- 編集したノートブックを他の人と共有できる

**トラブルシューティング：**
- コードが実行されない場合は、ランタイム → ランタイムを再起動 を試す
- セルを上から下へ順番に実行していることを確認する
- 必要なパッケージがすべてインストールされていることを確認する（Colab では通常自動）

---

## 全ノートブック一覧

| ノートブック | チュートリアル | トピック | 難易度 |
|----------|----------|--------|------------|
| `first_model.ipynb` | チュートリアル 1 | 離散確率、基礎 | ⭐ 初心者 |
| `conditioning.ipynb` | チュートリアル 1 | 条件付き確率 | ⭐ 初心者 |
| `bayesian_learning.ipynb` | チュートリアル 1 & 2 | ベイズ推論 | ⭐⭐ 中級 |
| `02_first_genjax_model.ipynb` | チュートリアル 2 | GenJAX プログラミング | ⭐⭐ 中級 |
| `gaussian_bayesian_interactive_exploration.ipynb` | チュートリアル 3 | 連続ベイズ、混合 | ⭐⭐⭐ 上級 |
| `solution_1_gaussian_bayesian_update.ipynb` | チュートリアル 3 | ガウス推論 | ⭐⭐⭐ 上級 |
| `solution_2_gaussian_clusters.ipynb` | チュートリアル 3 | 混合モデル | ⭐⭐⭐ 上級 |
| `dpmm_interactive.ipynb` | チュートリアル 3 | ベイズノンパラメトリクス | ⭐⭐⭐⭐ エキスパート |
| `07_generalization.ipynb` | チュートリアル 3 | 概念学習、サイズ原理 | ⭐⭐⭐ 上級 |
| `08_bayes_nets.ipynb` | チュートリアル 3 | ベイズネットワーク、DAG | ⭐⭐⭐ 上級 |
| `09_conditional_independence.ipynb` | チュートリアル 3 | d 分離、説明消去 | ⭐⭐⭐ 上級 |
| `10_causal_bayes_nets.ipynb` | チュートリアル 3 | 因果推論、do 演算子 | ⭐⭐⭐ 上級 |
| `11_information_theory.ipynb` | チュートリアル 3 | エントロピー、相互情報量 | ⭐⭐⭐ 上級 |
| `13_markov_chains.ipynb` | チュートリアル 3 | マルコフ連鎖、定常分布 | ⭐⭐⭐ 上級 |
| `14_random_walks_networks.ipynb` | チュートリアル 3 | ランダムウォーク、PageRank、π ∝ 次数 | ⭐⭐⭐ 上級 |
| `15_memory_search.ipynb` | チュートリアル 3 | センサード付きウォーク、記憶流暢性 | ⭐⭐⭐ 上級 |
| `16_monte_carlo.ipynb` | チュートリアル 3 | モンテカルロ、重要度サンプリング、ESS | ⭐⭐⭐ 上級 |
| `17_particle_filtering.ipynb` | チュートリアル 3 | 粒子フィルタ、逐次推論 | ⭐⭐⭐ 上級 |
| `18_markov_chain_monte_carlo.ipynb` | チュートリアル 3 | メトロポリス—ヘイスティングス、ギブス、混合 | ⭐⭐⭐ 上級 |
| `19_sampling_the_mind.ipynb` | チュートリアル 3 | MCMC with People、Kemp 階層サンプラー | ⭐⭐⭐⭐ エキスパート |

---

**ヘルプが必要ですか？** [メインチュートリアルページ](../)に戻るか、用語の定義については[用語集](../glossary/)を参照してください。

**ノートブックを楽しんでいますか？** このチュートリアルシリーズは[日本確率計算コンソーシアム協会（JPCCA）](https://jpcca.org/)の支援を受けて提供されています。
