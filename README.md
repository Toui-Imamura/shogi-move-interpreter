# shogi-move-interpreter

将棋の指し手の意図を、MCTSによる未来局面の変化と人間が解釈しやすい特徴量から説明するための研究用システム。

## 1. 研究目的

本研究では、単純な勝敗予測ではなく、「なぜこの指し手が選ばれたのか」を説明することを目的とする。

強い局面評価と説明可能な特徴量評価を分離し、MCTSの局面評価にはNNUE等の学習型評価器を利用し、指し手解釈にはF01～F40を利用する。

F01～F40は初期目標として40個を設定する。実験によって、相関が高い特徴量、情報量が低い特徴量、解釈に寄与しない特徴量は統合・削除する。

## 2. 基本設計

### 2.1 局面

- `S0`: 指し手を指す直前の現在局面
- `Sv`: MCTSによって得られた未来局面
- `S0 -> S1 -> ... -> Sv`: 1本の探索変化

### 2.2 先手・後手

基本特徴量は原則として黒・白を個別に計算する。

`D_i(S) = f_i^Black(S) - f_i^White(S)`

`M_i(S) = (f_i^Black(S) + f_i^White(S)) / 2`

### 2.3 B特徴量

現在局面と未来局面との差を基本とする。

`Δf_i^(v) = f_i(Sv) - f_i(S0)`

### 2.4 MCTS集約

探索変化 `v` の訪問回数を `Nv` とする。

`αv = Nv / Σu Nu`

`Δ̄f_i = Σv αv Δf_i^(v)`

訪問回数は「正しさ」や「意図の確率」ではなく、探索結果を集約するための重みとして扱う。

### 2.5 正規化

基本的な連続値特徴量は原則として[-1,1]へ正規化する。

`norm(x) = tanh(x / c)`

`c` は特徴量ごとに実験で決定する。

### 2.6 動的重み

手数だけで序盤・中盤・終盤を決定しない。

`Z = (P, D, I)`

から序盤・中盤・終盤の所属度

`P(S) = [PE, PM, PL]`

を求め、

`wi(S) = PE wiE + PM wiM + PL wiL`

とする。

## 3. F01～F40

| ID | 名称 | 分類 | カテゴリ |
|---|---|---|---|
| F01 | 駒得差 | A | Material |
| F02 | 持ち駒価値差 | A | Material |
| F03 | 駒交換による戦力変化 | B | Exchange |
| F04 | 大駒バランス | A | Material |
| F05 | 小駒構成 | A | Material |
| F06 | 駒の利き | A | Activity |
| F07 | 駒の移動可能性 | A | Activity |
| F08 | 重要地点への利き | A | Activity |
| F09 | 遊び駒の改善 | B | Activity |
| F10 | 活動性の変化 | B | Activity |
| F11 | 相手玉への攻撃圧力 | A | Attack |
| F12 | 攻撃参加駒数 | A | Attack |
| F13 | 駒得形成過程 | B | Material |
| F14 | 攻撃拠点形成 | B | Attack |
| F15 | 攻撃継続性 | B | Attack |
| F16 | 駒の進出変化 | B | Activity |
| F17 | 玉周辺の守備駒数 | A | Defense |
| F18 | 玉周辺の利き | A | Defense |
| F19 | 攻撃圧力変化 | B | Attack |
| F20 | 攻撃参加駒数の変化 | B | Attack |
| F21 | 攻撃集中度の変化 | B | Attack |
| F22 | 脅威形成の変化 | B | Attack |
| F23 | 防御力の変化 | B | Defense |
| F24 | 玉安全性の変化 | B | Defense |
| F25 | 相手攻撃への対応力の変化 | B | Defense |
| F26 | 玉の安全性 | A | KingSafety |
| F27 | 守備駒配置の変化 | B | Defense |
| F28 | 陣形の変化 | B | Formation |
| F29 | 囲いの進展 | B | Formation |
| F30 | 弱点の形成・解消 | B | Formation |
| F31 | 歩の連結性 | A | Formation |
| F32 | 攻守転換度 | C | Strategy |
| F33 | 戦力配置の変化 | B | Strategy |
| F34 | 戦略方向性変化 | C | Strategy |
| F35 | 複数特徴量の一貫性 | C | Integration |
| F36 | 変化の持続性 | C | Integration |
| F37 | MCTS変化頻度 | C | Integration |
| F38 | MCTS訪問重み付き変化 | C | Integration |
| F39 | 特徴量変化の集中度 | C | Integration |
| F40 | 指し手による局面転換度 | C | Integration |

## 4. 特徴量詳細

### F01 駒得差

`extract_material_balance(board)`

`F01 = Material_Black - Material_White`

初期駒価値:

- 歩: 1
- 香: 3
- 桂: 3
- 銀: 5
- 金: 6
- 角: 8
- 飛: 9
- 成歩・成香・成桂・成銀: 9
- 馬: 11
- 龍: 12

正規化:

`F01_norm = tanh(F01 / 20)`

未来局面でも再計算し、`ΔF01 = F01(Sv) - F01(S0)` とする。

F01は最終的な物質差を表す。F03は交換過程、F13は駒得形成過程を表す。

### F02 持ち駒価値差

`extract_hand_value(board)`

`F02 = HandValue_Black - HandValue_White`

持ち駒価値:

- 歩: 1
- 香: 3
- 桂: 3
- 銀: 5
- 金: 6
- 角: 8
- 飛: 9

正規化:

`F02_norm = tanh(F02 / 20)`

現在持っている攻撃材料や交換後の持ち駒状態を表す。

### F03 駒交換による戦力変化

`extract_exchange_transition(before, move, after)`

`analyze_exchange_sequence(boards, moves)`

駒取り、直接交換、交換連鎖を検出する。

内部指標:

- E1: 交換発生回数
- E2: 交換された駒の価値
- E3: 大駒の交換量
- E4: 小駒の交換量
- E5: 盤上から持ち駒への移行
- E6: 駒構成の変化

`F03 = Σ βj Ej`

βは実験または学習によって決定する。

F03は交換というイベントと過程を表し、単純な`ΔF01 + ΔF02`とはしない。

### F04 大駒バランス

`extract_major_piece_balance(board)`

対象:

- 角
- 飛車
- 馬
- 龍

`F04 = MajorValue_Black - MajorValue_White`

価値だけでなく、角・飛車・馬・龍の有無も保持する。

### F05 小駒構成

`extract_minor_piece_composition(board)`

小駒の構成ベクトル:

`C = (P, L, N, S, G)`

成駒は必要に応じて対応する未成駒種へ戻した構成も保持する。

単一スカラーだけでなく、種類別ベクトルを内部的に保持する。

### F06 駒の利き

`extract_attack_control(board)`

各マスについて黒・白が何枚の駒で利いているかを計算する。

`Control_c(x) = # { p | p attacks x }`

局面全体の総利き量を正規化する。

### F07 駒の移動可能性

`extract_mobility(board)`

- 各側の合法手数
- 駒ごとの平均合法手数
- 駒種別の分布

を計算する。

`Mobility_c = #LegalMoves_c`

### F08 重要地点への利き

`extract_key_square_control(board)`

重要地点集合Kへの利きを計算する。

`F08_c = Σx∈K Control_c(x)`

対象には中央、相手玉周辺、攻撃拠点候補などを含める。

Kは玉位置や攻撃方向に応じて動的に生成する。

### F09 遊び駒の改善

`compute_idle_piece_change(current, future)`

遊び駒を、

- 利きが少ない
- 合法手が少ない
- 重要地点への関与が少ない

など複数条件から判定する。

`IdleChange = Idle(S0) - Idle(Sv)`

遊び駒が減少すれば改善とする。

### F10 活動性の変化

`compute_activity_change(current, future)`

`Activity = a1 Control + a2 Mobility + a3 KeyControl`

`ΔF10 = Activity(Sv) - Activity(S0)`

F06・F07を単純加算するのではなく、全体活動性として扱う。

### F11 相手玉への攻撃圧力

`extract_king_attack_pressure(board, player)`

`Pressure = a1 KingControl + a2 Attackers + a3 Invasion`

相手玉周辺の利き、玉への距離、攻撃参加駒、侵入駒などを組み合わせる。

チェックそのものだけを攻撃圧力とはしない。

### F12 攻撃参加駒数

`extract_attacking_piece_count(board, player)`

攻撃領域へ利きを持つ、または攻撃拠点へ進出している駒数を数える。

### F13 駒得形成過程

`compute_material_gain_process(sequence)`

F01の最終差ではなく、駒得がどの手で形成されたかを追跡する。

`GainProcess = Σt max(0, ΔF01_t)`

連続する同一イベントを二重計上しない。

### F14 攻撃拠点形成

`compute_attack_base_change(current, future)`

相手陣や玉周辺で、

- 攻撃駒が複数利く地点
- 侵入可能地点

を攻撃拠点候補とする。

`BaseGain = Base(Sv) - Base(S0)`

### F15 攻撃継続性

`compute_attack_continuity(sequence)`

`Continuity = #{t : Pressure_t >= threshold} / T`

攻撃圧力が複数未来局面で維持されるかを測る。

### F16 駒の進出変化

`compute_piece_advance(current, future)`

`Advance = Σp Development(p)`

単純な段数だけでなく、駒種と方向を考慮する。

### F17 玉周辺の守備駒数

`extract_king_defender_count(board, player)`

玉周辺の一定領域、約8マス程度に存在する自駒数を数える。

### F18 玉周辺の利き

`extract_king_area_control(board, player)`

`KingAreaControl = OwnControl - OpponentControl`

自玉周辺の防御態勢と侵入危険を表す。

### F19 攻撃圧力変化

`compute_attack_pressure_change(current, future)`

`ΔF19 = F11(Sv) - F11(S0)`

F11が状態、F19が変化を表す。

### F20 攻撃参加駒数の変化

`compute_attacker_count_change(current, future)`

`ΔF20 = F12(Sv) - F12(S0)`

攻撃に新しい駒を参加させたことなどを表す。

### F21 攻撃集中度の変化

`compute_attack_concentration_change(current, future)`

`px = Control(x) / Σy Control(y)`

`Concentration = Σx px²`

広く攻めるのか、特定地点へ攻撃を集中するのかを区別する。

### F22 脅威形成の変化

`compute_threat_change(current, future)`

脅威の例:

- 次に駒を取れる
- 王手
- 詰めろ候補
- 重要駒への強い攻撃
- 受けを要求する攻撃

単一条件ではなく複数指標を統合する。

### F23 防御力の変化

`compute_defense_change(current, future)`

`Defense = a1 Defenders + a2 KingControl + a3 Response`

守備駒数、玉周辺の利き、相手攻撃への応答可能性を統合する。

### F24 玉安全性の変化

`compute_king_safety_change(current, future)`

`ΔF24 = F26(Sv) - F26(S0)`

玉を安全にする、危険にさらすなどを表す。

### F25 相手攻撃への対応力の変化

`compute_counter_attack_response(current, future)`

以下の複数の対応可能性を評価する。

- 受ける
- 逃げる
- 攻め合う
- 反撃する

`Response = a1 LegalDefense + a2 KingEscape + a3 CounterAttack`

### F26 玉の安全性

`extract_king_safety(board, player)`

`KingSafety = a1 OwnControl - a2 EnemyControl + a3 Escape - a4 Attackers`

入力:

- 相手からの利き
- 自駒による防御
- 玉の合法逃げ手
- 周辺の空きマス
- 攻撃駒数

囲いを単純な玉周辺駒数だけで判定しない。囲いの進展はF29で扱う。

### F27 守備駒配置の変化

`compute_defensive_placement_change(current, future)`

玉周辺および攻撃方向に対する守備駒の配置変化を測る。

### F28 陣形の変化

`compute_formation_change(current, future)`

以下を利用する。

- 駒間距離
- 相互利き
- 守備領域への配置
- 攻撃領域への配置

Kaneko et al. の駒関係モデルを参考とする。

### F29 囲いの進展

`compute_castle_progress(current, future)`

以下から囲いの完成度を連続値で評価する。

- 玉の位置
- 金銀の配置
- 玉周辺の防御密度
- 相互利き
- 攻撃方向に対する守備

穴熊・美濃などの名称は補助的ラベルとして使用可能だが、特徴量本体は特定戦法に依存しない。

### F30 弱点の形成・解消

`compute_weakness_change(current, future)`

弱点:

- 守備が薄い地点
- 相手の複数利きが成立する地点
- 歩による防御が不足する地点
- 侵入を許しやすい地点

`WeaknessChange = Weakness(S0) - Weakness(Sv)`

### F31 歩の連結性

`extract_pawn_connectivity(board)`

`Connectivity = connected pawns / relevant pawns`

補助情報:

- 孤立
- 連結
- 歩の位
- 交換可能性

### F32 攻守転換度

`compute_attack_defense_transition(feature_deltas)`

`Transition = (ΔAttack - ΔDefense) / (|ΔAttack| + |ΔDefense| + ε)`

守りから攻めへ、攻めから受けへ移った可能性を示す。

### F33 戦力配置の変化

`compute_force_distribution_change(current, future)`

盤面を複数領域へ分け、各領域の戦力を比較する。

`Distribution = [D1, D2, ..., Dn]`

F21が攻撃の集中、F33が盤全体の戦力配置を担当する。

### F34 戦略方向性変化

`compute_strategy_direction(feature_deltas)`

特徴量を5方向へまとめる。

- Material
- Attack
- Defense
- Activity
- Formation

`R = [RM, RA, RD, RC, RF]`

現在と未来の方向ベクトルの差から戦略方向の変化を求める。

### F35 複数特徴量の一貫性

`compute_feature_consistency(feature_deltas)`

同じ意味方向を持つ特徴量が同時に変化している度合いを計算する。

例えば、

- 攻撃圧力増加
- 攻撃参加駒数増加
- 攻撃拠点形成

が同時に発生した場合、一貫性が高いとする。

### F36 変化の持続性

`compute_change_persistence(sequence_features)`

`Persistence_i = #{t : sign(Δf_i,t) = sign(Δf_i,end)} / T`

一時的な変化と継続する戦略的変化を区別する。

### F37 MCTS変化頻度

`compute_mcts_change_frequency(variations)`

`Frequency_i = #variations with change_i / #variations`

複数の探索結果で繰り返し観測される変化を根拠として扱う。

**頻度は意図の確率ではない。**

### F38 MCTS訪問重み付き変化

`compute_visit_weighted_change(variations)`

`Δ̄f_i = Σv (Nv / Σu Nu) Δf_i^(v)`

訪問回数を重みとして複数の探索系列を集約する。

**訪問回数は意図確率ではない。**

### F39 特徴量変化の集中度

`compute_feature_change_concentration(feature_deltas)`

`pi = |Δfi| / Σj |Δfj|`

`Concentration = Σi pi²`

少数の特徴量だけが大きく変化している場合は高く、多数へ分散している場合は低くなる。

解釈候補を2～3個へ絞る際に利用する。

### F40 指し手による局面転換度

`compute_move_transition_degree(current, future, feature_deltas)`

`TransitionDegree = sqrt(Σi λi (Δfi)²)`

- 小さい: 局面の微調整
- 中程度: 既存方針の強化
- 大きい: 局面の方針転換

大きいから良い手とは限らない。

## 5. 特徴量の役割分担

### Material

F01 物質差  
F02 持ち駒差  
F03 交換過程  
F04 大駒構成  
F05 小駒構成  
F13 駒得形成過程

### Activity

F06 利き  
F07 移動可能性  
F08 重要地点  
F09 遊び駒  
F10 活動性  
F16 進出

### Attack

F11 攻撃圧力  
F12 攻撃参加駒数  
F14 攻撃拠点  
F15 攻撃継続性  
F19 攻撃圧力変化  
F20 攻撃参加駒数変化  
F21 攻撃集中  
F22 脅威形成

### Defense / King Safety

F17 守備駒数  
F18 玉周辺の利き  
F23 防御力変化  
F24 玉安全性変化  
F25 対応力  
F26 玉安全性  
F27 守備配置

### Formation

F28 一般的陣形  
F29 囲い  
F30 弱点  
F31 歩形

### Strategy / Integration

F32 攻守転換  
F33 戦力配置  
F34 戦略方向  
F35 複数根拠の一貫性  
F36 持続性  
F37 MCTS出現頻度  
F38 MCTS訪問重み  
F39 変化集中度  
F40 局面転換度

## 6. 共通データ構造

```text
FeatureValue
    black
    white
    difference
    mean

Feature
    extract(board)

AFeature
    extract(board)

BFeature
    compute_delta(current, future)

CFeature
    aggregate(feature_values, mcts_results)