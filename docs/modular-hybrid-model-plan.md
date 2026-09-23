# 交換可能な個別NN・統合E2E・決定論ハイブリッド実装計画

更新日: 2026-09-22

## 目的

Yamabikoのお手本再現モデルを、開発時には機能ごとに独立して学習・評価でき、配備時には一つの計算グラフとして高速実行できる構造へ整理する。同じ入出力契約を持つ決定論的アルゴリズムも用意し、任意のブロックをNN版、決定論版、Oracle診断版へ交換できるようにする。

この計画では、次の3方式を同じ評価集合で並行比較する。

1. `modular_learned`: 個別学習済みNNを凍結して接続する
2. `joint_e2e`: 個別NNを接続後、補助損失を残して全体微調整する
3. `hybrid`: 任意のブロックを決定論的アルゴリズムへ交換する

個別NN接続と一つながりのNNは排他的ではない。個別モジュールを単一の `CompositeImitationModel` に登録すれば、モジュール境界、個別checkpoint、個別評価を維持したまま、最終的に一つのforward graphへexportできる。

## 全体構成: 1つの耳・3つのモード・2種類の記憶（2026-09-23）

お手本の入力と自分の演奏の入力は、同じマイクと同じNeural Ear（同じ重み）を通る。二本のパイプラインに見えるのは、モードによって耳の出力の行き先が変わるためで、全体としては一つのNNである。同じ耳で聴くので、マイク特性や部屋の響きによる偏りは比較の段で打ち消される。

| モード | 入力 | お手本の記憶 | 機体の記憶 | 出力（PWM） |
|---|---|---|---|---|
| 聴く | お手本 | 書く | 読まない | なし |
| 学習（較正） | 自分の音 | 読む（較正曲） | 書く | あり |
| 演奏 | 自分の音 | 読む | 読む | あり |

- お手本の記憶は次のお手本を聴くまで、機体の記憶は次の学習モードまで残る。演奏中の速い補正はその場限り。
- 鳴禽の歌学習（手本を聴いて鋳型を覚える時期と、自分の声を聴いて合わせる時期）や、発話運動制御のDIVAモデル（聴覚目標・フィードフォワード・フィードバック）と同じ構造である。

### 演奏側の役割分担

| 段 | 役割 |
|---|---|
| Position Planner | 記憶したお手本から、どのタイミングでどの位置にいるべきかを決める。休符中に次の音へ先回りする計画を含む |
| Motor Control | 10 msごとにPWMを計算し直す。エンコーダが無いので「今どこにいるか」の推定も含む |
| Feedback（その場） | 自己音の音程で位置推定と指令を直す。Motor Controlとほぼ一体 |
| 機体の記憶（次回へ） | 機体のくせを保持し、PlannerとMotor Controlへ渡す |

### 閉管シミュレーターでの検証から分かったこと

- 閉管笛で変位に線形なのは周期（1/f）で、音程（cent）ではない。1オクターブを直線近似すると±52 cent。`PhysicalPlantConfig.realistic()` を以後の標準とする。
- エンコーダ無しでは「音程を位置センサーにする」オブザーバが最も効く（推測航法のみ約300 → 90 cent）。
- 残りの誤差の大半は音の立ち上がりのモーター移動時間で、距離に応じた先回りが効く（90 → 56 cent）。
- 発音中は音程サーボにより笛の切片・傾きの誤差が打ち消されるため、真の係数を与えても改善は約1 cent。機体の記憶で覚える価値が高いのは、笛の係数よりモーター側（速さ・不感帯・動き出しのタイミング）である。

## 共通パイプライン

```text
reference waveform
  → Audio Frontend / Neural Ear
  → Tempo & Beat
  → Musical Memory
  → Temporal Aligner
  → Position Planner
  → Position Controller ──────────────────────────┐
                                                  ├→ actuator action
self waveform → Neural Ear → Error Comparator → Feedback Residual
previous actions → Motor State Estimator ─────────┘
```

推論周期は100 Hzを基本とする。BPM、拍位相、音符・休符は音楽時間を表し、100 Hzクロックはセンサーとアクチュエータを更新する物理時間として扱う。

## ブロック契約

各ブロックはNN版と決定論版で同じTensor形状、単位、マスク、信頼度を返す。中間表現は可能な限り以下の名前付き契約に固定する。

### `EarOutput`

- `embedding`: 音響特徴
- `pitch`: 正規化した連続音程
- `voiced_probability`: 発音確率
- `confidence`: 推定信頼度
- `mask`: 有効フレーム

### `BeatOutput`

- `tempo_bpm`: BPMまたはBPM候補分布
- `beat_phase`: 連続拍位相
- `bar_phase`: 小節内位相
- `subdivision_phase`: 1/8、1/16等の細分位置
- `boundary_probability`: 拍・小節境界確率
- `confidence`: 拍が明確である確率

### `MusicalMemoryOutput`

- `cell_embedding`: 拍セルごとの潜在表現
- `continuous_pitch`: セル内の連続音程
- `pitch_slope`: セル内音程傾斜
- `voiced_probability`: 発音確率
- `rest_probability`: 休符確率
- `onset_probability`, `offset_probability`
- `harmonic_context`: 任意の長期和声潜在表現
- `cell_mask`: 有効セル

### `PlannerOutput`

- `target_position`: 目標プランジャー位置
- `target_velocity`: 望ましい移動速度
- `urgency`: 次の発音までに移動すべき緊急度
- `valve_probability`: バルブ開確率
- `done_probability`: 演奏終了確率

### `MotorStateOutput`

- `estimated_position`: 推定現在位置
- `estimated_velocity`: 推定速度
- `rig_embedding`: 個体差の低速潜在表現
- `confidence`: 状態推定信頼度

### `ErrorOutput`

- `signed_pitch_error`: 目標－自己演奏の符号付き誤差
- `timing_error`: 発音・拍タイミング誤差
- `voicing_mismatch`: 発音／休符不一致
- `confidence`: 比較の信頼度

## 実装モード

### 1. 個別NN接続モード

各Gateで採用されたcheckpointを読み込み、全パラメーターを凍結して接続する。中間Tensorを保存でき、任意の地点でbefore/after WAVを生成できる。接続時の入力分布変化を検出する基準モデルとする。

### 2. 統合E2Eモード

個別NNを `CompositeImitationModel` のsubmoduleとして登録し、最終的な演奏損失から全段へ勾配を流す。ただし中間能力の崩壊を防ぐため、以下の補助損失を残す。

```text
total loss =
    physical performance loss
  + neural ear pitch/voice loss
  + tempo/phase loss
  + musical memory reconstruction loss
  + planner position loss
  + motor state loss
  + signed error loss
  + stability/actuation penalty
```

最初は後段のみを解凍し、次に各段へ小さい学習率を設定する。候補checkpointは総合報酬ではなく固定ホールドアウトの全Gateで選ぶ。E2E微調整後に一つでも既存Gateが退化した場合、そのcheckpointは採用しない。

### 3. ハイブリッドモード

設定ファイルで各ブロックの実装を選択する。

```yaml
mode: hybrid
ear: learned
tempo_beat: learned
musical_memory: learned
temporal_aligner: deterministic_clock
position_planner: learned
motor_state: deterministic_dead_reckoning
position_controller: deterministic_pid
error_comparator: learned
feedback: learned
```

想定する決定論版は以下とする。

| ブロック | NN版 | 決定論版・比較版 |
|---|---|---|
| Audio Frontend | raw waveform CNN | STFT/CQT frontend |
| Neural Ear | learned pitch/voice | YIN等、評価・ベースライン限定 |
| Tempo/Beat | learned tempo/phase | 固定BPM clock、既知BPM |
| Musical Memory | GRU/Transformer | 与えた拍グリッド、Oracle score |
| Temporal Aligner | monotonic attention | 固定100 Hz/固定BPM index |
| Position Planner | learned mapper | 管長物理式 |
| Motor State | recurrent estimator | nominal dead reckoning、encoder |
| Position Controller | learned policy | PID/open-loop/oracle |
| Error Comparator | learned comparator | target-self単純差 |
| Feedback | learned residual | 比例補正、無補正 |

決定論版を本番経路として許可するかは実験ごとに明記する。コンテストで「完全E2E」と主張する評価では全learned構成を使用し、hybridは原因分析、性能上限、フェイルセーフ、アブレーションに用いる。

## ソフトウェア構造

```text
flute_rl/yamabiko/modules/
  contracts.py
  ear.py
  tempo_beat.py
  musical_memory.py
  aligner.py
  planner.py
  motor_state.py
  controller.py
  comparator.py
  feedback.py
  deterministic.py
  composite.py
```

`build_block(kind, implementation, config)` が共通インターフェースを満たす実装を返す。決定論版も `nn.Module` 互換のwrapperにし、同じforwardとexport経路を使う。状態を持つブロックは `initial_state`, `step`, `reset_mask` を共通化する。

checkpointには次を保存する。

- 全体format version
- ブロック種別と実装名
- 正規化定数と単位
- 各submoduleのstate dict
- 学習データversion、seed、git SHA
- 単独Gateと接続Gateの評価結果
- 対応するWAV manifest

## 学習手順

1. 各ブロックを正解入力で単独学習する
2. 未知データで単独Gateを評価する
3. 前段NNの実出力を入力する接続Gateを評価する
4. 合格した前段を凍結して次段を学習する
5. 全ブロックを接続し、凍結状態で物理閉ループ評価する
6. 後段から順に解凍して接続誤差を吸収する
7. 全体を低学習率でE2E微調整する
8. 個別Gate、接続Gate、物理Gateをすべて再評価する
9. 個別接続、joint E2E、主要hybrid構成を同一seedで比較する

## 評価マトリクス

各ブロックについて最低3種類を測る。

1. `oracle input`: 正解の前段出力を与える単独性能
2. `predicted upstream`: 実際の前段NN出力を与える接続性能
3. `closed loop`: 物理シミュレータをモデル自身の操作で進める性能

比較表には次を含める。

- 精度、P90/P99、失敗率
- レイテンシ、メモリ、モデルサイズ
- UNO Qでの100 Hz deadline達成率
- clean、音響乱数化、rig乱数化、実録音
- upstreamを決定論版へ交換した場合の差
- 個別接続からjoint E2Eにした場合の改善と退化

## WAVとHTML

各Gate完了時に次の音を固定ホールドアウトの中央値例で公開する。

- ブロック入力を診断用に可聴化したbefore WAV
- ブロック出力を診断用に可聴化したafter WAV
- 制御段以降は物理シミュレータを通した演奏WAV
- 決定論版、個別NN接続版、joint E2E版の同一曲比較
- 不採用試行も削除せずattempt履歴として保存

診断用sonificationと物理演奏音をHTMLで明確に区別する。manifest JSONを唯一の数値source of truthとし、HTMLへ成功値を手書きしない。

## UNO Q配備

開発時はPC容量を優先する。成立後に以下を行う。

1. block別プロファイルで支配的コストを特定
2. hidden幅、層数、拍セル長をアブレーション
3. teacherからUNO Q studentへ蒸留
4. INT8または適切な量子化を検証
5. Python境界を避け、全learned構成を一つのexport graphへ統合
6. hybridフェイルセーフを別設定として保持

目標は平均速度だけでなく、100 Hz周期に対するP99レイテンシを満たすこととする。

## 完了条件

- 全ブロックが単独Gateとpredicted-upstream Gateを通過
- 個別NN接続モデルが物理Gateを通過
- joint E2Eが個別接続を改善し、既存Gateを壊さない
- 少なくとも3種類のhybrid構成と全learned構成を同条件比較
- UNO Q用exportが実時間制約を満たす
- 全構成のcheckpoint、manifest、WAV、HTMLが再生成可能

