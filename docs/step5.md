# step5 — 共通の耳（2026-11-24、録音とシミュレーションで評価、実機の評価はまだ）

笛も口笛も、キャリブレーションとその日のずれの測定も、**同じ 1 つの耳の NN** で聞く版。推論の経路から YIN が外れた。
[models/step5/](../models/step5/) にあり、step1〜4 と並べて置いてある。計画は [nn-e2e-plan.md](nn-e2e-plan.md) の step5。

## 中身

| 部分 | step4 | step5 |
|---|---|---|
| お手本の口笛 → 音程 | YIN（`whistle_track`） | **耳の NN**（`Ear.track`、10 ms ごとに直前の 20 ms） |
| キャリブレーション・ずれの測定の音程 | YIN | **耳の NN**（サーバーの `cmd_calibrate`・`cmd_tune` に `ear="nn"`） |
| 演奏中のフィードバック | 耳の NN（`ear_01`、笛の音域だけ） | **耳の NN**（`ear_04`、笛と口笛） |
| アクチュエータ・弁 | NN（`nn_f80_t4_s06`・`valve_t4_01`） | 同じ |

`profile.json` の `"common_ear": true` で、口笛コピーの画面（`scripts/yamabiko_web.py`）が口笛の解析とキャリブレーションにも耳の NN を使う。
口笛コピーの画面は、口笛の音声を `runs/real/whistle/<日時>_whistle.npz` に残すようになった（実機の口笛で耳を学び直すためのデータ）。

## 耳の NN（`ear_04`）

- 音域：25 cent 刻み 178 区分、−600〜+3825 cent（約 310 Hz〜4 kHz）。step3・4 の `ear_01` は 96 区分（〜1250 Hz）
- 入力：直前の 20 ms（320 サンプル）と、**そのフレームの音量**（dBFS）。波形は音量をそろえてから読むので、音量は別に渡す
- 中身は step3 と同じ 3 層の 1 次元 CNN＋全結合 2 層。`EarConfig(low_cents, bins, level)` で作り、重みのファイルに区分と音量の有無を書く（`ear_01` もそのまま読める）
- 学習データ：
  - 実機の笛の録音 28 分（step3 と同じ）
  - **合成の口笛** 30 万フレーム（`flute_rl/yamabiko/ear_synth.py`）：500 Hz〜3.8 kHz の音を、グライド・区切り・ビブラート・揺らぎ・立ち上がりと余韻・第 2 倍音・息の雑音つきで作り、10 ms ごとの正しい音程も同時に出す。実機の静かなフレームを下に重ねる。−58 dB 未満のフレームは「鳴っていない」（実機の笛のラベルと同じ基準）
  - **雑音だけのフレーム** 6 万（白色・ピンク・ブラウンの雑音とクリック、−75〜−35 dB）：「鳴っていない」
- 学習：GPU で約 30 秒

```bash
python scripts/yamabiko_ear_train.py --wide --level --synth 300000 --noise 60000 --demo runs/ear/demo --out runs/ear/ear_04
python scripts/export_step_models_onnx.py models/step5
```

`runs/ear/demo/` はデモ動画 3 本（`media/step1/デモ*.mp4`）の音声を 16 kHz にしたもの（スマホで撮った口笛と、それを吹き返した笛）。

### 成績（学習に使っていないデータ、±25 cent 以内に聞き取れた割合と誤差の中央値）

| | ear_04（step5） | ear_01（step3・4） | YIN |
|---|---|---|---|
| 実機の笛の録音 | **99.4 %**、1.8 cent | 96.6 %、1.9 cent | 96.7 %、3.9 cent |
| 合成の口笛 | **99.6 %**、1.5 cent | 範囲外 | 94.4 %、2.8 cent |
| デモ動画の口笛（本物、スマホ） | 88.4 %、5.0 cent | 範囲外 | **91.3 %**、4.7 cent |
| 静かなところを「鳴っている」と誤る（デモ動画） | 1.3 % | | 0 % |

途中の版で分かったこと：

| 版 | 変えたこと | デモ動画の静かなところの誤り |
|---|---|---|
| ear_02 | 音域を広げ、合成の口笛を足した | 31 % |
| ear_03 | ＋音量の入力、雑音だけのフレーム | 12 % |
| ear_04 | ＋合成の口笛を小さい音まで作り、−58 dB 未満は「鳴っていない」 | **1.3 %** |

ear_02 は、口笛の合間のほとんど無音（−56〜−69 dB）で 3 kHz 台のありもしない音を聞き、口笛の音域が広がって移調がずれた（カエルの合唱で 525 cent）。

### 口笛コピーの楽譜（デモ動画の口笛から、YIN 版と耳の NN 版を比べる）

| 曲 | 全体のずれ | ずれを除いた差の中央値 | ±25 cent 以内 | 鳴らすかの一致 |
|---|---|---|---|---|
| カエルの合唱 | −64 cent | 8.3 cent | 84 % | 97 % |
| うぐいす | −7 cent | 5.6 cent | 90 % | 98 % |
| スーパーマリオ | +4 cent | 3.7 cent | 90 % | 98 % |

口笛コピーは音程の間隔を保って移すので、全体のずれは演奏に影響しない。カエルの合唱の全体のずれは、耳の NN 版が口笛の音域を少し広く聞いた（667 cent と 529 cent）ため。

### UNO Q での計算時間

耳 1 回あたり ear_01 が 5.12 ms、ear_04 と同じ大きさの ear_02 が 5.24 ms（NumPy）。演奏の 10 ms の周期の約半分。口笛 10 秒の解析に約 5 秒かかる。

## 2026-11-24 追記：実機で録った口笛で学び直した（ear_06）

画面の「録音だけ（データ集め）」で録った口笛 3.5 分・話し声など 5.5 分・無音 2 分と、口笛コピーで録った口笛 4 本を足して学び直した
（`--recorded runs/real/whistle_data runs/real/whistle`、実機の口笛は 4 回繰り返して使う）。口笛 2 本・話し声 1 本・無音 1 本は評価用に外した。

| 学習に使っていない実機の録音 | ear_06 | ear_04 | YIN |
|---|---|---|---|
| 口笛の ±25 cent 以内 | **93.5 %** | 91.1 % | 91.7 % |
| 話し声・無音を「鳴っている」と誤る | **1.2 %** | 12.8 % | 1.6 % |
| 笛（実機の録音） | 99.2 % | 99.4 % | 96.7 % |

step5・step6・step7・step7_practice・classic_ear の耳を ear_06 に替えた（2026-11-24。それより前の実機評価は ear_04）。

```bash
python scripts/yamabiko_ear_train.py --wide --level --synth 300000 --noise 60000 --recorded runs/real/whistle_data runs/real/whistle     --recorded-test 20261124_111404_whistle 20261124_111830_whistle 20261124_113803_other 20261124_112741_quiet     --compare models/step5/ear_04_runtime.npz --demo runs/ear/demo --out runs/ear/ear_06
```

## 残っている課題

- **本物の口笛のデータがない**。デモ動画（スマホ）では YIN にわずかに届かない。実機のマイクで口笛を 10〜20 分録り（口笛コピーの画面が音声を残すようになった）、`scripts/yamabiko_ear_data.py` で正解を付けて学び直す
- 耳が UNO Q で 5 ms かかる。演奏の周期に収まってはいるが、step6 以降で記憶や演奏 NN が大きくなると足りなくなる。畳み込みの計算の仕方（今は NumPy の einsum）を見直す余地がある
- 実機の評価（step4 と比べる）は許可を得てから

```bash
UNOQ=ssh bash scripts/uno_q_push.sh
python scripts/yamabiko_step_eval.py --steps step4 step5 --targets f80_doremi f80_scale f80_chromatic --repeats 2
```
