# step7 — ツインに頼らない演奏（2026-11-24、シミュレーターと実機で評価）

step6 と同じ耳（`ear_04`）・お手本の記憶のまま、演奏中に **ツインの物理式を使わない** 版。
アクチュエータの NN も弁の NN も、ツインのモーターの式による推定位置・速度と、音程から逆算した位置を入力に使わない。
音が聞こえない間の位置は、GRU が自分の指令の履歴から推し量る（カーナビの「GPS が届かない間はジャイロ」を NN が自分で行う）。
[models/step7/](../models/step7/)。計画は [nn-e2e-plan.md](nn-e2e-plan.md) の step7。

## 中身

| 部分 | step5・6 | step7 |
|---|---|---|
| アクチュエータの NN の入力 | 楽譜の先読み・聞こえた音程・目標との差・前回の PWM・**ツインの推定位置と速度・目標と聞こえた音程の位置** | 楽譜の先読み・聞こえた音程・目標との差・前回の PWM（29 → 25 個） |
| 弁の NN の入力 | 同上の推定位置と速度を含む 27 個 | 25 個 |
| 学習 | ツイン（シミュレーター）の中 | 同じ（外したのは演奏中の入力だけ） |

`NNConfig(twin_free=True)`・`ValveConfig(twin_free=True)`。step4 の重みから、入力を名前で対応づけて引き継いだ（`NNTwinPerformer.from_checkpoint` が並びの違う重みを名前で移す）。

```bash
python scripts/yamabiko_nn_train.py runs/real/twin_y1f80_04.json --teacher runs/yamabiko_residual_v1.pt --init models/step4/nn_f80_t4_s06.pt --twin-free --soft-deadband 0.05 --steps 1500 --smooth 0.6 --out runs/real/nn_f80_t7_init
python scripts/yamabiko_valve_train.py runs/real/twin_y1f80_04.json --actuator runs/real/nn_f80_t7_init.pt --target travel --twin-free --out runs/real/valve_t7_free
```

最初から学んだ版（`nn_f80_t7_scratch`、教師のまね → 1500 回）は PWM の反転が少ないが、音階で 54 % と落ちたので使わなかった。

## 成績

| 曲（±25 cent 以内） | シミュレーター step4 | シミュレーター step7 | 実機 step5 | 実機 step7 |
|---|---|---|---|---|
| ドレミファミレド | 80 % | **85 %** | **75 %** | 50 % |
| 音階 | 63 % | **73 %** | 50 % | **60 %** |
| 半音階 | 71 % | **77 %** | 50 % | **86 %**（全版で最高） |

シミュレーターは `scripts/yamabiko_compare_sim.py`（32 台）、実機は `scripts/yamabiko_step_eval.py`（各 2 回、2026-11-24）。
シミュレーターでは step4 より良くなった（学習回数が 600 → 1500 回と多い分も含む）。実機では曲による差が大きい。
古典制御を含めた比較は [実機比較 11-24](https://airpocket-soundman.github.io/chikuwa-flute-robot/rig_1124/)（`reports/rig_1124/`）。

## 比較用：classic_ear

[models/classic_ear/](../models/classic_ear/) は classic（PID・観測器・規則の弁）の耳だけを `ear_04` にした版。
実機でドレミ 84 %・音階 74 %・半音階 75 % と 3 曲とも安定した（classic は YIN が高音を取りこぼし、音階が崩れた）。PWM の反転は 1 曲 50〜104 回と多い。

## 2026-11-25 追記：step7b（タンギングを楽譜どおりに）

実機で速い口笛を吹かせると、step3〜5 の弁は楽譜で鳴らす時間の約半分しか開かず、ほとんど音が出なかった
（シミュレーターでは 87〜89 % 開く。聞こえた音程を入力にしていて、目標から離れる・聞こえないと閉じ、閉じるとさらに聞こえない、の悪循環と見ている）。
弁の NN を、楽譜の「鳴らす／休む」をそのまま目標にし、合成の口笛から規則で作った速い楽譜も半分混ぜて学び直した（`valve_t7_score`）。

| 速い口笛の楽譜（シミュレーター、48 台） | 楽譜で鳴らす時間に開いている | 休符の間に閉じている | 開け閉めの回数（1 曲） |
|---|---|---|---|
| 楽譜（理想） | 100 % | 100 % | 17.6 |
| step7 の弁（`valve_t7_free`） | 94 % | 90 % | 10.4 |
| **step7b の弁（`valve_t7_score`）** | **100 %** | **99 %** | **17.5** |

聞こえにくい条件（動いている間に 6 割聞こえない、機体のばらつき ±25 %）でも 100 % 開いた。step7b はこの弁だけを替えた版（耳 ear_06・記憶 mem_04・演奏の NN は step7 と同じ）。

```bash
python scripts/yamabiko_valve_train.py runs/real/twin_y1f80_04.json --actuator models/step7/nn_f80_t7_init.pt --twin-free     --target score --whistle-songs 0.5 --out runs/real/valve_t7_score
```

## 2026-11-25 追記：機体の記憶（step7 の残りの課題）の結果

演奏の NN に、同じ機体の上で次の曲へ持ち越す小さな記憶（`NNConfig.rig_memory`）を足した（PyTorch と NumPy で一致、`reset_rig()` で消せる）。
シミュレーターでの結果は、**効果なし**：

| 試したこと | 結果（±25 % ばらつかせた機体、5 曲） |
|---|---|
| 記憶ごと学習（3 曲続けて、2 回） | 次の曲へ持ち越しても、毎曲消しても差は 0.4 cent 以内。記憶にほとんど書かれず、読まれもしない |
| ツインそのものとの差 | ツイン 54 cent、ばらつかせた機体 69 cent（14 cent の差） |
| 1 段目：本当の機体の特性を入力にして学習（RMA の考え方） | 本当の値 61.7、ゼロ 64.0、別の機体の値 66.8 cent。**知っていても約 2 cent しか良くならない** |
| 2 段目：聞きながら特性を推し量る記憶だけを学習 | 推し量った値と本当の値の相関はほぼ 0。演奏も変わらない |

このシミュレーターでは、機体のくせを知っても取り戻せる分が小さく（遅いモーターは知っていても間に合わない）、
今の書き込み方（その時の音と指令だけを見る）では、摩擦や力の違いを読み取れなかった。
機体の記憶が効くのは、シミュレーターが再現していない実機のくせのはずなので、実機の演奏の記録で学ぶ方法を考える。

```bash
python scripts/yamabiko_nn_train.py runs/real/twin_y1f80_04.json --teacher runs/yamabiko_residual_v1.pt --init models/step7/nn_f80_t7_init.pt     --rig-memory 7 --rig-oracle --spread .25 --soft-deadband 0.05 --steps 600 --lr 3e-3 --smooth 0.6 --out runs/real/nn_f80_t7_rig_oracle
python scripts/yamabiko_nn_train.py runs/real/twin_y1f80_04.json --teacher runs/yamabiko_residual_v1.pt --init runs/real/nn_f80_t7_rig_oracle.pt     --rig-memory 7 --rig-rate .1 --rig-fit --songs-per-rig 3 --spread .25 --soft-deadband 0.05 --steps 300 --lr 3e-3 --new-lr 1 --smooth 0.6 --out runs/real/nn_f80_t7_rig_fit
```

