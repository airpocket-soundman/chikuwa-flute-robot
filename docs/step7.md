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
