# step4 — ハード変更後の機体に合わせ直す（2026-11-23、シミュレーターで評価、実機の評価はまだ）

step3 と同じ組み合わせ（アクチュエータ・タンギング・耳がすべて NN）のまま、ハード設定を変えた後の機体でツインを当てはめ直し、
アクチュエータと弁の NN を学び直した版。[models/step4/](../models/step4/) にあり、step1〜3 と並べて置いてある。

## なぜ学び直したか

ハード設定を変えた後に、同じ指令でキャリブレーションを取り直すと（`calibration_y1f80_03`、`calibration_y1f80_mid_02`）、
前の記録より音程が約 30 cent 低く、全行程の最高音も 1211 → 1089 cent に下がっていた。
step2・3 が使っていたツイン（`twin_y1f80_03`）で今日の実機の演奏を再現すると、誤差は 30〜129 cent、全体に 40〜120 cent 低くずれていた。

## 中身

| 部分 | step4 での担い手 |
|---|---|
| 耳 | NN（`ear_01`、step3 と同じ） |
| DC リニアアクチュエータの PWM | **NN**（`nn_f80_t4_s06`、step3 の重みから新しいツインで 600 回、滑らかさの重み 0.6） |
| 切換え弁（タンギング） | **NN**（`valve_t4_01`、新しいアクチュエータの NN に合わせて学び直し） |
| 演奏前の戻し・キャリブレーション | 規則 |

## 作り方

1. **ツイン**（`twin_y1f80_04`）：新しいキャリブレーション 2 本と、同じ日の step1〜3 の実機評価 18 回分を 1 つの機体として当てはめた。
   今日の演奏の再現誤差は 22〜77 cent（前のツインは 30〜129 cent）、全体のずれは ±35 cent 程度（記録ごとの揺れ）。
   `scripts/yamabiko_twin_check.py` で記録ごとに確かめられる。
2. **アクチュエータの NN**：滑らかさの重み 0.3 と 0.6 を学んで比べ、音程が同じで反転の少ない 0.6 を選んだ。
3. **弁の NN**：アクチュエータの NN を固定して、`--target travel` で学んだ。

```bash
python scripts/yamabiko_twin_fit.py runs/real/calibration_y1f80_03.npz --songs runs/real/calibration_y1f80_mid_02.npz runs/real/eval/20261123_174554_step*_f80_*.npz --joint --plant yamabiko1_fan80 --out runs/real/twin_y1f80_04
python scripts/yamabiko_nn_train.py runs/real/twin_y1f80_04.json --teacher runs/yamabiko_residual_v1.pt --init runs/real/nn_f80_t3_01.pt --soft-deadband 0.05 --steps 600 --smooth 0.6 --out runs/real/nn_f80_t4_s06
python scripts/yamabiko_valve_train.py runs/real/twin_y1f80_04.json --actuator runs/real/nn_f80_t4_s06.pt --target travel --out runs/real/valve_t4_01
python scripts/yamabiko_nn_sim_eval.py runs/real/twin_y1f80_04.json --nets models/step3/nn_f80_t3_01.pt models/step4/nn_f80_t4_s06.pt
```

## 成績（シミュレーター、新しいツインの周りの 48 台）

アクチュエータ（`yamabiko_nn_sim_eval.py`、音の頭 0.3 秒を除いて ±25 cent 以内、反転は PWM の向きが変わった回数）:

| 曲 | step1 | step3 | step4（重み 0.6） | 参考：重み 0.3 |
|---|---|---|---|---|
| ドレミファミレド | 23 %、49 cent | 68 %、25 cent、反転 18 | **80 %、19 cent、反転 21** | 80 %、18 cent、反転 26 |
| 音階 | 27 %、51 cent | 54 %、30 cent、反転 28 | **66 %、25 cent、反転 29** | 66 %、24 cent、反転 35 |
| 半音階 | 49 %、45 cent | 75 %、24 cent、反転 18 | 72 %、**19 cent**、反転 18 | 74 %、18 cent、反転 24 |

弁（48 台、ランダムな曲）:

| | ±25 cent 以内 | 滑る音 | 楽譜の音が鳴った割合 |
|---|---|---|---|
| 楽譜どおり（規則） | 71.4 % | 19.7 % | 95.7 % |
| 弁の NN（step4） | **72.6 %** | **18.0 %** | 94.2 % |

## 実機での評価（許可を得てから）

```bash
UNOQ=ssh bash scripts/uno_q_push.sh
python scripts/yamabiko_step_eval.py --steps step3 step4 --targets f80_doremi f80_scale f80_chromatic --repeats 2
```
