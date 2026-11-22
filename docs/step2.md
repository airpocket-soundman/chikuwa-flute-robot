# step2 — タンギングも NN（2026-11-23、シミュレーターで評価、実機の評価はまだ）

step1 に、切換え弁（タンギング）を動かす NN を足し、アクチュエータの NN を実機に合わせ直したツインで学び直した版。
[models/step2/](../models/step2/) にあり、step1・step3 と並べて置いてある（どれでも動かせる。[step1.md](step1.md) の冒頭）。

## 中身

| 部分 | step2 での担い手 |
|---|---|
| 耳 | YIN（step1 と同じ） |
| DC リニアアクチュエータの PWM | **NN**（`nn_f80_t3_01`、step1 の重みから、新しいツインで 600 回学び直した） |
| 切換え弁（タンギング） | **NN**（`valve_t3_01`、`flute_rl/yamabiko/valve_nn.py`）。アクチュエータの NN と並べて動かす（`residual_numpy.WithValve`） |
| 演奏前の戻し・キャリブレーション | 規則（step1 と同じ） |
| 口笛の解析 | 50 ms 以下の途切れはつながず、30 ms 以上の区切りを残す（タンギングを写す） |

## 作り方

1. **ツインを実機の演奏で当てはめ直す**（`twin_y1f80_03`）：全行程の較正、中くらいの PWM（0.55〜0.8）の較正、step1 の演奏 12 回分を 1 本につなげて、1 つの機体として当てはめた（`yamabiko_twin_fit.py --joint`）。
   step1 までのツインは NN の演奏を 64〜214 cent 高く予測していた（較正の矩形の指令しか見ていなかった）が、新しいツインは 9〜28 cent、当てはめに使っていない演奏でも 16〜17 cent で再現する。
2. **アクチュエータの NN を学び直す**：step1 の続き（`nn_f80_03`）から 600 回（`yamabiko_nn_train.py --soft-deadband 0.05`）。
3. **弁の NN を学ぶ**：アクチュエータの NN を固定して、楽譜とツインだけから決まる目標（休符と連打の区切りで閉じる、跳躍の移動時間だけ閉じる、ほかは開ける）を、サーボの遅れ（3 ステップ）を見越して学ぶ（`yamabiko_valve_train.py --target travel`）。
   「音程が合ってから開ける」を目標にした版は実機で失敗した（閉じている間は音が聞こえず、開ける手掛かりが無い）。

```bash
python scripts/yamabiko_twin_fit.py runs/real/calibration_y1f80_02.npz --songs <較正 mid と step1 の演奏> --joint --plant yamabiko1_fan80 --out runs/real/twin_y1f80_03
python scripts/yamabiko_nn_train.py runs/real/twin_y1f80_03.json --teacher runs/yamabiko_residual_v1.pt --init runs/real/nn_f80_03.pt --soft-deadband 0.05 --steps 600 --out runs/real/nn_f80_t3_01
python scripts/yamabiko_valve_train.py runs/real/twin_y1f80_03.json --actuator runs/real/nn_f80_t3_01.pt --target travel --out runs/real/valve_t3_01
```

## 成績（シミュレーター、新しいツインの周りの機体）

アクチュエータ（音の頭 0.3 秒を除いて ±25 cent 以内）:

| 曲 | step1 | step2 |
|---|---|---|
| ドレミファミレド | 50 % | **76 %** |
| 音階（1 オクターブ往復） | 39 % | **59 %** |
| 半音階 | 51 % | **61 %** |

新しいツインで評価した step1 の値は、実機の step1（63 %、45 %）に近い。

弁（48 台、ランダムな曲）:

| | ±25 cent 以内 | 滑る音 | 楽譜の音が鳴った割合 |
|---|---|---|---|
| 楽譜どおり（規則） | 71 % | 20 % | 96 % |
| 弁の NN | 69 % | 18 % | 92 % |

弁の NN はシミュレーターでは規則とほぼ互角。実機でタンギングの切れ方を聞き比べる。

## 実機での評価（許可を得てから）

```bash
python scripts/yamabiko_step_eval.py --steps step1 step2 step3 --targets f80_doremi f80_scale f80_chromatic --repeats 2
```
