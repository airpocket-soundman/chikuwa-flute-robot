# step1 — NN が PWM を出し、口笛をまねる（2026-11-22）

git のタグ `step1` と、[models/step1/](../models/step1/) のファイルで再現できる。

## 中身

| 部分 | step1 での担い手 |
|---|---|
| 耳（マイクの音 → 音程） | YIN（`flute_rl/yamabiko/hw.py` の `heard_pitch`。250 Hz 未満を除いてから） |
| 手（アクチュエーターの PWM） | **NN**（`flute_rl/yamabiko/nn_performer.py` の `NNTwinPerformer`、`models/step1/nn_f80_02_runtime.npz`）。PD 制御も観測器もない。フィードバックも NN |
| 弁（SG90） | 規則（楽譜に音があれば開ける） |
| 演奏前の戻し | 規則（PWM −1.0 で 1.3 秒） |
| 調の合わせ込み | 起動後 1 回とボタンでキャリブレーション（縮み切りの音と、PWM 0.6 の 1 往復） |
| 口笛コピー | テストページの `/whistle`（`scripts/yamabiko_web.py`、`flute_rl/yamabiko/whistle.py`） |

## ハードの設定

| 項目 | 値 |
|---|---|
| ファン | **80 %**（裏返らずに 433〜約 1000 Hz。目標の音域は −10〜+1200 cent、1 オクターブ） |
| 弁（SG90） | 開 1560 µs、閉 1820 µs、動き 20 ms（`yamabiko_web.py` の `VALVE_US`） |
| アクチュエーター | 駆動・ブレーキの PWM、上限 1.0。|PWM| 0.5 以下では動き出さない |
| マイク | MIC1（B 基板、SAI1）。外れていると録音が全部 0 になる。付け直したら UNO Q ごと再起動 |
| 縮み切りの音 | 433.3 Hz（セッティングし直す前後で同じ） |

## 成績（実機、各 1〜2 回）

| 曲 | 隣り合う音の音程の誤差 | 調のずれを除いた平均誤差 | ±25 cent 以内 | 音程が取れた割合 | PWM の符号反転 |
|---|---|---|---|---|---|
| ドレミファミレド | 17〜27 cent | 32〜38 cent | 55〜63 % | 99 % | 7〜9 回 |
| 音階 1 オクターブ往復 | 21〜23 cent | 38〜43 cent | 42〜49 % | 99 % | 17〜19 回 |

比較: 骨格（PD＋観測器）＋NN は同じ曲で音程の誤差 21〜64 cent、符号反転 250〜590 回（細かく上下する）。
シミュレーター（ツインの周りの 48 台）: 平均誤差 64.2 cent、|ΔPWM| 0.024（骨格＋NN は 45.6 cent、0.48）。

## 再現の手順

### 1. UNO Q に入れる（PC の Git Bash、Wi-Fi 経由）

```bash
git checkout step1
UNOQ=ssh bash scripts/uno_q_push.sh
scp models/step1/nn_f80_02_runtime.npz models/step1/target_f80_*.npz unoq:/home/arduino/yamabiko/runs/real/
ssh unoq crontab -l      # 常駐サーバー（--host 0.0.0.0）とテストページが @reboot にあること
```

SSH の鍵が無い PC では、一度だけ USB の adb で `bash scripts/uno_q_setup_wifi.sh`。

### 2. 動かす

UNO Q の電源を入れると、常駐サーバー（15 秒後）とテストページ（45 秒後）が立ち上がり、テストページが一度キャリブレーションする（笛が約 10 秒鳴る）。
http://192.168.50.160:8080/whistle で「スタート」→ 3 秒のカウントの後に口笛 → 演奏。2 つのグラフと、録音・演奏の音を聞くボタン。

PC から同じことをするなら `python scripts/yamabiko_whistle_copy.py --seconds 5`。決まった曲は `Remote("192.168.50.160", forward=False).play(...)`（[handoff.md](handoff.md) の 0 章）。

### 3. 作り直す（較正 → ツイン → NN）

```bash
# 較正（UNO Q、ファン 80 %、全行程。約 17 秒鳴る）
python -c "import sys; sys.path.insert(0,'.'); from flute_rl.yamabiko.remote import Remote; r=Remote('192.168.50.160', forward=False); r.valve_positions(1560,1820,20); print(r.calibrate(out='/home/arduino/yamabiko/runs/real/calibration_y1f80_02.npz', plan='yamabiko1_full', max_cents=1700, fan=0.8))"
scp unoq:/home/arduino/yamabiko/runs/real/calibration_y1f80_02.npz runs/real/
# ツイン（PC、GPU で約 20 分）
python scripts/yamabiko_twin_fit.py runs/real/calibration_y1f80_02.npz --plant yamabiko1_fan80 --out runs/real/twin_y1f80_01
# NN（PC、GPU で約 1.5 時間 + 約 50 分）
python scripts/yamabiko_nn_train.py runs/real/twin_y1f80_01.json --teacher runs/yamabiko_residual_v1.pt --positions --steps 400 --out runs/real/nn_f80_01
python scripts/yamabiko_nn_train.py runs/real/twin_y1f80_01.json --teacher runs/yamabiko_residual_v1.pt --init runs/real/nn_f80_01.pt --soft-deadband 0.05 --steps 300 --out runs/real/nn_f80_02
```

学習は GPU の非決定性があるので、重みはビット単位では一致しない（成績は同じ程度になる）。step1 の重みそのものは `models/step1/nn_f80_02.pt`。

## models/step1/ のファイル

| ファイル | 内容 |
|---|---|
| `nn_f80_02.pt` / `.json` / `_runtime.npz` | step1 の NN（PyTorch、学習の記録、UNO Q 用 NumPy） |
| `nn_f80_01.json` | 1 段目の学習の記録 |
| `twin_y1f80_01.json` / `_runtime.npz` | ファン 80 % のツイン（再現誤差の中央値 10 cent） |
| `calibration_y1f80_02.npz` | ツインを当てはめた較正の記録 |
| `fan80_sweep_060.npz` | ファン 80 %、PWM 0.6 の 3 往復の録音と状態（`scripts/yamabiko_motion_analyze.py` で見られる） |
| `target_f80_*.npz` | 目標（ドレミファミレド、音階、大跳躍） |

## step1 の後に入ったもの（step1 には含めない）

- 弁と戻しも NN が出す版（`NNConfig.valve/current/self_home`、学習中）
