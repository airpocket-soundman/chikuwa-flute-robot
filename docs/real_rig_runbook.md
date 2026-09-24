# 実機の手順書: 較正 → デジタルツイン → 演奏

更新日: 2026-09-24。部品が届いたら、この順に進める。前提の配線・ファームウェア・UNO Q の準備は [yamabiko_real.md](yamabiko_real.md) のとおり。

## 0. 較正の前に確かめること（新しい機構では必ず）

| 項目 | 確かめ方 | 対応 |
|---|---|---|
| アクチュエーターの向き | `pwm > 0` で管が短くなる（音が上がる）こと | 逆なら全部のプログラムに `--flip` |
| 端のストッパーへの衝突 | 全力（`--max-pwm 1.0`）で端に当てて壊れないか。心配なら `--max-pwm 0.6` から | 較正は `--max-pwm` で上限を下げられる |
| 倍音（オーバーブロー） | ゆっくり管を短くしていき、音が裏返る位置を聴く | 裏返るなら `--sweep 0.5` で掃引を短くする |
| 送風 | 較正の約9秒間、ずっと鳴っていること（`--fan` の強さ） | 鳴らない・裏返るなら `--fan` を調整 |
| マイクの音程 | `scripts/yamabiko_mic_check.py`（ばらつき 0.1〜0.6 cent が既知の値） | |

## 1. 較正（UNO Q、約 12 秒）

```
python3 scripts/yamabiko_twin_calibrate.py --out runs/real/calibration_01.npz [--flip] [--max-pwm 0.6 --sweep 0.5]
```

原点合わせ（1.3 秒）の後、バルブを開けたまま決まった PWM 列を流す：不感帯の階段 → 全力の往復（各 1 秒）→ 中速の往復 → ゆっくりした掃引。記録するのは指令と聞こえた音程だけ。`late steps` が多い（数十以上）なら、UNO Q の負荷を減らす。

## 2. ツインの当てはめと実行ファイルの書き出し（PC、数分）

```
adb pull /home/arduino/yamabiko/runs/real/calibration_01.npz runs/real/
python scripts/yamabiko_twin_fit.py runs/real/calibration_01.npz --out runs/real/twin_01 [--temp 24]
```

- `runs/real/twin_01.json`：ツインの値（速さ、トルク、摩擦、不感帯、管長、遅れ）と、較正の記録の再現誤差（シミュレーターでは中央値 3 cent。実機で 10 cent を大きく超えるなら、較正のやり直しか、物理式が実機を表せていない）。
- `runs/real/twin_01_runtime.npz`：UNO Q 用の numpy 版の演奏プログラム（骨格＋NN補正、ツイン入り）。

## 3. 演奏（UNO Q）

```
python scripts/yamabiko_twin_target.py --phrase large-leaps --out runs/real/target_leaps.npz   # PC で目標を作る
adb push runs/real/twin_01_runtime.npz runs/real/target_leaps.npz /home/arduino/yamabiko/runs/real/
python3 scripts/yamabiko_twin_play.py --runtime runs/real/twin_01_runtime.npz --target runs/real/target_leaps.npz --repeats 3 --out runs/real/play_01.npz
```

同じ曲を 3 回繰り返し、曲の記憶を持ち越す。各回の「聞こえた音程と目標の差」を表示する（実機で測れる唯一の成績）。シミュレーターでは 1 回目 92 %、3 回目 93 % の命中率（±25 cent、0.3 秒以内）。

## 4. 実機の演奏でツインを直す（任意）

```
adb pull /home/arduino/yamabiko/runs/real/play_01.npz runs/real/
python scripts/yamabiko_twin_fit.py runs/real/calibration_01.npz --songs runs/real/play_01.npz --out runs/real/twin_02
```

演奏の記録も原点合わせから始まるので、較正と同じ方法で当てはめられる。直したツインで実行ファイルを書き出し直し、3 に戻る。

## 5. 実機で最初に測って、シミュレーターに戻す値

較正が通ったら、ツインの値を `PhysicalPlantConfig` の標準値と乱数化の幅に反映し、ベンチマーク（`scripts/evaluate_yamabiko_fitting.py`）の機体の幅を実測の周りに狭める。今の幅（モーター 0.4〜2.5 倍）は推測である。

## 既知の制限

- 骨格の観測器は「音程 = 位置」の関係が単調であることを前提にする。倍音で音が裏返る位置は使わない（掃引を短くする）。
- 聞こえる遅れは 0〜3 ステップの整数として当てはめる。
- UNO Q での 1 ステップの時間は未計測（PC で 0.34 ms。旧 GRU 版の UNO Q 実測 9.6 ms から、10 ms 以内の見込み）。
