# やまびこ1号 実機で動かす・実機で学習する

シミュレーターで学習した GRU を実機で動かすための、ファームウェアと 3 本のプログラム。
実機ではエンコーダーがないので、実機の記録から**シミュレーターの機体の値を当てはめ**、その機体を中心に学習し直し、
それを実機で動かす、という順で回す。

```
UNO Q (Linux)                         PC
yamabiko_collect.py  ── 記録 .npz ──▶ yamabiko_fit.py   ── rig_fit.json
                                      yamabiko_train.py （YAMABIKO_RIG=rig_fit.json）── 方策 .npz
yamabiko_play.py     ◀── 方策 .npz, rig_fit.json ──
   └ 記録 .npz（演奏も学習データになる）──▶ yamabiko_fit.py
```

| 置き場所 | 動く場所 | 役目 |
|---|---|---|
| `mcu/yamabiko_fw/` | UNO Q の MCU | マイクの音声を Linux へ（`yamabiko_link` と同じ）＋ A 基板の IO。指令が 100 ms 来ないとアクチュエーターを止めて弁を閉じる |
| `flute_rl/yamabiko/hw.py` | UNO Q の Linux | 実機を、シミュレーターの `Rig` と同じ `step(pwm, valve)` で動かす包み。10 ms ごとに指令を送り、直近 20 ms の音から音程を出す |
| `scripts/yamabiko_collect.py` | UNO Q の Linux | **学習データ収集**。学習と同じ曲の生成器で曲を吹かせ、制御器（既定は手書きの観測器）の指令にゆっくり変わる雑音を足して、動き方を広げる |
| `scripts/yamabiko_fit.py` | PC | **学習（その 1）**。記録の指令をシミュレーターで再生し、聞こえた音程が最も合う機体の値を探す。合わなかった分を聞き取りの雑音にする |
| `scripts/yamabiko_train.py` | PC（GPU） | **学習（その 2）**。既存の学習をそのまま、当てはめた機体を中心に回す |
| `scripts/yamabiko_play.py` | UNO Q の Linux | **推論実行**。EXEC ボタンで口笛を録り、GRU で吹く。記憶は曲をまたいで持ち越し、起動時だけ消す |

## 準備（UNO Q、1 回だけ）

```
sudo apt install python3-numpy                                  # UNO Q の Linux で
sudo systemctl disable --now arduino-router arduino-router-serial   # 済みなら不要
```

PC から（Git Bash）:

```
scripts/uno_q_push.sh                     # flute_rl/、スクリプト、ファームウェア、runs/yamabiko_gru.npz を /home/arduino/yamabiko へ
```

ファームウェアの書き込み（UNO Q の `adb shell` で。再起動するとスケッチが消えるので、そのたびに書き直す）:

```
cd /home/arduino/yamabiko
TMPDIR=/tmp HOME=/home/arduino arduino-cli compile -b arduino:zephyr:unoq --output-dir build_fw yamabiko_fw
TMPDIR=/tmp HOME=/home/arduino ~/.arduino15/packages/arduino/tools/remoteocd/0.0.4-rc.4/remoteocd upload \
  -f ~/.arduino15/packages/arduino/hardware/zephyr/0.53.1/variants/arduino_uno_q_stm32u585xx/flash_sketch.cfg \
  /home/arduino/yamabiko/build_fw/yamabiko_fw.ino.elf-zsk.bin
python3 scripts/yamabiko_link.py info     # INFO fw yamabiko_fw ... と IO ... が出れば書けている
```

`TMPDIR=/tmp` を付けないと、remoteocd は adb の既定の `/data/local/tmp` を探して止まる。

## 実機を組んだら最初に合わせること

| 項目 | 合わせ方 |
|---|---|
| アクチュエーターの向き | pwm > 0 で**管が短くなる（音が上がる）**向き。逆なら全部のプログラムに `--flip` |
| 弁のサーボ位置 | `--valve-open` / `--valve-closed`（SCS0009 の 0〜1023、既定 590 / 512 は仮の値）、`--valve-ms`（動く時間） |
| ファンの強さ | `--fan`（0〜1、既定 0.7）。笛が裏返らず鳴る強さ |
| アクチュエーターの上限 | `--max-pwm`（ファームウェアでも切る） |
| 室温 | `yamabiko_fit.py --temp`。音程からは管長と温度を分けられないので、温度は測った値で固定する |
| 管の中の可動範囲 | `yamabiko_fit.py --stroke`（既定 0.105 m） |

## 回し方

```
# UNO Q: 学習データ（20 曲で 2 分ほど）
python3 scripts/yamabiko_collect.py --songs 20 --out runs/real/collect_01.npz
# PC: 取ってくる
adb pull /home/arduino/yamabiko/runs/real/collect_01.npz runs/real/
# PC: 機体を当てはめる（数分）
python scripts/yamabiko_fit.py runs/real/collect_01.npz --temp 24 --out runs/real/rig_fit.json
# PC: 当てはめた機体を中心に学習し直す
YAMABIKO_RIG=runs/real/rig_fit.json python scripts/yamabiko_train.py --device cuda --gens 300 --pop 256 --rigs 64 --songs 5 --out runs/yamabiko_gru_real.npz
# UNO Q へ送って演奏
scripts/uno_q_push.sh runs/yamabiko_gru_real.npz runs/real/rig_fit.json
python3 scripts/yamabiko_play.py --rig runs/real/rig_fit.json --policy runs/yamabiko_gru_real.npz
```

- `yamabiko_play.py --demo 5` はボタンも口笛も使わず、ランダムな曲を 5 曲吹く（実機の試運転）。`--key` はボタンの代わりに Enter
- 方策は、学習したときと**同じ `--rig` で動かす**。制御器の中の推測航法が、名目の機体の値を使っているため
- 記録（`runs/real/*.npz`）には、各ステップの指令・聞こえた音程・YIN の確からしさ・音量・電流・ファンの回転数・サーボ位置と、全区間の音声（16 kHz）が入る。
  `yamabiko_fit.py` は、収集の記録と演奏の記録のどちらも読める

## 当てはめで決まるもの・決まらないもの

`yamabiko_fit.py --selftest` は、シミュレーターのランダムな機体で同じ形の記録を作り、それを当てはめて真の値と比べる。
10 曲（約 40 秒）での結果:

| よく決まる | ずれが残る |
|---|---|
| 速さ（`v_in`, `v_out`、1% 以内）、不感帯（0.003 以内）、管長（1〜2 mm） | `press_cents`（管長とほぼ同じ効き方をする。合わせて音程の絶対値は合う） |
| 聞き取りの欠け（`dropout`）、オクターブの取り違え | 開いた直後の過渡（`onset_cents`, `surge_cents`）、ガタ（`backlash`） |
| 遅れの**和**（指令→聞こえるまで、弁→鳴るまで） | 遅れの**内訳**が 1 ステップ入れ替わることがある |

- 指令の遅れと聞こえる遅れは、音だけでは和しか分からない。実機の記録ではモーター電流から指令の遅れを別に出して固定する
  （PWM の大きさと電流の相関が 0.3 を超えたとき。`--fix cmd_delay=1` で手で与えてもよい）
- 聞こえる遅れと弁の遅れも、鳴り始めの時刻では入れ替えがきく（弁 1 ステップ ＋ 聞こえ 3 ステップ ≒ 2 ＋ 2）。
  学習は既定のばらつき（遅れ ±2 ステップ）で回すので、1 ステップのずれは覆える
- 当てはめで合わなかった分（`pitch_noise`）は、実機の音程のばらつきより大きめに出る。学習では雑音が多めになる側に倒れる

## 試験の状況（2026-09-19）

- ファームウェアは書き込んで動作を確認: アクチュエーター 20 kHz（TIM4、ARR 8000）、ファン 25 kHz（TIM1、ARR 6400）、
  状態パケット 10 ms ごと。音声の遅れは状態パケットを足しても**中央値 2.76 ms**（p99 3.35、10 秒で遅れ 5 ms 超 0）
- A 基板がまだないので、アクチュエーター・ファン・サーボ・ボタン・電流・タコは未確認（サーボは無応答と出る）
- `hw.py` の制御ループ・記録・保存・`yamabiko_fit.py` での読み込みは、PC 上の偽の MCU で確認（1 ステップ 9.6 ms）
- UNO Q 上での実行は numpy を入れてから
