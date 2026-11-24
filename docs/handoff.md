# 引き継ぎ — やまびこ1号（2026-11-22）

> **2026-11-24 追記:** 制御側の現状（step4 の 3 つの NN と入出力）と、全部 NN にして E2E のフィジカル AI にするまでの段取り（step5〜8）は
> [nn-e2e-plan.md](nn-e2e-plan.md) にまとめた。比較用の古典制御（PID、NN なし）は [classic.md](classic.md)。
> 実機で再開するときの手順は nn-e2e-plan.md の 6 章。以下は機体・基板・UNO Q の記録。

> **2026-11-24 追記（画面とボタン）:** UNO Q の `http://<UNO Q>:8080/` が演奏の画面になった（`scripts/yamabiko_web.py`。手動テストは `/bench`）。
> classic・step1〜（`models/*/profile.json` があるもの全部）を画面で選び、選んだ版は `runs/web_step.txt` に残る。
> - 電源を入れると crontab の `@reboot` でサーバー（15 秒後）と画面（45 秒後）が立ち上がり、**画面が起動後に一度キャリブレーションする（笛が約 10 秒鳴る）**。
> - 基板の EXEC ボタン：**1 回押す**＝録音開始（新しい曲、練習の記録は消える）→ **1 回押す**＝録音を終えて聞き取り・演奏（練習 1 回目）→
>   **すばやく 2 回押す**（0.5 秒以内）＝録音せずに同じ曲をもう一度（練習 2 回目、3 回目…）。キャリブレーション・演奏中に押すと止まる。
> - 練習：サーバーが同じ曲の演奏器を次の演奏まで持ち続ける（`cmd_play` の `practice`）。曲の記憶（`song_slot`）を持つ版なら、前の演奏で書いた記憶を読んで上達する。持たない版は毎回同じに吹く。
> - 画面の「録音」「練習：もう一度」「止める」はボタンと同じ働き。基板のボタンが押されると、画面の同じボタンが光る（状態の `press`）。
> - 手で起動し直す：UNO Q の `bash /home/arduino/yamabiko/restart.sh`（`scripts/uno_q_restart.sh`）。`--no-calibrate-at-start --no-button` で音を出さない起動。
> - PC で画面だけ試す：`python scripts/yamabiko_web_dev.py --port 8081`（偽の MCU、音は出ない。`POST /dev/press`、`{"n": 2}` で 2 回押し）。

A・B 基板を UNO Q に載せて初めて通電し、PC から命令で動かせるところまで来た。その時点の引き継ぎ。
前の版（2026-09-18、基板設計の前段）は git の履歴にある（`git log -- docs/handoff.md`）。

ブランチは `main` だけ（作業ブランチはすべて取り込み済み）。最新のコミットは `2518521`。

## 0. 2026-11-22 の進展（最新。以下の章は 11-20 時点）

**機構の性質（測定は `scripts/yamabiko_motion_test.py`、解析は `scripts/yamabiko_motion_analyze.py`。記録は runs/real/motion/、git 外）**

| 項目 | 結果 |
|---|---|
| ファン 50 % | 縮み切り 421 Hz（日により 400 Hz で弱い）。約 600 Hz（行程の半分、全力で 0.7 秒分）を越えると裏返る。使えるのは約 6 半音 |
| **ファン 80 %** | **裏返らない**。430 Hz から伸び切りの約 1030 Hz まで基音のまま。止まった状態できれいに鳴るのは約 880 Hz まで → **1 オクターブ使える**。音が大きくモーター音に負けにくい |
| ファンと音程 | 強くすると上がる。縮み切りで 50→100 % が +72 cent、伸ばした側ほど大きい（0.7 秒の位置で +286 cent） |
| アクチュエーター | 静止摩擦: |PWM| 0.5 以下では動き出さない。動けば速さは |PWM| − 0.2 にほぼ比例（全行程 1.0: 1.39 s、0.8: 1.84 s、0.6: 2.8 s、ばらつき ±0.03 s） |
| 端の判定 | 内蔵のリミットで端に着くと電流が 0 になる |
| 繰り返し | 縮み切りから同じ時間だけ全力で伸ばしたときの音程のばらつき 2〜7 cent。パルスを重ねると 20〜30 cent までずれる |
| モーター音 | 全力で動かすとファン 50 % では約 6 割のステップで音程が取れない |

**直した不具合**: 命令に枠（A5・命令・和）を付けた（1 バイト欠けると PWM 0.6 の 0x58 が 'X' と読まれ音が止まっていた。ファームウェア書き換え済み）／サーバーが音の停止を 0.5 秒で自動復帰／録音の返信を Wi-Fi でも送り切る／較正・演奏の後にファンが止まっていなかった／音程推定の前に 250 Hz 未満を除く（ファン 80 % の唸りで YIN が失敗していた）／音程推定の下限 300 Hz

**UNO Q は Wi-Fi で扱う**: `ssh unoq`（鍵は `scripts/uno_q_setup_wifi.sh` で登録済み）、`UNOQ=ssh bash scripts/uno_q_push.sh`。常駐サーバーは `--host 0.0.0.0`（認証なし、家庭内 LAN のみ）、テストページ（8080）とともに起動時に自動で立ち上がる。PC の `Remote("192.168.50.160", forward=False)`

**制御**:
- 設定 `PhysicalPlantConfig.yamabiko1`（ファン 50 %）と `yamabiko1_fan80`。較正の手順 `plan="yamabiko1"`／`"yamabiko1_full"`。静止摩擦（`kinetic_deadband_ratio`）を入れてツインの再現誤差は 27→10 cent
- 目標は**相対的なメロディーコピー**（音程の間隔を保って音域に移調。演奏前に縮み切りの音を測ってその日の笛に合わせる、`relative=True`）
- 骨格＋NN（従来）は実機で動くが、不感帯を飛び越すための ±0.55 で 1 曲に 200〜600 回 PWM の符号が反転する（細かく上下する）
- **NN 主体の演奏器**（`nn_performer.py`、PD も観測器もなく NN が PWM を直接出す。骨格をまねてからシミュレーター越しに音程誤差＋|ΔPWM| で学習、`scripts/yamabiko_nn_train.py`）: 動きは滑らか（反転 3〜13 回）になったが、ファン 50 % の版は音程が粗い。ファン 80 % で学習し直し中

## 1. いまの状態（ひとことで）

- シミュレーターの上では計画どおりに動いている。固定ベンチマークで、命中率は 1 回目 92 %、3 回目 93 %（[README](../README.md)、[fitting-twin-plan.md](fitting-twin-plan.md)）
- 実機は、基板と電子回路の確認がほぼ終わった。笛・モーター・ファンをつないでの動作はまだ
- UNO Q は、電源を入れて約 1 分待てば、PC から USB 越しに命令で動かせる（4 章）

## 2. 実機の確認結果（2026-11-20）

詳細は [yamabiko_real.md](yamabiko_real.md) の「A・B 基板を載せた最初の確認」と「PC から命令で動かす」。

| 回路 | 結果 | 備考 |
|---|---|---|
| A 基板だけに 12V | 正常 | ヘッダー 32 本に漏れなし、6V 系 6.07 V |
| アクチュエーター出力（TB6643、J2） | 正常 | +で 1 番、−で 2 番が 12V。無負荷では惰性モード（オフ時は出力が切り離される）のため平均ではなく 12V と表示される |
| ファン電源（Q1・Q2、J3-2） | 正常 | オンで 12V。普段はオフ（R5・R7 で切れる側） |
| 電磁弁出力（Q4、J5、本来は DNP） | 正常 | 開で J5 の 1–2 間 12V |
| EXEC ボタン（D8） | 正常 | |
| サーボ線（U3 74HC125 → J4） | 正常 | 自分の送信が D0 に折り返す。待機 3.29 V |
| **SCS0009** | **不良** | 全速度・全 ID で無応答、常時駆動・一斉送信・M5Stack CoreS3 からの直結でも動かず。配線は仕様書どおり（1 GND、2 Vcc、3 信号） |
| MIC1（B 基板、SAI1）→ Linux | 正常 | 途切れ 0。静かな部屋 −57〜−71 dBFS、指で叩くと −6〜0 dBFS |
| 口笛の音程 | 未確認 | 小さな音しか出せず、2.1〜2.9 kHz の山は見えたが RMS が足切り（−60 dBFS）未満 |
| ファンの PWM・タコ | 未確認 | ウェブページ（`scripts/yamabiko_web.py`、http://<UNO Q>:8080/）で試せる |
| **SG90**（D1 = PB6、TIM16_CH1N、50 Hz） | 正常 | 2026-11-20、`2d91b66` のファームウェアで 1000〜2000 µs の往復を確認 |
| **アクチュエーター**（実物をつないで） | 動く | 2026-11-20。出力を切り離す PWM では 0.6 でも力が足りなかった（平均電圧が約 (2×duty−1)×12V になる）ので、オフの間をブレーキにする方式に変更。±0.3 でもまだ弱く、±1.0（12V 連続）で動いた。電流の読みは約 120 mA。サーバーの上限 0.6 は見直しが要る |
| ファームウェアの長時間運転 | 不良→修正 | 2026-11-21。起動から約 12.4 時間で MCU が応答しなくなった。音の語数（u32、96 k 語/秒）が一周したとき `pump()` の `done + 2 <= total` が約 2^31 フレームの間真になり、命令も状態も止まった。差で比べるよう修正。12.4 時間を越える運転はまだ確かめていない |
| モーターの向き（`--flip` の要否） | 未確認 | `pwm > 0` で管が短くなる（音が上がる）こと |

B 基板は MIC1 の分だけ実装（J14/J15、J1、R1–R3、R7、C1）。MIC2・予備の I2C/ADC は未実装でよい。
実装図: <https://claude.ai/artifact/9ARfwzv9BS3RHK5Q5qoZkA>（非公開。オーナーのみ）

## 3. 決まったこと

| 決定 | 理由 |
|---|---|
| 弁のサーボを SCS0009 から **SG90（PWM）** に替える | SCS0009 が不良。弁は開閉の 2 位置だけで、読み戻しは記録用だけ |
| マイクは **MIC1（MCU の SAI1）が本命**、MIC2（MI2S0、1.8V）は使わない | SAI1 のほうが約 26 ms 速い。`pcb_design.md` 151 行目付近の「SAI1 は使わない」は古い記述（未修正） |
| UNO Q の **UNO Ping を廃止**、**arduino-app-cli を停止** | 起動のたびに MCU を書き換え、arduino-router が /dev/ttyHS1 を横取りしていた |
| UNO Q で動くものは **torch を使わない** | UNO Q に torch は無い。`tests/test_yamabiko_uno_q_imports.py` で検査 |
| 実機は **常駐プログラム経由で動かす** | シリアルは 1 プロセスしか持てない。安全側への戻しを 1 か所にまとめる |

## 4. UNO Q の扱い

| 項目 | 内容 |
|---|---|
| 接続 | USB-C、`adb`（シリアル 2261748543）。adb は `arduino` ユーザーで動く |
| sudo | `airpocket` のパスワードが要る。ユーザーに `adb shell` → `su - airpocket` → `sudo ...` を打ってもらう（エージェントはパスワードを入力しない） |
| 再起動 | `adb reboot` は拒否される。USB を抜いても本体の電源は落ちない。`sudo reboot` を頼む |
| 常駐プログラム | `scripts/yamabiko_server.py`。arduino の crontab の `@reboot` で起動。ログは `/home/arduino/yamabiko_server.log` |
| 起動時にすること | `gpioset -c /dev/gpiochip1 -t0 37=0 70=1`（arduino-router のユニットがしていた操作。これがないと MCU のスケッチが動かない）→ ファームウェアに問い合わせ → 無応答なら `build_fw` を書き込む |
| 止めたもの | arduino-app-cli（`sudo systemctl disable --now`。戻すときは `enable --now`）、arduino-router、UNO Ping の既定アプリと crontab の音声処理（`/home/arduino/crontab.backup-2026-11-20*` に控え） |
| ファームウェアのビルド | UNO Q 上で `arduino-cli compile`（[yamabiko_real.md](yamabiko_real.md) の「準備」）。出力は `/home/arduino/yamabiko/build_fw` |
| ファイルの配置 | `bash scripts/uno_q_push.sh`（Git Bash）で flute_rl とスクリプト、ファームウェアのソースを `/home/arduino/yamabiko` へ |
| numpy | あり（2.2.4）。torch は無い |

Git Bash から adb を使うときは `MSYS_NO_PATHCONV=1` を付ける。`adb shell "..."` の中で `pkill -f 名前` や `pgrep -f` を使うと、
そのシェル自身に一致して自分を落とす（`[b]racket` の awk で PID を選ぶ）。

## 5. PC から動かす（命令の一覧）

```
python scripts/yamabiko_remote.py status | info | stop
python scripts/yamabiko_remote.py watch --seconds 10
python scripts/yamabiko_remote.py actuator 0.3 --hold 2       # 上限 0.6（limit 0.8 で変える）
python scripts/yamabiko_remote.py fan on --duty 0.5 --hold 5
python scripts/yamabiko_remote.py valve open --hold 3
python scripts/yamabiko_remote.py mic --watch 10
python scripts/yamabiko_remote.py record 3 --out a.wav
python scripts/yamabiko_remote.py calibrate --max-pwm 0.6 --sweep 0.5 --pull runs/real/
python scripts/yamabiko_remote.py play runtime.npz target.npz --repeats 3 --pull runs/real/
```

Python からは `from flute_rl.yamabiko.remote import Remote`（`with Remote() as rig: rig.actuator(.3)`）。

- 出力はそれを出した接続のもの。接続が切れるか 2 秒無音なら、アクチュエーター 0・弁閉・ファン停止
- 常駐中は単独のスクリプト（`yamabiko_link.py`、`yamabiko_bench_watch.py` など）はシリアルを開けない
- `calibrate` と `play` は実機（笛・モーター）ではまだ一度も通していない

## 6. 次にやること（順番）

1. ~~**SG90 用のファームウェア**~~ 済み（`2d91b66`。既定は開 1800 µs、閉 1200 µs、`valve_positions` で変える。弁に付けたら角度を合わせる）
   - D7 を常に LOW にして U3 を通しっぱなしにし、D1（PB6）から 50 Hz の PWM を出す。TIM4 はアクチュエーター（20 kHz）が使っているので別のタイマー（PB6 は TIM16_CH1N など）
   - 開・閉の位置はパルス幅（µs）で持つ。`K` 命令の意味を変えるか新設する。状態パケットの `servo_pos` と「サーボ無応答」の旗は意味がなくなる
   - **指令が 100 ms 途絶えたらファンも止める**（今はアクチュエーターと弁だけ。`io_tick()` の見張りに 1 行）
   - SG90 は J4 に 1 GND（茶）、2 6V（赤）、3 信号（橙）。コネクターは 2.54 mm なので変換が要る
2. **モーターとファンをつないで確認**: `actuator ±0.3` で動く向きと電流（`isense_mv`）、`fan on` の duty 0.3／0.7 で回転数が変わるか
3. **口笛の音程**: `mic --watch 10` でマイクから 10 cm で吹く（`pitch_hz` が 800〜2000 Hz）
4. **手順書どおりに実機で回す**（[real_rig_runbook.md](real_rig_runbook.md)）: 0 の確認 → 1 較正 → 2 ツインの当てはめ（PC）→ 2b NN の学び直し（PC、GPU）→ 3 演奏
5. **GPU のある PC で torch のテストを回す**: `tests/test_yamabiko_twin_finetune.py`、`tests/test_yamabiko_rig_adaptive.py` など。この PC には torch が無く、`ac720da` と `9574f4e` で触った箇所は torch なしのテストしか通していない
6. 文書の掃除: `pcb_design.md` の SAI1 についての古い節

## 7. 基板の既知の注意（監査より）

- **U3 の足と UNO Q の JSPI（6 ピン SPI ヘッダー）**: U3 の 7 番（GND）が JSPI の内側の列の真上に来る。今回は通電できたが、足を短く切ってカプトンテープで絶縁しておくのが安全
- **SCS0009 の信号レベル**: 3.3V 駆動。SG90 に替えるので問題にならない見込み
- **+12V の配線が細い箇所**と **U2（NJM7806FA）の放熱**: モーター・サーボの電流が大きいときに温度を見る
- **J5 まわり（Q4・R15・R16・D1）**: 設計は DNP だが実装済みで、動作も確認済み。電磁弁をつなぐなら D1 は必須
- **J4 は XH（2.5 mm）**: SCS0009・SG90 の 2.54 mm のコネクターとは形が違う

## 8. この PC の環境

- Windows、PowerShell と Git Bash。Python 3.12（numpy、pytest あり、torch なし）
- torch なしのテストは `python -m pytest -q --ignore=tests/test_yamabiko_rig_adaptive.py --ignore=tests/test_yamabiko_twin_finetune.py`（2026-11-20 時点で 87 件合格、11 件スキップ）
- KiCad 10（`C:\Program Files\KiCad\10.0\bin`）。基板データは `hardware/`、生成元は `hardware/gen/design.py`
- PlatformIO あり。CoreS3 への書き込みで進捗バーが cp932 で落ちるので `PYTHONIOENCODING=utf-8` を付ける
- `runs/` は git 管理外。学習済みの重み（`runs/yamabiko_residual_v1.pt` など）はこの PC には無い

## 9. 関連文書

| 文書 | 中身 |
|---|---|
| [real_rig_runbook.md](real_rig_runbook.md) | 実機の手順書（較正 → ツイン → 学び直し → 演奏） |
| [yamabiko_real.md](yamabiko_real.md) | ファームウェアと実機のスクリプト、マイクの測定、今回の確認、命令の仕組み |
| [fitting-twin-plan.md](fitting-twin-plan.md) | 開発計画と合格基準 |
| [pcb_design.md](pcb_design.md) | 基板の設計メモ |
| `hardware/README.md` | 基板の製造と組み立ての注意 |
