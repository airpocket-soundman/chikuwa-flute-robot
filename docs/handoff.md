# 引き継ぎ — やまびこ1号（2026-11-20）

A・B 基板を UNO Q に載せて初めて通電し、PC から命令で動かせるところまで来た。その時点の引き継ぎ。
前の版（2026-09-18、基板設計の前段）は git の履歴にある（`git log -- docs/handoff.md`）。

ブランチは `main` だけ（作業ブランチはすべて取り込み済み）。最新のコミットは `2518521`。

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
