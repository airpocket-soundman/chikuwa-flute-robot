# 引き継ぎ — やまびこ1号の基板とマイク（2026-09-18）

クラウド（Claude Code on the web）で進めた分の引き継ぎ。**ローカルの Claude Code で続ける**ための指示。
ブランチは `claude/upbeat-johnson-krcm1g`。`main` は触っていない。PR はまだ無い。

## 何をしていたか

やまびこ1号の**基板 2 枚**を KiCad 10 で設計する。その前段として、
**マイク（ICS-43434）が本当に音程を返すか**を実機で確かめるところまで来た。

| 枚 | 名前 | 載せ先 |
|---|---|---|
| A | ドライバーシールド | Arduino UNO Q の R3 ヘッダー |
| B | オーディオ子基板 | UNO Breakout Carrier（ASX00085）の J14 / J15 |

どちらも外部デバイスとは XH コネクタでつなぐ。

## 決まったこと（根拠つき）

1. **拡張ボードは UNO Breakout Carrier（ASX00085）**。`docs/idea.md` が前提にしていた
   UNO Media Carrier ではない。よって「Media Carrier のマイク端子 → Linux」という保険経路は
   B 基板の `MIC2_INP/INM/BIAS`（J14-20/22/24）に置き換わる。
2. **マイクは MCU の SAI1（3.3V）につなぐ。** ピンは PE5(SCK) / PE4(FS) / PE6(SD)、すべて AF13。
   Carrier では J15-8 / 9 / 10。Zephyr の `hal_stm32` の
   `dts/st/u5/stm32u585aiix-pinctrl.dtsi` で AF を確認済み。
3. **LR は GND**（左チャンネル、スロット 0）。内部 100kΩ プルダウンがあるのでこちらがフェイルセーフ。
4. **48kHz 固定。** ICS-43434 の High-Performance モードは 23〜51.6kHz なので、
   16kHz は選べない（Low-Power モードに落ちて SNR が下がる）。
   SCK 上限は 3.30MHz で、48kHz の 3.072MHz は その 93%。逃げ道は 44.1kHz まで。
5. **SCK を止めない。** クロック開始から出力開始まで 32,768 SCK = 10.7ms、
   感度が整うまで最大 20ms。`idea.md:138` の「5〜15ms」も、やまびこの休符（50〜120ms）も
   これで消し飛ぶ。消費電流 490µA なので止める利得もない。

## 未確定（ここから再開する）

1. **電源電圧。** アクチュエーターが 24V（`idea.md:41`）、ファンが 12V（`yamabiko.md:74`）で食い違う。
   24V 入力 → 12V/6V 降圧か、12V 入力でアクチュエーターを 12V 品にするか。**ユーザーに聞く。**
2. **アクチュエーターのストール電流。** H ブリッジの型番と XH の線径が決まらない。**ユーザーに聞く。**
3. **D0/D1 の衝突。** `mcu/uno_q_hil/uno_q_hil.ino:27` の HIL が `Serial1` を使う。
   SCS0009 も UART なので、本番は Bridge 経由で D0/D1 を空けられるかを確認する。
4. **UNO Q の D ピン ↔ STM32U585 の対応と PWM タイマー。**
   `pcb_design.md` の A 基板のピン割り当ては Uno R3 の慣習からの推定。
   UNO Q のデータシート（ABX00162）で裏を取る。**ローカルなら公式サイトを直接読める。**
5. **Arduino コア（Zephyr ベース）から SAI1 を触れるか。** マイクを MCU 側で動かす前提の根幹。
   `I2S` ライブラリが SAI1 に張られていなければ、DT オーバーレイ + Zephyr の `i2s` ドライバーか HAL 直叩き。

1〜5 が埋まれば KiCad ファイルの生成に入れる。**まだ `.kicad_sch` も `.kicad_pcb` も 1 つも作っていない。**

## 次の一手

**ラズパイで先にマイクを通す。** 素性の分かった I2S ホストで通しておけば、
UNO Q で音が出ないときに「マイクが悪いのか SAI の設定が悪いのか」を切り分けられる。

1. `docs/mic_test_on_pi.md` の配線と `config.txt` のオーバーレイ
2. `python3 scripts/mic_test.py --device hw:0,0 --seconds 3`
3. 通ったら UNO Q の SAI1 へ（`docs/i2s_mic_trial.md` の手順 1〜6）
4. 手順 6 の**遅延測定が本題**。`idea.md:138` の 5〜15ms に収まるかで制御の組み方が決まる

ユーザーは「UNO Q に書き込まれて自動起動しているサービスは全部初期化してよい」と言っている。
ただし**ラズパイで通すまで UNO Q には触らないほうが変数が減る**。

## ローカルに移ってできるようになること

クラウド側では次の 2 つが塞がれていた。ローカルでは両方できる。

- **実機に触れない。** ラズパイ・UNO Q への USB / シリアル / SSH が一切届かなかった。
  `arecord` の実行も `config.txt` の編集もユーザーに手作業をお願いしていた。
- **外部サイトが読めない。** TDK・Arduino 公式・スイッチサイエンス・Mouser・DigiKey が
  すべて egress でブロックされ、データシートはユーザーに貼ってもらった
  （`docs/datasheets/` に 2 本保存済み）。

## 書いたもの

| ファイル | 中身 |
|---|---|
| `docs/pcb_design.md` | 2 枚の基板の設計メモ。デバイス一覧、ピン割り当て、コネクタ表、機械寸法、未確定事項 |
| `docs/i2s_mic_trial.md` | SAI1 のピン指定、SAI の設定値、データシート実値、試す順番 |
| `docs/mic_test_on_pi.md` | ラズパイでの先行テスト。配線、オーバーレイ、出力の読み方 |
| `scripts/mic_test.py` | 録音 → L/R 配線の診断 → 音程抽出。numpy と alsa-utils だけ |
| `docs/datasheets/ASX00085-*.pdf` | UNO Breakout Carrier |
| `docs/datasheets/ICS-43434-*.pdf` | マイク（DS-000069 Rev 1.2） |

## 参考にした先行作

<https://github.com/airpocket-soundman/IchiPing-UNO-Q> に、**同じ 2 枚構成が製造・動作済み**。
コネクタの型番（秋月 112247〜112251）、ソケットの選び方、設定は 0Ω ではなくジャンパ、
そして **J14/J15 の奇数列・偶数列の向きを一度間違えている**（`board/REPAIR_STATUS.md`）。
ここは必ず公式資料の上面図と嵌合面で照合する。

なお同プロジェクトのマイクは **INMP441 を 1.8V の MI2S0（SoC 側）**で動かしている。
やまびこが MCU 側 SAI1 を選ぶのは、低遅延フィードバックがこの経路の前提だから
（`idea.md:138`）。**MCU 側で動いた公開例は見つかっていない**ので、そこは賭けになっている。
ラズパイでの先行テストと、手順 6 の遅延測定は、この賭けを早く判定するためにある。

## 追記（2026-09-18、ローカル）— MCU の SAI1 でマイクが動いた

ラズパイは使わず、**UNO Q の MCU（SAI1）で直接**試した。マイクは Breakout Carrier の J15-8/9/10 に配線済み。

**Zephyr コア 0.53.1 のローダーには `CONFIG_I2S` が無い**ので、`mcu/sai1_mic_test/sai1_mic_test.ino` は
RCC / GPIOE / SAI1 を CMSIS で直接叩く（スケッチは特権モード・MPU 無効で動く）。
クロックは HSE 16MHz → PLL3（M=4, N=49, FRACN=1245, P=4）= 49.152MHz、MCKDIV=16 で SCK 3.072MHz / 48kHz。

| 項目 | 結果 |
|---|---|
| PLL3 | ロック、SAI1 のクロック源 |
| サンプリング | 実測 48,031.9Hz（物差しは DWT、CPU 160.013MHz） |
| L/R | 左だけ動き、右は全部 0 → LR = GND どおり、フレームの位置も合っている |
| オーバーラン | 8192 フレーム（0.17 秒）で 0 |
| 中身 | 下位 8bit は常に 0（24bit を MSB 詰め）。隣り合うサンプルの相関 0.98〜0.99、エネルギーの 9 割以上が 4kHz 以下 → 本物の音 |
| レベル | 部屋の音で -44〜-58 dBFS |

**未確認:** 定常音での音程の読み（PC のビープは出せなかった）と、手順 6 の遅延測定。

### 再開の手順

- UNO Q には **adb で入れる**（USB、`adb devices`）。Git Bash では `MSYS_NO_PATHCONV=1` を付ける。
  `adb shell` 内で arduino-cli を使うときは `TMPDIR=/tmp HOME=/home/arduino` を付ける。
- ビルドと書き込みは UNO Q 上で行う:
  `arduino-cli compile -b arduino:zephyr:unoq --output-dir build sai1_mic_test` のあと、
  `~/.arduino15/packages/arduino/tools/remoteocd/0.0.4-rc.4/remoteocd upload -f <variant>/flash_sketch.cfg build/sai1_mic_test.ino.elf-zsk.bin`
- スケッチは Serial1（= `/dev/ttyHS1`、921600 baud）を占有するので、**arduino-router を止める**。
  止めるには sudo のパスワードが要るので、ユーザーに実行してもらう:
  `adb shell -t "su - airpocket -c 'sudo systemctl stop arduino-router arduino-router-serial'"`
- `python3 scripts/sai1_capture.py info | stats | capture --out /tmp/x.raw`（UNO Q 上、標準ライブラリのみ）。
  UNO Q には numpy が無いので、解析は PC で `python scripts/mic_test.py --from-raw x.raw --channels 1 --rate 48000`。
- IchiPing アプリ（`user:ichiping-uno-q`）は止めてあり、MCU は試験用スケッチで上書きしてある。

### 次の一手

1. マイクのそばで 440Hz / 880Hz の定常音を鳴らして `capture` → 音程とばらつき（セント）
2. 遅延測定。方法（MCU がブザーを鳴らして DWT で測る / オシロ）はユーザーと相談中
3. ユーザーの問い「Linux 側（MI2S0）につないだほうが早くないか」。MCU⇔Linux が 115200 baud の UART しかなく、
   GRU の C 版も無いので、**全体構成では Linux 側が素直**という見立てを伝えてある。MCU で動くことは確かめたので、
   遅延を測ったうえでどちらにするか決める（MI2S0 は 1.8V 系なのでマイクの VDD も 1.8V にする）
