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

## 追記（2026-09-18、ローカル）— 定常音で音程を読んだ

PC のスピーカーから正弦波（440 / 880Hz、振幅 0.5）を鳴らし、マイクを 10〜30cm 程度に置いて `capture` を 5 回ずつ
（1 回 8192 フレーム = 0.17 秒）。解析は PC で `python scripts/mic_test.py --from-raw ... --channels 1 --rate 48000`。

UNO Q の再起動で MCU のスケッチが消えていた（`info` が無応答）。`~/.cache/arduino/sketches/A1551A27641A0702B55914A04FE66C3C/`
のビルド済み `.elf-zsk.bin` を remoteocd で書き直せば、ビルドし直さずに戻る。arduino-router の停止は、`adb shell` に入ってから
`sudo systemctl stop arduino-router arduino-router-serial`（ユーザー arduino のパスワード）で通った。

| | レベル | 音程の中央値 | ばらつき |
|---|---|---|---|
| 無音（部屋） | -64〜-68 dBFS | — | — |
| 440Hz × 4 回 | -36.3 dBFS | 440.02〜440.08 Hz（+0.1〜+0.3 セント） | 0.2〜0.3 セント |
| 880Hz × 5 回 | -36.9 dBFS | 879.94〜879.98 Hz（-0.05〜-0.1 セント） | 0.1 セント |

- **マイク → SAI1 → YIN の経路のばらつきは 0.1〜0.3 セント。** 管長 150mm で 1mm あたり 11.5 セントなので、
  位置に直して約 0.03mm。シミュレーターの `pitch_noise`（3 セント）よりひと桁小さい。
  ただし純音・無風・近距離の値で、笛の息の音・音程のゆらぎ・ファンの騒音は入っていない（下限の確認）
- **絶対値も 0.3 セント以内で合う。** スケッチが出す `measured_hz 48031.9` をそのまま信じると、音程は一律 -1.15 セントずれるはずだが、
  そうなっていない。SAI の実際のサンプリングは PC の DAC の時計に対して 48,000Hz にほぼ一致していて、
  **48,031.9 は物差し（DWT / CPU クロックの見積もり）の誤差**と読める
- 無音時の -64 dBFS は、`pitch_track` の足切り（-60 dBFS）のすぐ下。ファンを回すとここが上がるので、
  ファンの騒音を録ったら、足切りと笛の音量の差を見直す

残り: 手順 6 の遅延測定。試験後も arduino-router は止めたまま（戻すときは `sudo systemctl start arduino-router arduino-router-serial`）。

## 追記（2026-09-18、ローカル）— MCU 経由で Linux に音を流す経路と、その遅れ

`mcu/yamabiko_link/yamabiko_link.ino` と `scripts/yamabiko_link.py`（UNO Q 上、標準ライブラリのみ）。

- **MCU**: SAI1 は `sai1_mic_test` と同じ設定のまま、**GPDMA1 チャネル 7 で 16KB のリングへ連続転送**（リンクリストが自分自身を指す循環。割り込みなし）。
  ループは残りバイト数から書き込み位置を読み、左スロットを 31 タップ FIR（7kHz）で 16kHz に間引き、上位 16bit にして、
  **5ms（80 サンプル）ごとにパケット**で Serial1（921600 baud、32kB/s ＝帯域の約 1/3）へ送る。パケットには通し番号と 16kHz のサンプル番号が入る
- **Linux**: `Link.packets()` が音声を届いた順に、到着時刻とサンプル番号つきで返す。**制御器や NN はここから受け取る**
- 遅れの測り方: Linux から ping を打ち、MCU は「ping を読んだ時点で取り込み済みの SAI フレーム数」を返す。往復の短い ping で両者の時計を結び、
  各パケットが届いた時刻から、その中の最新サンプルが録音された時刻を引く

つまずき: 動いている GPDMA チャネルは `EN = 0` を無視するので、`RESET` の前に `SUSP` で止めて `SUSPF` を待つ。これを忘れると、2 回目の開始で MCU が固まる。

**結果（60 秒、11,961 パケット）**

| 項目 | 値 |
|---|---|
| 最新サンプルが Linux のプロセスに届くまで | **中央値 2.73ms**、p95 2.83、p99 2.87ms |
| 遅れたパケット | 5ms 超が 5 個、10ms 超が 4 個、20ms 超が 2 個（最大 28ms）。Linux 側のスケジューリングと見られる |
| 取りこぼし | 通し番号の欠け 0、チェックサム不一致 0、MCU のオーバーラン 0 |
| 時計の結びつけの誤差 | ±0.59ms（往復の短い ping の往復時間の半分） |
| SAI のサンプリング周波数（Linux の時計で） | 47,999.0Hz（-20ppm）。スケッチが出していた 48,031.9Hz は DWT の物差しの誤差と確定 |
| 間引き FIR の遅れ | +0.31ms（上の数字には入っていない） |
| 中身の確認 | PC から 880Hz を鳴らして 3 秒録音 → 880.18Hz、ばらつき 0.1 セント、欠けなし |

2.7ms の内訳は、パケット 171 バイトを 921600 baud で送る時間（約 1.9ms）と、Linux 側の受け取りの遅れ（約 0.8ms）でほぼ説明できる。
やまびこのシミュレーターの聞こえる遅れ（`obs_delay` 公称 3 ステップ = 30ms、機体差 1〜5 ステップ）より 1 桁小さい。
ただし音程の推定には窓（16kHz で 320〜512 サンプル = 20〜32ms）が要るので、制御に効く遅れは窓の長さで決まる。

**次: Linux 側 MI2S0 との比較。** 同じ条件で比べるには、どちらの経路にも同じ「時刻の分かった音」を入れる必要がある。
MCU のピンに圧電ブザーをつなぎ、Linux から UART で鳴らした時刻（往復 1ms 未満で分かる）と、音の立ち上がりを検出した時刻の差を測る案。
MI2S0 側はカーネル修正で**取り込みの単位が 20ms に固定**されている（`/home/arduino/ichiping-audio/white-noise-cycle.sh` のコメント）。
（ここで「クロックを出すために再生側を回し続ける必要がある」と書いていたが誤り。下の比較で、録音だけで動くことを確かめた）

## 追記（2026-09-18、ローカル）— MCU 経由と MI2S0 の遅れを同じ物差しで比べた

ブザーの代わりに **ICS-43434 を 2 本並べた**（MCU の SAI1 に 3.3V で 1 本、MI2S0 に 1.8V で 1 本。ピンは重ならない）。
PC のスピーカーから不規則な間隔のチャープ（30ms、500→6000Hz）を鳴らし、`scripts/latency_compare.py record` が
UNO Q 上で両方を同時に読んで到着時刻を記録する。PC の `analyze` が 2 本の録音を相互相関で突き合わせ
（1 秒ごとの窓 19 個、当てはめの残差 0.0 フレーム、2 つのサンプリングクロックの差 +15ppm）、
MCU 側で分かっている録音時刻を MI2S0 の各サンプルに移す。

- 最初の MI2S0 マイクは壊れていて、データが全部 0 だった。交換して解決（-57dBFS、左スロット）
- **MI2S0 は録音だけで動く。** 再生を回さなくてよい（IchiPing の手順は再生と録音を同時に試すためのもの）
- arduino-router は `sudo systemctl disable --now arduino-router arduino-router-serial` で自動起動も切った。
  戻すときは `enable --now`。再起動で `/tmp` が消えるので、UNO Q 上のスクリプトは `/home/arduino/yamabiko/` に置く。
  MCU のスケッチも再起動で消えるので、`build_link/yamabiko_link.ino.elf-zsk.bin` を書き直す

**結果（20 秒、同じ音を同時に）**

最新サンプルがマイクに届いてから Python が受け取るまで [ms]:

| 経路 | 受け取りの単位 | 中央値 | p95 | p99 | 最大 |
|---|---|---|---|---|---|
| MCU SAI1 → UART | 5ms（80 サンプル @16kHz） | **3.0** | 3.2 | 3.3 | 31 |
| MI2S0 → ALSA | 20ms（960 フレーム @48kHz） | **21.6** | 21.8 | 21.9 | 41 |

チャープの立ち上がりそのものが届くまで（そのサンプルがブロックのどこに入るかで変わる）:
MCU 3.1〜7.8ms（中央値 5.1）、MI2S0 21.4〜42ms（中央値 29.5、外れ値 1 個 70ms）。上の表と矛盾しない。

- MI2S0 は、20ms のブロックの**最新サンプルですら 21.6ms 古い**。ブロックの長さとは別に、DSP の中で**もう 1 ブロック分（約 20ms）**たまっている。
  1 つのサンプルの遅れは 21.6〜41.6ms で、平均 31.6ms
- MCU 経由は、どのサンプルでも 3〜8ms（平均約 5.5ms）。**平均で約 26ms 速い**
- やまびこの制御周期は 10ms。音程の推定窓（20ms なら中心で +10ms）を足すと、MCU 経由は約 15ms（1〜2 ステップ）、
  MI2S0 は約 40ms（4 ステップ）で、しかも 2 ステップぶんずつまとめて届く。シミュレーターの `obs_delay`（1〜5 ステップ）には
  どちらも収まるが、MI2S0 は上の端に近い
- ただし MCU 経由は UART を音声が使う（帯域の約 1/3）。アクチュエーターの指令も同じ UART に載せる前提で、プロトコルは作ってある（型つきパケット）

**判断の材料**: 遅れだけなら MCU 経由が明確に良い。MI2S0 の良さは、ファームウェアが要らないこと、UART が空くこと、48kHz のまま使えること。
