# ラズパイで ICS-43434 を先に試す

UNO Q の SAI1 に行く前に、**素性の分かっている I2S ホスト**でマイクと音程抽出を通しておく。
ここで通っていれば、UNO Q で音が出ないときに「マイクが悪いのか、SAI の設定が悪いのか」を切り分けられる。

## 配線（Raspberry Pi の I2S）

| ICS-43434 ブレークアウト | Raspberry Pi | 40 ピンヘッダー |
|---|---|---|
| VDD | 3V3 | 1 または 17 |
| GND | GND | 6, 9, 14, 20, 25, 30, 34, 39 |
| SCK（BCLK） | GPIO18 / PCM_CLK | 12 |
| WS（LRCLK） | GPIO19 / PCM_FS | 35 |
| SD（DOUT） | GPIO20 / PCM_DIN | 38 |
| **LR** | **GND** | 左チャンネル |

ICS-43434 は 1.65〜3.63V なので **3V3 直結**。5V には繋がない。
**VDD を先に入れてからクロックを入れる**（データシートが逆を禁じている）。
配線は短く。SD の駆動能力は 85pF まで。

## ラズパイ側の設定

`/boot/firmware/config.txt`（古い OS では `/boot/config.txt`）に追記して再起動:

```
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

`googlevoicehat-soundcard` は 48kHz ステレオ 32bit スロットの I2S 入力を出すので、
ICS-43434 の要求（64 SCK / フレーム、24bit を 32bit スロットに MSB 詰め）と合う。
再起動後に device を確認する:

```
arecord -l
```

## 実行

このリポジトリを持ってきて:

```
git clone https://github.com/airpocket-soundman/chikuwa-flute-robot
cd chikuwa-flute-robot
sudo apt install -y alsa-utils python3-numpy
python3 scripts/mic_test.py --device hw:0,0 --seconds 3
```

録音中は定常的な音を鳴らす（音叉、スマホの発振アプリ、笛そのもの）。
何も鳴らさなくても暗騒音の値は出る。**その暗騒音がこの作品の位置センサーのノイズ**なので、
どちらも一度は取っておく。

## 読み方

スクリプトは 3 つを順に答える。

**1. 音が出ているか**

```
  left  (slot 0):   -17.0 dBFS   peak 0.20000
  right (slot 1):    -inf dBFS   all zero
```

**2. L/R の配線が合っているか**

| 出方 | 意味 | 手当て |
|---|---|---|
| 左だけ動く | **正常**（LR = GND） | — |
| 右だけ動く | LR が High | GND へ落とす |
| 両方に同じ波形 | フレーム同期が 1 スロットずれている | オーバーレイ / SAI のフレーム設定を見直す |
| 両方ゼロ | 音が来ていない | VDD → SCK/WS が出ているか → SD の線、の順で当たる |

**3. 音程が読めるか**

```
  median   880.00 Hz   spread   0.8 cents   confidence 0.95
```

`spread`（セント）が本番で効く数字。`flute_rl/pitch.py` の YIN を実音で通した結果で、
これがそのまま**音響位置センサーの分解能**になる。
`yamabiko.md` の誤差予算は管長 150mm で 1mm あたり 11.5 セントなので、
たとえば 5 セントのばらつきは 0.4mm 相当。

先頭 20ms は捨てている（ICS-43434 はクロック開始から 10.7ms 無音、20ms まで感度が整わない）。

## この後

ここが通ったら UNO Q の SAI1 へ移る。ピンと手順は [`i2s_mic_trial.md`](i2s_mic_trial.md)。
同じ `scripts/mic_test.py` が UNO Q の Linux 側でもそのまま使える（MI2S0 経由の場合）。
MCU 側 SAI1 の場合は、MCU で取ったバッファを Bridge 経由で降ろしてから
`--from-raw` に食わせる。
