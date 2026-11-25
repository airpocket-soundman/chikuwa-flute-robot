# アーカイブの口笛言語・鳥の声を、笛の体で吹き直すデモ（2026-11-25）

[vision.md](vision.md) の 4 章「野鳥の言葉・人の口笛言語・LLM をつなぐ」の最初の検証。
アーカイブの口笛言語（シルボ・ゴメロ）と鳥の声（シジュウカラ）を、今の体（ツインの上の模擬機体）で吹き直し、
音程の形がどこまで残るかを見る。実機でも同じ楽譜を吹ける。

## 1. アーカイブを取る

```
python scripts/archive_fetch.py silbo          # シルボ・ゴメロ（OpenSLR SLR137）、約 530 MB
python scripts/archive_fetch.py tit --pages 2  # シジュウカラ（xeno-canto、日本・品質 A）、35 本・約 210 MB（2026-11-25）
```

`runs/archive/` に置く（git の外）。MP3 を読むには `pip install soundfile`。

| アーカイブ | 中身 | 利用条件 |
|---|---|---|
| シルボ・ゴメロ（`runs/archive/silbo_gomero/`） | 熟練者 4 人の読み上げ。単語・文・短い断片の 3 種と書き起こし。WAV 44.1 kHz | CC BY-NC-SA 4.0。作成 Agata Jakubiak、録音 Silbo Gomero Teaching Project。引用は Interspeech 2023（doi 10.21437/Interspeech.2023-989） |
| シジュウカラ（`runs/archive/xeno_canto/parus_minor/`） | 録音ごとの MP3 と `manifest.json`（録音者・場所・日付・種類・ライセンス） | 録音ごとに違う（多くは CC BY-NC-ND か BY-NC-SA）。解析とデモには録音者を表示して使う。ND のものは編集した音を公開しない |

xeno-canto の API v3 は個人の鍵が要るので、`archive_fetch.py` は公開の検索ページを読んでいる。

## 2. デモを動かす

```
python scripts/archive_demo.py --list silbo                     # 取ったものの一覧
python scripts/archive_demo.py --list bird
python scripts/archive_demo.py runs/archive/silbo_gomero/sentences/<file>.wav --kind silbo
python scripts/archive_demo.py runs/archive/xeno_canto/parus_minor/XC1147321.mp3 --kind bird --seconds 2.3 --slow 4
```

1. **聴く**: 10 ms ごとの音程（YIN。学習側の耳。推論の経路には入れない）。口笛言語は音程の線で言葉を運び、鳥の声も音程の線に近い
2. **体に合わせる**: 笛の音域（プロファイルの low..high）へ移す。原音の幅が音域より広いときは `--fit compress` で音程の間隔を縮め（形は保つ）、`--slow` で時間を伸ばす
3. **吹く**: `--steps` の版（既定は classic と step7c）が、ツインのまわりの模擬機体 `--rigs` 台で吹く（`yamabiko_compare_sim.py` と同じ仕組み、NumPy だけ）

出力は `runs/archive_demo/<日時>_<名前>/`:

| ファイル | 中身 |
|---|---|
| `source.wav` | 切り出した原音（アーカイブの音そのまま） |
| `source_slow.wav` | `--slow` のとき、原音を音程を変えずにデモと同じ速さへ伸ばしたもの（WSOLA） |
| `listen.wav` | 聞き比べ用に全部を順につないだもの: 原音 → デモの速さの原音 → 楽譜 → 各版の演奏（0.7 s 間隔） |
| `target.wav` / `target.npz` | 体に合わせた楽譜（音と、`cents`・`voice`） |
| `<step>.wav` | 模擬機体の 1 台目が吹いた音程を音にしたもの（鳴っている間だけ） |
| `demo.png` | 原音のスペクトログラムと聴き取った線、楽譜、各版の演奏 |
| `report.json` | 下の数値 |

**音を流す**: `--play` で、作り終えた後に上の順でスピーカーから流す（Windows は標準の winsound、Mac/Linux は afplay/aplay）。
前に作ったものをもう一度流すときは `--replay`。シルボ・ゴメロの文なら、流す前に意味（書き起こし）を表示する。

```
python scripts/archive_demo.py runs/archive/silbo_gomero/sentences/sentences/clips/01-sentence-001.wav --kind silbo --slow 2 --play
python scripts/archive_demo.py --replay runs/archive_demo/<run>
```

数値（模擬機体の平均）:
- **shape**: 調をそろえた後の |演奏 − 楽譜| の平均（cent）。口笛言語は音程の線が言葉なので、これが本命の指標
- **cover**: 楽譜の音のある時間のうち、笛が鳴っていた割合
- **corr**: 演奏と楽譜の音程の線の相関
- **fast**（楽譜について）: プランジャーが追いつけない速さ（約 1100 cent/s、ファン 80 %、全行程 1.39 s）を求める割合

**実機で吹く**: 楽譜はそのまま実機に渡せる。

```
python scripts/yamabiko_whistle_copy.py --target runs/archive_demo/<run>/target.npz
```

## 3. 最初の結果（模擬機体 8 台）

### シジュウカラ（XC1147321、先頭 2.3 s。上がっていく短い声が 5 つ）

原音は約 3〜7 kHz で幅 1536 cent。間隔を 0.71 倍に縮めて笛の音域に入れた。

| 時間の伸ばし | fast | classic shape / cover / corr | step7c shape / cover / corr |
|---|---|---|---|
| 1 倍 | 67 % | 90 cent / 24 % / 0.90 | 227 cent / 97 % / 0.56 |
| 2 倍 | | 85 cent / 32 % / 0.91 | 181 cent / 100 % / 0.72 |
| 4 倍 | | 61 cent / 63 % / 0.96 | 116 cent / 100 % / 0.86 |

- そのままの速さでは、楽譜の 3 分の 2 がプランジャーの追いつけない速さ。step7c は鳴り続けるが、声の形はなめらかな山にならされる（corr 0.56）
- 4 倍ゆっくりにすると、step7c で corr 0.86 まで形が残る。classic は合っている所では正確（corr 0.96）だが、外れると弁を閉じて黙るので、鳴っている時間が短い
- **「鳥の声をそのまま返す」には今の体は遅すぎる**。ゆっくり返すか、もっと速い機構が要る。どれだけゆっくりでも鳥に通じるのかは、鳥で確かめるしかない

### シルボ・ゴメロ

**1 文の例**（`01-sentence-001.wav`「¿Cuántos años tiene tu abuelo?」＝「おじいさんは何歳？」、先頭 8 s、模擬機体 8 台）。
原音は約 0.8〜4 kHz で幅 2571 cent（倍音への飛びを一部含む）。間隔を 0.42 倍に縮めた。

| 時間の伸ばし | fast | classic shape / cover / corr | step7c shape / cover / corr |
|---|---|---|---|
| 1 倍 | 43 % | 64 cent / 47 % / 0.93 | 166 cent / 100 % / 0.70 |
| 2 倍 | 26 % | 58 cent / 61 % / 0.94 | 78 cent / 100 % / 0.92 |
| 4 倍 | 7 % | 35 cent / 86 % / 0.98 | 31 cent / 100 % / 0.98 |

**12 文の平均**（話者 4 人 × 3 文、各先頭 8 s、模擬機体 4 台。間隔の縮め方は平均 0.61 倍）

| 時間の伸ばし | fast | classic shape / cover / corr | step7c shape / cover / corr |
|---|---|---|---|
| 1 倍 | 34 % | 73 cent / 45 % / 0.88 | 174 cent / 99 % / 0.64 |
| 2 倍 | 19 % | 53 cent / 63 % / 0.94 | 101 cent / 99 % / 0.84 |
| 4 倍 | 8 % | 34 cent / 83 % / 0.98 | 45 cent / 100 % / 0.95 |

- 口笛言語は、鳥の声より体に合う。そのままの速さでも追いつけない所は約 3 分の 1 で、2 倍ゆっくりで step7c の形が corr 0.84、4 倍で 0.95 まで残る
- そのままの速さだと、step7c は鳴り続けるが速い動きをならしてしまい、classic は合う所だけ正確に吹いて残りは黙る。どちらも「言葉の形」としては欠ける
- **4 倍ゆっくり話せば、今の体でも音程の線はほぼ保てる**。ただ、口笛言語の話者に 4 倍遅い言葉が通じるかは別の問題で、確かめる必要がある
- 音域に入れるため、間隔を平均 0.61 倍に縮めている。シルボ・ゴメロは母音を音の高さの段で区別するので、縮めすぎると母音が聞き分けにくくなるおそれがある
- 耳（YIN）が倍音に飛ぶ所があり、原音の幅を大きめに見積もっている（そのぶん余計に縮めている）

実行の例:

```
python scripts/archive_demo.py runs/archive/silbo_gomero/sentences/sentences/clips/01-sentence-001.wav --kind silbo --slow 2
```

（展開すると `sentences/sentences/clips/` と `sentences/sentences/transc.csv` になる。`words`・`fragments` も同じ形）

## 4. 次の一手

- シルボ・ゴメロの文を何本も吹き直し、速さ（`--slow`）ごとの shape をまとめる。口笛言語の話者に、合成音と笛の音を聞き比べてもらえると一番よい
- 速い機構（サーボ駆動のスライドなど）を足した体をツインに入れて、同じ楽譜で比べる
- 耳を YIN から耳の NN（step5）に替える。鳥の声は 3〜7 kHz にあり、耳の NN の聞く範囲（`ear_04` で 310 Hz〜4 kHz）を越えるので、広げて学び直す必要がある
