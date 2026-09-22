# 長期音楽文脈・コード理解・アドリブ演奏 拡張計画

更新日: 2026-09-22

## 目的

お手本を正確に再現する機能を土台に、数小節から8小節以上の長期文脈を理解し、調性・コード進行・フレーズ構造を保持した上で、物理的に演奏可能なアドリブを生成する。最終目標は「音を真似る装置」ではなく、「音楽構造を聴き、身体の制約に合わせて新しい応答を演奏するフィジカルAI」である。

この拡張はお手本再現Gateを置き換えない。再現モデルと生成モデルは共通の聴覚・拍・音楽メモリを共有し、`Imitation Head` と `Improvisation Head` に分岐する。

## 全体構造

```text
reference waveform
  ├→ Raw Waveform Encoder ─────────────┐
  └→ Multi-resolution STFT Encoder ────┤
                                       ↓
                              Audio Fusion
                                       ↓
                              Tempo / Beat NN
                                       ↓
                         Beat-synchronous tokenizer
                                       ↓
                    Long-context Musical Transformer
                      ├→ melody/rhythm memory
                      ├→ harmonic latent
                      ├→ phrase/bar structure
                      └→ style/interaction context
                               ┌───────┴────────┐
                        Imitation Head   Improvisation Head
                               └───────┬────────┘
                                  Playability NN
                                       ↓
                              physical controller
```

## なぜSTFTを使うか

コードは複数周波数の同時関係であるため、生波形だけより周波数表現の方が学習効率を高めやすい。STFTは時間×周波数の2次元表現であり、和声の時間変化は「スペクトログラム画像の右側を予測する」問題に近い。

ただし単一FFTでは時間変化を失うため、複数窓長のSTFTを使用する。

| 窓 | 主な役割 |
|---|---|
| 約32 ms | onset、短い発音、リズム |
| 約128 ms | メロディ、主要倍音 |
| 約256 ms以上 | 低音、コード構成音、和声 |

STFTはコードを決める規則ではなく、微分可能な入力表現とする。コード、調性、休符、続きを判断する処理はNNが学習する。E2E性を強くするため、生波形branchも残し、attention fusionで統合する。

## 拍同期トークン

スペクトログラム全画素の続きを直接生成すると、音色と位相の再現へ容量を使いすぎる。そこでSTFTを拍同期の音楽トークンへ圧縮する。

4/4拍子、16分音符単位なら8小節は128セルであり、長期Transformerが扱いやすい。各セルは次を保持する。

- 連続音程と音程傾斜
- voiced/rest確率
- onset/sustain/release/offset状態
- 音価と拍内位置
- ビブラート、グリッサンド等の表情
- bass/root候補
- chord/harmony潜在表現
- 小節・フレーズ境界
- 音色・演奏スタイル潜在表現

十二平均律へ強制量子化せず、連続centを保持する。コード名は診断ヘッドとして出力できるが、生成の主経路は自由な和声潜在ベクトルとする。

## 長期メモリ

段階的に文脈長を伸ばす。

1. 1小節の復元
2. 2小節の反復・変奏理解
3. 4小節の問いと答え
4. 8小節のコード進行とフレーズ
5. 16小節以上のセクション構造

学習課題は複数を併用する。

- masked cell/bar reconstruction
- 過去4小節から次4小節を予測
- 同じ曲の異なる音色・移調版を近い潜在表現へする対照学習
- BPM変更後も同じ拍構造を保持する学習
- chord、key、bar boundaryラベルがあるデータで補助教師学習
- ラベルなし音源の次トークン／次小節予測

## コード理解

単旋律からコードは一意に決まらない。したがってコード理解は一点推定ではなく候補分布として扱う。

```text
Cmaj7 0.55 / Am7 0.25 / Em7 0.15 / other 0.05
```

伴奏を含む音源ではSTFTから直接和声を学習する。単旋律だけの場合は、大規模な音楽事前分布と前後文脈を使い、曖昧性をconfidenceに反映する。評価では「正解コード名」だけでなく、正解を候補上位に含む率、コード変化位置、次小節予測への寄与を測る。

BPMの半分・2倍を選ぶ曖昧性も同様に複数仮説として保持し、小節境界、onset、コード変化との整合で選ぶ。

## アドリブ生成

`Improvisation Head` は過去の音楽メモリ、現在の和声、直前に自分が演奏したフレーズ、物理的な演奏可能性を条件として未来の拍セルを生成する。

出力は以下とする。

- 次セルの連続音程分布
- rest/voiced/onset/offset確率
- 音価
- 音程傾斜と表情
- フレーズ終了確率
- harmonic fit confidence
- playability confidence

生成モードを分ける。

1. `continuation`: お手本の自然な続きを生成
2. `variation`: 元フレーズを保った変奏
3. `call_and_response`: お手本への応答
4. `solo`: コード進行に沿った自由アドリブ
5. `constrained`: 指定音域・密度・難易度内で生成

生成時にはtemperature、top-p等を使えるが、物理安全と音域制約は `Playability NN` と安全層で保証する。

## Playability NN

音楽的に妥当でも物理的に演奏できない旋律は採用しない。Playability NNは候補系列と現在のrig embeddingから次を予測する。

- 音域内か
- 次音までに移動可能か
- 必要PWM、予想到達誤差
- 発音開始までの余裕
- オーバーブロー・端当たりリスク
- 必要な休符・回復時間

候補生成時のrerankerとして使い、最終的には生成器自体へplayability損失を逆伝播する。手書き制約版もhybrid比較として保持する。

## 学習段階とGate

| Gate | 問い | 主な評価 |
|---|---|---|
| A1 Spectral | 複数音・倍音を保持できるか | STFT再構成、音色外汎化 |
| A2 Beat Tokenizer | 音を拍セルへ正しく割り当てるか | BPM、phase、境界F1、休符F1 |
| A3 Long Memory | 8小節を保持できるか | raw破棄後の8小節復元、順序対照 |
| A4 Harmony | 調性・コード変化を理解するか | chord recall、root/bass、境界時刻 |
| A5 Continuation | 文脈に沿う続きを作れるか | held-out future予測、反復コピー率 |
| A6 Improvisation | コピーでない妥当な変奏か | novelty、和声適合、フレーズ性 |
| A7 Playability | 物理的に演奏できるか | Oracle到達率、予測誤差、危険率 |
| A8 Physical Improv | 実際の笛で成立するか | 生成理想WAV対物理WAV、反復改善 |

## 評価上の注意

- 単一の主観スコアだけで成功を決めない
- train曲の断片コピーを検出する
- chord labelがなくても評価可能なcontinuation/self-supervised指標を持つ
- 理想生成WAVと物理演奏WAVを分離する
- 同じコードへの複数の正しいアドリブを許容する
- 音楽性評価と物理再現評価を混ぜない
- 人間評価ではblind比較と複数評価者を使う

主な指標候補は、pitch/rhythm distribution、chord-tone適合率、休符・onset F1、フレーズ反復、自己類似構造、novelty、playability、物理MAE、発音率、遅延である。

## WAVとHTML

各Gateで以下を公開する。

- 原音
- STFT/音楽トークンから再構成した音
- 記憶だけから再構成した8小節
- 推定和声を診断用伴奏として鳴らした音
- 正解未来と予測未来
- 理想アドリブWAV
- 物理シミュレータ演奏WAV
- 複数take後の適応済み演奏WAV

生成例はbest sampleではなく固定ホールドアウトの中央値例を基本とし、seed、モデルSHA、生成条件、temperature、使用ブロック構成をmanifestへ保存する。

## データ計画

段階的に以下を使用する。

1. BPM、拍、小節、コード、旋律を完全に制御できる合成データ
2. 音色・残響・雑音・テンポ揺れを加えた合成データ
3. chord/key/beatラベル付き公開音楽データ
4. ラベルなし音源による自己教師学習
5. 実機録音と伴奏を含む独自データ

train/validation/testは曲単位、可能なら作曲者・進行パターン単位で分離する。同一曲の移調・テンポ変更版がtrainとtestへ跨らないようにする。

## 計算資源とUNO Q

長期音楽理解はまずPC教師モデルで成立させる。8小節16分音符グリッドは128トークン程度なので、音響フレームを直接長時間保持するより小さい。

配備案は2通り比較する。

1. PC/UNO Q上で全段を実行する完全オンデバイス
2. お手本を聴いた直後に大きなencoderで音楽メモリを生成し、演奏中は小さなdecoder/controllerだけを100 Hz実行する

成立後に蒸留、量子化、attention削減、メモリ圧縮を行う。コード理解とアドリブ生成は低頻度、物理制御は100 Hzとし、マルチレート実行でdeadlineを守る。

## マイルストーン

1. Tempo/Beatと拍グリッド記憶をお手本再現経路で成立
2. Multi-resolution STFT Encoderの単独Gate
3. 1〜2小節masked reconstruction
4. 4〜8小節Long Musical Memory
5. Harmony Headと曖昧性評価
6. Continuation Head
7. Variation/Call-and-response生成
8. Playability NN接続
9. 物理シミュレータでのアドリブ演奏
10. joint E2E微調整、UNO Q圧縮、実機評価

## 完了条件

- 8小節入力後に音声入力なしで拍・音程・休符・和声文脈を保持
- 未知曲・未知音色・移調・テンポ変更で長期Gateを通過
- 単純コピーではないアドリブを生成
- 生成理想音と物理演奏の両方を公開
- 再現Headを壊さずImprovisation Headを追加
- 全learned、個別接続、主要hybrid構成を比較
- すべての結果がcheckpoint、manifest、WAV、HTMLから再現可能

