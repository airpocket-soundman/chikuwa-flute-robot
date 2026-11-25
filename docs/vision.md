# ビジョン — 人の揺らぎを学び、野鳥と言葉を交わす（2026-11-25）

コンテストでの見せ方（E2E を主役にし、classic_ear を比較の基準にする）は [nn-e2e-plan.md](nn-e2e-plan.md) の 1.2。
この文書はその一段上、「このロボットで何が得られるのか」を書く。

## 1. 何が得られるのか

### 1.1 人らしさを学ぶ

人の口笛は、楽譜どおりではない。音の入りのずり上げ、伸ばしている間のゆらぎ、ビブラート、テンポの溜め。
機械制御で目標の音程に合わせると、この揺らぎは「誤差」として消される。

このロボットは、口笛を楽譜（音程の列）に直して真似るのではなく、**聴いた音をそのまま重みに書き込み、演奏で思い出す**（[nn-e2e-plan.md](nn-e2e-plan.md) の 1.1、step6）。
楽譜を通らないので、揺らぎも消さずに持ち運べる。目指すのは「正確な機械の演奏」ではなく、**人の吹き方の癖まで含めて真似る演奏**。

### 1.2 その先：野鳥との会話

口笛の次は、野鳥のさえずりを聴いて**返事をする**。

- 録音して再生するのではなく、**演奏する**。相手の音程・タイミング・歌の型に合わせて、その場で音を作って返せる。
- 鳥どうしの歌い交わしも、決まった録音の応酬ではなく、相手に合わせて変えている（3 章）。演奏だからこそ、鳥にとって自然な返事になる可能性がある。
- 人の口笛の揺らぎを学んでいることが、ここでも効く。鳥は「その土地の人の口笛の癖」を覚えて応えることが分かっている（2.1）。

## 2. 人・機械と鳥のやりとりの研究例

2026-11-25 にウェブで調べた範囲。確かさの高い順。

### 2.1 人と野生の鳥の「会話」：ノドグロミツオシエ（Greater Honeyguide）

- アフリカの鳥で、蜂の巣まで人を案内する。モザンビークのヤオ族は「ブルルル・フム」という呼び声で呼ぶ（2016 年の研究で、この呼び声を使った探索の約 3 分の 2 で案内された）。
- タンザニアのハッザ族は、鳥の声に似た**メロディーのある口笛**で呼ぶ。
- 2023 年の *Science*（Spottiswoode & Wood, doi:10.1126/science.adh4129）の再生実験：それぞれの土地の鳥は**その土地の人の呼び声によく応える**。タンザニアではハッザの口笛にヤオの呼び声の 3 倍以上、モザンビークではヤオの呼び声にほぼ 2 倍。
- 人と野生の生き物が、互いに相手に向けた合図を交わす例として、ほぼ唯一とされる。
- **このロボットにとって**: 口笛で野生の鳥とやりとりが成り立つ実例。しかも鳥は人の口笛の「癖」を覚えている。

### 2.2 機械と鳥の閉ループ

| 例 | 内容 | このロボットにとって |
|---|---|---|
| カナリアとコンピューター（Lerch, Roy, Pachet, Nagle, *Animal Cognition* 14, 2011、Sony CSL パリ） | コンピューターがカナリアの声を聴いて分類し、録音した声で返す。実験者は手を出さない。関連特許（US 8838260）では、**同じ種類の声で返すとカナリアの声が増えてやりとりが続き、逆の種類で返すと鳴きやんだ** | 一番近い先行例。返事の中身が会話を続けるかを決める。ただし返事は録音の再生 |
| ゼブラフィンチのロボット（*J. Exp. Biol.* 225, 2022 ほか） | 止まり木のセンサーで鳥を感じ、声や動きを返すロボット。鳴き交わすロボットの研究もあるようだが、書誌情報は未確認 | 実験の道具としての先行例 |
| ナイチンゲールの音程合わせ（マックス・プランク研究所、2023） | 野生の個体が、**聞こえた口笛のような音の高さに自分の声をその場で合わせて**返す | 演奏で音程を合わせて返すロボットなら、この「合わせ」を相手にできる。ビジョンの核心を一番直接に支える |

### 2.3 「返事」の文法になる鳥の習性

- **ソングスパロー（ウタスズメ）の型合わせ**: 隣どうしで同じ歌の型を歌い返す。再生で型を合わせると長い歌い交わしが続いた（Nielsen & Vehrencamp 1995）。型合わせは強い挑発で、共有している別の歌で返すのは穏やかな返事（Beecher らの研究）。**何を返すかで意味が変わる**。
- **ナイチンゲールの重ねと交互**: 相手の歌に重ねて歌う個体と、交互に歌う個体がいる。**いつ返すかにも意味がある**（未検証の学会要旨）。

### 2.4 日本で対象にしやすい鳥

- **シジュウカラ**（鈴木俊貴さんの研究）: 「ピーツピ」＝警戒しろ、「ヂヂヂヂ」＝集まれ。つなげると「警戒して集まれ」、語順を逆にするとほとんど反応しない（2017）。2 つの声を **1 羽が続けて鳴いたか**まで聞き分ける。1 つの音源からだと反応し、2 つの音源から別々に流すと反応しない（2022）。意味のある返事を設計できる一番の候補。
- **ウグイス**: 口笛で真似ると鳴き返すという経験談はあるが、科学的に確かめた研究は見つからなかった。谷渡り鳴きを野外で再生しても雌は寄ってこなかったという報告がある（国立科学博物館）。ロードマップの段階 6 ではお手本の候補だが（[roadmap.md](roadmap.md)）、会話の相手として反応するかは自分たちで確かめる必要がある。

## 3. ビジョンとのつながり

- **人らしさ**: 鳥が反応するのは「その土地の人の口笛の癖」（2.1）。揺らぎを学ぶことに、具体的な意味がある。
- **演奏だから自然**: 相手の音程に合わせる（ナイチンゲール）、返す歌の型で意味を変える（ソングスパロー）、返すタイミングで意味を変える（ナイチンゲール）。録音の再生ではできない、その場で合わせる返事が鍵。
- **今の E2E 方針との関係**: 楽譜を外に持たず、聞いた音を重みに書き込んで思い出す仕組み（step6）は、揺らぎを持ち運ぶための前提。耳（step5）を鳥の声まで広げ、返事の作り方（何を・いつ返すか）を学ぶのが、口笛の真似の先の段になる。

## 4. 注意

- 繁殖期の野鳥に音を聞かせ続けると、ストレスを与え、縄張りや子育てを乱すおそれがある。野外で試すときは、時間と回数を絞る。
- 上の研究例は検索で集めたもので、原論文まで読んで確かめたのは一部だけ。引用するときは原論文を確認する。

## 出典

- [Science News: Humans, birds communicate to collaborate](https://www.sciencenews.org/article/humans-birds-communicate-collaborate)
- [Cambridge: human honeyguide communication](https://www.cam.ac.uk/stories/human-honeyguide-communication)
- [Audubon: Meet the Greater Honeyguide](https://www.audubon.org/magazine/meet-greater-honeyguide-bird-understands-humans)
- [Lerch et al. (Sony CSL)](https://csl.sony.fr/wp-content/uploads/2011/03/lerch-10a.pdf)
- [US 8838260 Animal-machine audio interaction system](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8838260)
- [Design of a robotic zebra finch for experimental studies (JEB)](https://cob.silverchair.com/jeb/article/225/3/jeb242949/274312/Design-of-a-robotic-zebra-finch-for-experimental)
- [Nightingale pitch matching (PMC10414052)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10414052)
- [MPG: Poetic birdsong, precisely tuned](https://www.mpg.de/20620153/poetic-birdsong-precisely-tuned)
- [How Nightingales develop their vocal competence (IOC Durban)](https://internationalornithology.org/PROCEEDINGS_Durban/Symposium/S03/S03.5.htm)
- [Song sparrow type matching (PMC3574360)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3574360)
- [Nielsen & Vehrencamp 1995](https://link.springer.com/doi/10.1007/BF00164156)
- [京都大学 2017: シジュウカラの文法](https://www.kyoto-u.ac.jp/ja/research-news/2017-07-28)
- [京都大学 2022: シジュウカラの発話者の認識](https://www.kyoto-u.ac.jp/research-news/2022-09-29-6)
- [BE-PAL: ウグイスの鳴きまね](https://www.bepal.net/archives/651153)
- [国立科学博物館 プレス資料](https://web.kahaku.go.jp/procedure/press/pdf/1475680.pdf)
