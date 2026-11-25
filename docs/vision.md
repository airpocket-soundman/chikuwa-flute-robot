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
- 鳥どうしの歌い交わしも、決まった録音の応酬ではなく、相手に合わせて変えている（2.2・2.3）。演奏だからこそ、鳥にとって自然な返事になる可能性がある。
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

## 4. 目標：野鳥の言葉・人の口笛言語・LLM をつなぐ（2026-11-25）

```
人の口笛言語（シルボ・ゴメロ）──┐                      ┌── 口笛言語の音程の動き ──┐
野鳥の声（シジュウカラ）────────┼─ 耳 ─→ LLM（意味）─→ ┤                          ├─ 体に合わせて圧縮 → 笛
人の話し言葉 ────────────────────┘                      └── 鳥の語彙からの返事 ────┘
```

- **合成音でも成立してしまう問題**: 言葉のやりとりだけなら、スピーカーで合成音を鳴らすほうが楽で正確。鳥の再生実験もほぼすべてスピーカーで行われている。
  だから体の役目を「会話ができること」には置かない。
- **体の役目**: 口笛言語は、話し言葉を人の口という狭い体に押し込めたもの（母音は音の高さ、子音は音程の動き方）。
  やまびこ1号も狭い体（1 オクターブ、端から端まで約 1.4 秒、静止摩擦）。LLM の言いたいことを**この体で出せる動きに自分で圧縮し直す**のが、合成音にはない部分。
  体が違えば吹き方（方言）も変わるはず。
- **見せ方**: 制約のない合成音と、この体で吹いた音を並べ、口笛言語の話者や鳥に聞かせて、それでも通じるかを比べる。
- **LLM の主張の範囲**: 鳥の「言葉」は語彙が少なく、意味もほとんどが警戒や集合。「LLM が鳥と会話する」とは言わず、「分かっている語彙の範囲で、意味の通る返事を選ぶ」まで。
- **最初の検証**: アーカイブの口笛言語と鳥の声を、今の体（ツインの上）で吹き直し、音程の形がどこまで残るかを見る。デモの手順と結果は [archive-demo.md](archive-demo.md)。

## 5. アート・インスタレーションとしての可能性

### 5.1 近い先行作品

| 作品 | 内容 | この作品との違い |
|---|---|---|
| Alexandra Daisy Ginsberg《Machine Auguries》（2019〜、ロンドン・トレド・ウメオ） | 夜明けの鳥のコーラスに、AI が作った鳥の声が少しずつ入り込み、置き換えていく音と光のインスタレーション。鳥の減少への問いかけ | 声はスピーカーから出る合成音で、生きた鳥とのやりとりではない |
| Tully Arnot《Bird Song》（APT11） | コードで作った鳥の声を、電子回路と空気ポンプで機械的に鳴らす | 機械の声だが、聞いて返すことはしない |
| David Rothenberg（Nightingales in Berlin、Nightingale Cities） | クラリネット奏者がベルリンの公園でナイチンゲールと即興演奏する。アルバム・映画・本になった。「どの種も一種だけでは作れない音楽」 | 人が演奏する。縄張りに自信のある雄は、聴いて、間を置いて、短く返すという |
| Céleste Boursier-Mougenot《from here to ear》 | ギターの弦にとまるキンカチョウが音を鳴らすインスタレーション | 生きた鳥が演奏者 |
| Yapci Ramos《Freedom》（2019）、《yo, Guirre》（2026、ラ・ゴメラ） | 鳥の録音と作家の口笛のぶつかり合い（火山石の 2 チャンネル音響作品）。後者はシルボ・ゴメロと羊飼いの杖を扱う映像・音響作品 | 口笛は人が吹く |
| 《Stuck》（2014、リガ、SKAN II） | 口笛言語の話者 Kico Correa が、ラトビア語とロシア語を口笛言語に訳そうとした音響作品 | 翻訳は人がする |
| 18 世紀の自動さえずり鳥の箱（メトロポリタン美術館所蔵ほか） | 箱から機械の鳥が出て、首を振り羽ばたきながら鳴く | 機械の鳥の系譜。決まった歌を鳴らすだけ |

### 5.2 この作品ならではの案

1. **夜明けのコーラスに加わる笛**: 《Machine Auguries》は合成音が鳥の声を「置き換える」。こちらは本物の笛の音で、鳥の声に「加わり、返す」。屋外の展示。
2. **聴いて、間を置いて、返す演奏者**: Rothenberg のセッションの演奏者を機械にする。ナイチンゲールが相手の音程に合わせて返す習性（2.2）と組み合わせる。
3. **翻訳する笛**: 来場者が話す → LLM → 口笛言語 → 笛が吹く。来場者が口笛で返すと、笛が聴いて訳す。《Stuck》の翻訳を機械が担う。
4. **方言が生まれる群れ**: 管の長さ・太さ・素材の違う笛を何台も並べ、互いに聴いて真似し合う。体ごとに出せる動きが違うので、同じ言葉でも台ごとに吹き方（方言）が分かれていく。
   ミツオシエが土地ごとに違う人の呼び声に応えること（2.1）の、機械版。
5. **自動さえずり鳥の箱の現代版**: 決まった歌を鳴らす 18 世紀の箱に対して、聴いて覚え、練習で上達し、笛を替えても合わせ直す箱。

## 6. 注意

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
- [artnet: Machine Auguries: Toledo](https://news.artnet.com/art-world/alexandra-daisy-ginsberg-machine-auguries-ai-toledo-museum-2300348)
- [FACT: Machine Auguries (2019)](https://www.fact.co.uk/artwork/machine-auguries-2019)
- [QAGOMA: Tully Arnot (APT11)](https://www.qagoma.qld.gov.au/exhibition/the-11th-asia-pacific-triennial-of-contemporary-art-apt11/tully-arnot)
- [UChicago Press: Nightingales in Berlin](https://pressblog.uchicago.edu/2019/05/23/read-an-excerpt-from-nightingales-in-berlin-searching-for-the-perfect-sound.html)
- [DailyGood: David Rothenberg, interspecies music](https://www.dailygood.org/story/3048/david-rothenberg-the-joy-and-mystery-of-interspecies-music-making-david-rothenberg/)
- [Bonart: Yapci Ramos, yo, Guirre](https://www.bonart.cat/en/n/47229/yapci-ramos-revives-the-flight-of-the-guirre-through-contemporary-art)
- [musikprotokoll: Silbadores / Stuck](https://musikprotokoll.orf.at/en/blog/silbadores/iestredziszastryal-stuck)
- [The Met: Mechanical singing bird box](https://www.metmuseum.org/en/art/collection/search/910012?pkgids=913)
- [OpenSLR SLR137 Silbo Gomero Speech Corpus](https://openslr.org/137)
- [xeno-canto: Japanese Tit](https://xeno-canto.org/species/Parus-minor)
