# ProtoPedia公開資料

公開記事: https://protopedia.net/prototype/10569 （2026-11-19）

- `article.md`: 記事本文。基板BoMは`board-bom.md`を末尾の折りたたみ欄に掲載。
- `make_assets.py`: 説明図のSVG/PNGと、基板設計データからBoMを生成するスクリプト。PythonとPillow、WindowsのMeiryoフォントを使用。
- `yamabiko-overview.svg` / `.png`: 装置の概略図。
- `yamabiko-control-results.svg` / `.png`: 制御方式とシミュレーション成果。
- `yamabiko-a-pcb.png` / `yamabiko-b-pcb.png`: KiCad PCBデータから出力した3Dレンダリング。一部部品の3Dモデルは省略。実物のヘッダーの実装方向については記事を参照。

説明図とBoMの再生成:

```powershell
python docs/protopedia/make_assets.py
```

基板画像の再生成（A基板。B基板は名前のaをbに変更）:

```powershell
& 'C:/Program Files/KiCad/10.0/bin/kicad-cli.exe' pcb render --width 1600 --height 1000 --background opaque --quality high --rotate '0,0,0' --output docs/protopedia/yamabiko-a-pcb.png hardware/yamabiko_a/yamabiko_a.kicad_pcb
```

記事の進捗は基板製作・部品購入完了、これから組み立て。掲載した演奏評価の数値はシミュレーションの結果であり、実機では未検証。
