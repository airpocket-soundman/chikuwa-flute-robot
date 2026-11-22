"""PC: make the step1 overview video (about 3 minutes) from media/step1 and the demo records.

    python scripts/make_step1_video.py --out media/step1/step1_overview.mp4

Slides are drawn with Pillow / Matplotlib (Japanese in Yu Gothic).  The narration is Microsoft's neural voice
ja-JP-NanamiNeural through edge-tts (the narration text goes to Microsoft's speech service), one sentence at
a time so every subtitle lasts exactly as long as its sentence; a sentence can be shown and spoken
differently (UNO Q is spoken "ウノキュー").  ffmpeg from imageio-ffmpeg cuts and joins everything.  The three
demo videos keep their own sound (the flute); their graphs come from the whistle-copy records of the demos
(media/step1/records/, copied from the UNO Q).
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

MEDIA = pathlib.Path("media/step1")
RECORDS = MEDIA / "records"
W, H = 1920, 1080
SUB_TOP = 930                      # slides keep their content above this; the subtitles live below
FONT_B = "C:/Windows/Fonts/YuGothB.ttc"
FONT_M = "C:/Windows/Fonts/YuGothM.ttc"
VOICE = "ja-JP-NanamiNeural"
BG, FG, ACC, MUTE, CARD = (246, 247, 249), (29, 35, 48), (37, 99, 235), (107, 114, 128), (255, 255, 255)
ORANGE = (234, 88, 12)

DEMOS = [  # (video, record stamp, title)
    ("デモ1カエルの合唱.mp4", "20261122_112344", "デモ1　カエルの合唱"),
    ("デモ2うぐいす.mp4", "20261122_112557", "デモ2　うぐいす"),
    ("デモ3スーパーマリオ.mp4", "20261122_112626", "デモ3　スーパーマリオ"),
]


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args):
    subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


def duration(path):
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


# ----- drawing ------------------------------------------------------------------------------------------
def font(size, bold=False):
    from PIL import ImageFont
    return ImageFont.truetype(FONT_B if bold else FONT_M, size)


def canvas(title=None, subtitle=None):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    if title:
        d.rectangle([0, 0, W, 140], fill=CARD)
        d.rectangle([0, 136, W, 140], fill=ACC)
        d.text((80, 34), title, font=font(58, True), fill=FG)
        if subtitle:
            d.text((80 + d.textlength(title, font=font(58, True)) + 40, 58), subtitle, font=font(32), fill=MUTE)
    return im, d


def photo(path, size):
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    return ImageOps.fit(im, size, Image.LANCZOS)


def label(d, xy, text, size=32):
    f = font(size, True)
    x, y = xy
    tw = d.textlength(text, font=f)
    d.rounded_rectangle([x, y, x + tw + 36, y + size + 24], 12, fill=(17, 24, 39))
    d.text((x + 18, y + 9), text, font=f, fill=(255, 255, 255))


def bullets(d, x, y, items, size=40, gap=26):
    for item in items:
        head, _, rest = item.partition("|")
        if head.startswith("　"):                                     # a sub-item: indented, no dot
            d.text((x + 70, y), head.strip(), font=font(size - 4), fill=MUTE)
            y += size + gap
            continue
        d.ellipse([x, y + size * .45, x + 14, y + size * .45 + 14], fill=ACC)
        cx = x + 34
        if rest:
            fb = font(size, True)
            d.text((cx, y), head, font=fb, fill=FG)
            cx += d.textlength(head, font=fb) + 16
            d.text((cx, y), rest, font=font(size), fill=FG)
        else:
            d.text((cx, y), head, font=font(size), fill=FG)
        y += size + gap
    return y


def box(d, rect, text, fill=CARD, edge=ACC, size=32, sub=None, sub_size=24):
    d.rounded_rectangle(rect, 16, fill=fill, outline=edge, width=4)
    x0, y0, x1, y1 = rect
    f = font(size, True)
    lines = text.split("\n")
    subs = sub.split("\n") if sub else []
    total = len(lines) * (size + 8) + len(subs) * (sub_size + 6) + (6 if subs else 0)
    y = (y0 + y1) / 2 - total / 2
    for line in lines:
        d.text(((x0 + x1) / 2 - d.textlength(line, font=f) / 2, y), line, font=f, fill=FG)
        y += size + 8
    y += 6
    for line in subs:
        fs = font(sub_size)
        d.text(((x0 + x1) / 2 - d.textlength(line, font=fs) / 2, y), line, font=fs, fill=MUTE)
        y += sub_size + 6


def arrow(d, a, b, color=ACC, width=7, both=False):
    import math

    def head(p, q):
        ang = math.atan2(q[1] - p[1], q[0] - p[0])
        s = 24
        d.polygon([q, (q[0] - s * math.cos(ang - .45), q[1] - s * math.sin(ang - .45)),
                   (q[0] - s * math.cos(ang + .45), q[1] - s * math.sin(ang + .45))], fill=color)
    d.line([a, b], fill=color, width=width)
    head(a, b)
    if both:
        head(b, a)


def path_arrow(d, points, color, width=7):
    d.line(points, fill=color, width=width)
    arrow(d, points[-2], points[-1], color, width)


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font_manager.fontManager.addfont(FONT_M)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_M).get_name()
    return plt


def graph(stamp, out, title):
    """Whistle heard (top) and the flute's copy (bottom) for one demo record."""
    plt = setup_matplotlib()
    t = np.load(RECORDS / f"{stamp}_target.npz"); p = np.load(RECORDS / f"{stamp}_play.npz")
    tc, tv, wc = t["cents"], t["voice"], t["whistle_cents"]
    h, v = p["0/heard"][0], p["0/valid"][0]
    shift = float(p["shift_cents"]) if "shift_cents" in p.files else 0.0
    key = float(np.median((h - shift - tc)[v & tv]))
    hz = lambda c: 440 * 2 ** (np.asarray(c) / 1200)
    lead = int(np.flatnonzero(tv)[0])
    fig, axes = plt.subplots(2, 1, figsize=(16, 7.2), sharex=True)
    tw = np.arange(len(wc)) * .01
    first = np.flatnonzero(np.isfinite(wc))[0]
    axes[0].plot(tw - tw[first] + lead * .01, hz(wc), ".", ms=4, color="#6b7280")
    axes[0].set_ylabel("口笛 [Hz]", fontsize=15); axes[0].set_title("聞いた音（口笛）", fontsize=18, loc="left")
    tt = np.arange(len(tc)) * .01
    axes[1].plot(tt, np.where(tv, hz(tc + key), np.nan), color="#111827", lw=3, label="目標（口笛を笛の音域へ移したもの）")
    axes[1].plot(tt[:len(h)][v], hz(h[v] - shift), ".", ms=5, color="#ea580c", label="スライドホイッスルが鳴らした音")
    axes[1].set_ylabel("笛 [Hz]", fontsize=15); axes[1].set_xlabel("時間 [秒]", fontsize=15)
    axes[1].set_title("再生した音（スライドホイッスル）", fontsize=18, loc="left"); axes[1].legend(fontsize=13, loc="upper right")
    for ax in axes:
        ax.grid(alpha=.3); ax.tick_params(labelsize=13)
    fig.suptitle(title, fontsize=22, x=.06, ha="left")
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close(fig)


def fit_into(im, path, top=150, bottom=SUB_TOP - 10, side=40):
    from PIL import Image
    g = Image.open(path)
    g.thumbnail((W - 2 * side, bottom - top))
    im.paste(g, ((W - g.width) // 2, top + (bottom - top - g.height) // 2))


# ----- the slides and what is said over them --------------------------------------------------------------
# A line of narration is "shown|spoken" when the two differ (UNO Q is spoken ウノキュー), else one string.
def slides(tmp):
    from PIL import Image
    out = []
    # 1 title
    im, d = canvas()
    im.paste(photo(MEDIA / "装置全景.jpg", (W, SUB_TOP)), (0, 0))
    d.rectangle([0, 640, W, SUB_TOP], fill=(17, 24, 39))
    d.text((90, 670), "やまびこ1号　Step1", font=font(92, True), fill=(255, 255, 255))
    d.text((94, 810), "口笛を聞いて、AI がスライドホイッスルでまねて吹く", font=font(48), fill=(209, 213, 219))
    im.save(tmp / "s01.png")
    out.append((tmp / "s01.png", ["やまびこ1号は、スライドホイッスルをAIが吹くロボットです。",
                                  "ステップ1では、口笛を聞き取り、AIがスライドホイッスルでまねて吹けるようになりました。"]))
    # 2 the instrument and its parts
    im, d = canvas("装置", "スライドホイッスルを機械で吹く")
    cells = [("笛部1.jpg", "スライドホイッスル（管体とピストン）"), ("リニアアクチュエータ.jpg", "DC リニアアクチュエータ"),
             ("ファン、タンギングサーボ、笛吹き込み部.jpg", "遠心ファン・サーボ・切換え弁"),
             ("笛吹き込み部、マイク.jpg", "吹き口と MEMS マイク")]
    for k, (name, cap) in enumerate(cells):
        x0 = 60 + (k % 2) * 910; y0 = 160 + (k // 2) * 380
        im.paste(photo(MEDIA / name, (880, 355)), (x0, y0))
        label(d, (x0 + 16, y0 + 288), cap, 30)
    im.save(tmp / "s02.png")
    out.append((tmp / "s02.png", ["管体の中のピストンをDCリニアアクチュエータで動かし、管の長さで音の高さを変えます。",
                                  "息は遠心ファンで送り、サーボで駆動する自作の切換え弁で、タンギングを再現します。",
                                  "口笛と笛の音は、MEMSマイクで聞き取ります。|口笛と笛の音は、メムスマイクで聞き取ります。"]))
    # 3 hardware
    im, d = canvas("ハードウェア構成")
    d.rounded_rectangle((60, 190, 640, 900), 22, fill=(239, 246, 255), outline=ACC, width=4)
    d.text((90, 205), "Arduino UNO Q", font=font(40, True), fill=FG)
    box(d, (95, 280, 605, 520), "Linux 側（MPU）", sub="耳（音程の推定）・NN・口笛コピー\nPython / NumPy、ウェブ画面", size=32)
    box(d, (95, 640, 605, 870), "マイコン側（STM32）", sub="PWM・I2S 取り込み・サーボ信号\n10 ms 周期の指令・安全停止", size=32)
    arrow(d, (350, 520), (350, 640), both=True)
    d.text((370, 562), "シリアル通信", font=font(24), fill=MUTE)
    box(d, (760, 190, 1170, 900), "自作基板", sub="モータードライバ\n（TB6643）\n\nファン駆動（PWM）\n\nサーボ信号\n\nMEMS マイク基板", size=34, sub_size=27)
    arrow(d, (605, 755), (760, 755), both=True)
    box(d, (1290, 175, 1860, 335), "DC リニアアクチュエータ", sub="エンコーダなし → ピストン → 管体の長さ", size=30)
    box(d, (1290, 360, 1860, 520), "遠心ファン", sub="吹き口へ空気を送る", size=30)
    box(d, (1290, 545, 1860, 705), "サーボ → 自作の切換え弁", sub="タンギング（空気の切換え）", size=30)
    box(d, (1290, 730, 1860, 900), "MEMS マイク（I2S）", sub="口笛とスライドホイッスルの\n音程センシング", size=30)
    for y in (255, 440, 625):
        arrow(d, (1170, y), (1290, y))
    arrow(d, (1290, 815), (1170, 815), ORANGE)
    im.save(tmp / "s03.png")
    out.append((tmp / "s03.png", ["制御の中心は、UNO Qです。|制御の中心は、ウノキューです。",
                                  "Linux側でAIを動かし、マイコン側が自作基板を通して、アクチュエータ、ファン、サーボを駆動します。"
                                  "|リナックス側でAIを動かし、マイコン側が自作基板を通して、アクチュエータ、ファン、サーボを駆動します。",
                                  "MEMSマイクの音はI2Sでマイコンに入り、1秒に100回の周期で制御します。"
                                  "|メムスマイクの音はアイツーエスでマイコンに入り、1秒に100回の周期で制御します。"]))
    # 4 how it works
    im, d = canvas("仕組み", "自分の音を聞きながら、NN が駆動信号を出す")
    box(d, (60, 200, 360, 350), "口笛", sub="MEMS マイク")
    box(d, (450, 200, 800, 350), "耳", sub="10 ms ごとに音程を推定")
    box(d, (890, 200, 1290, 350), "メロディーを\n笛の音域へ", sub="音程の間隔はそのまま", size=30)
    box(d, (950, 450, 1390, 680), "NN（再帰型）", fill=(219, 234, 254), sub="目標と、聞こえた自分の音から\n駆動信号を直接出す")
    box(d, (1480, 450, 1860, 680), "DC リニア\nアクチュエータ", sub="エンコーダなし")
    box(d, (1480, 760, 1860, 910), "スライドホイッスル", sub="ピストンの位置 → 音程", size=30)
    box(d, (450, 480, 800, 650), "耳", sub="自分の音を聞く")
    arrow(d, (360, 275), (450, 275)); arrow(d, (800, 275), (890, 275)); arrow(d, (1110, 350), (1150, 450))
    arrow(d, (1390, 565), (1480, 565)); arrow(d, (1670, 680), (1670, 760))
    path_arrow(d, [(1480, 835), (625, 835), (625, 650)], ORANGE)
    arrow(d, (800, 565), (950, 565), ORANGE)
    d.text((660, 850), "音程でフィードバック（位置センサーの代わり）", font=font(30, True), fill=ORANGE)
    im.save(tmp / "s04.png")
    out.append((tmp / "s04.png", ["マイクの音から、10ミリ秒ごとに音の高さを読み取ります。",
                                  "口笛のメロディーは、音程の間隔を保ったまま、笛の音域に移します。",
                                  "そしてニューラルネットワークが、目標と、聞こえた自分の音を比べて、アクチュエータの駆動信号を直接出します。",
                                  "このアクチュエータには位置を測るエンコーダが無いため、聞こえた音程から位置を知り、"
                                  "音程によるフィードバック制御も同時に行っています。"]))
    # 5 model and training
    im, d = canvas("モデルと学習", "デジタルツインの中で学び、実機で動かす")
    box(d, (80, 220, 470, 420), "実機で較正", sub="動かして、聞いた音を記録")
    box(d, (560, 220, 990, 420), "デジタルツイン", fill=(219, 234, 254), sub="静止摩擦・管の長さ・速さを\n実機から当てはめ")
    box(d, (1080, 220, 1480, 420), "シミュレーターで\n学習（GPU）", sub="少しずつ違う機体で")
    box(d, (1560, 220, 1860, 420), "UNO Q へ", sub="NumPy だけで推論")
    for a, b in (((470, 320), (560, 320)), ((990, 320), (1080, 320)), ((1480, 320), (1560, 320))):
        arrow(d, a, b)
    bullets(d, 110, 520, ["学習の目標|音程の誤差 ＋ 駆動信号の滑らかさ",
                          "実機の癖|動き出しに PWM 0.5 以上が要る（静止摩擦）、モーター音で笛が聞こえにくい",
                          "まず手本をまねてから|シミュレーターの中で直接最適化",
                          "推論|隠れ層 64 の GRU、1 ステップ約 1 ms（10 ms の周期に収まる）"], size=38, gap=40)
    im.save(tmp / "s05.png")
    out.append((tmp / "s05.png", ["NNは、実機に合わせたデジタルツインの中で学びます。",
                                  "静止摩擦や管の長さを実機から当てはめ、音程の誤差と動きの滑らかさを同時に学ばせました。",
                                  "学んだNNは、UNO Qの上で、1回1ミリ秒ほどで動きます。|学んだNNは、ウノキューの上で、1回1ミリ秒ほどで動きます。"]))
    # 6 demos with their graphs
    im, d = canvas()
    text = "デモ　口笛 → スライドホイッスル"
    d.text((W / 2 - d.textlength(text, font=font(96, True)) / 2, 380), text, font=font(96, True), fill=FG)
    im.save(tmp / "s06.png")
    out.append((tmp / "s06.png", ["では、デモです。"]))
    for k, (_, stamp, title) in enumerate(DEMOS):
        graph(stamp, tmp / f"g{k}.png", title)
    out.append(("DEMO", 0))
    im, d = canvas()
    fit_into(im, tmp / "g0.png", top=20)
    im.save(tmp / "s07_0.png")
    out.append((tmp / "s07_0.png", ["上が聞いた口笛、下がスライドホイッスルの音です。", "黒い目標を、オレンジの点がなめらかに追っています。"]))
    out.append(("DEMO", 1))
    out.append(("DEMO", 2))
    im, d = canvas()
    half = (SUB_TOP - 20) // 2
    for j, k in enumerate((1, 2)):
        g = Image.open(tmp / f"g{k}.png"); g.thumbnail((W - 60, half - 10))
        im.paste(g, ((W - g.width) // 2, 10 + j * half))
    im.save(tmp / "s07_12.png")
    out.append((tmp / "s07_12.png", ["うぐいすの声も、跳ぶ音の多いメロディーも、聞いた形を写して吹いています。"]))
    # 7 next steps
    im, d = canvas("これから", "部品ごとに NN にして、最後に合体する")
    bullets(d, 100, 200, ["Step2　タンギングの NN|",
                          "　休符・同じ音の連打・大きな跳躍で、切換え弁を動かすタイミングを NN が決める",
                          "Step3　耳の NN|",
                          "　20 ms の生の波形から音程を推定。誤差は従来の半分（1.9 セント）",
                          "　モーター音の中でも聞き取れる割合　従来 41 % → NN 73 %",
                          "その後|耳・アクチュエータ・タンギングの NN を合わせて、全体を学び直す"], size=38, gap=32)
    im.save(tmp / "s08.png")
    out.append((tmp / "s08.png", ["ステップ2では、タンギングを、ステップ3では、音を聞く耳を、それぞれNNにします。",
                                  "部品ごとに完成させてから、最後に一つにまとめます。"]))
    # 8 physical AI
    im, d = canvas("フィジカル AI としてのポイント")
    bullets(d, 100, 200, ["体の癖と付き合う|静止摩擦、空気の遅れ、ファンの唸り、モーターの音",
                          "音程を位置センサーにする|エンコーダなしで、聞こえた音から位置を知る",
                          "自分の音を聞いて直す|閉ループ。音程の誤差を NN 自身が修正する",
                          "デジタルツインで学ぶ|実機は少し動かすだけ。シミュレーターで大量に練習",
                          "実機のデータで育てる|較正と演奏の記録でツインを直し、学び直す",
                          "小さな計算機でリアルタイム|UNO Q の上で 10 ms ごとに推論"], size=40, gap=34)
    im.save(tmp / "s09.png")
    out.append((tmp / "s09.png", ["フィジカルAIとしてのポイントは、癖のある体で、AIが自分の音を聞いて、その場で直すこと。",
                                  "デジタルツインで学び、実機のデータで育て直すこと。",
                                  "そして小さな計算機の上で、リアルタイムに動くことです。"]))
    # 9 end
    im, d = canvas()
    im.paste(photo(MEDIA / "笛部2.jpg", (W, SUB_TOP)), (0, 0))
    d.rectangle([0, 700, W, SUB_TOP], fill=(17, 24, 39))
    d.text((90, 730), "やまびこ1号　Step1", font=font(78, True), fill=(255, 255, 255))
    d.text((94, 850), "次は Step2（タンギング）、Step3（耳の NN）へ", font=font(42), fill=(209, 213, 219))
    im.save(tmp / "s10.png")
    out.append((tmp / "s10.png", ["やまびこ1号、ステップ1でした。"]))
    return out


DEMO_SUBTITLES = ["口笛を聞いたあと、スライドホイッスルでまねて吹きます（カエルの合唱）",
                  "うぐいすの鳴き声を口笛で吹いて、まねさせます",
                  "スーパーマリオのメロディーを、口笛からまねて吹きます"]


def with_subtitle(png, text, out):
    """The slide with a subtitle band at the bottom (wrapped to two lines at most)."""
    from PIL import Image, ImageDraw
    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    subtitle_band(d, text)
    im.save(out)


def subtitle_band(d, text):
    """White subtitles on a dark band below SUB_TOP: one line, or two split at the comma nearest the middle."""
    f = font(42, True)
    lines = [text]
    if d.textlength(text, font=f) > W - 140:
        cuts = [i + 1 for i, ch in enumerate(text[:-1]) if ch in "、，"]
        mid = len(text) / 2
        cut = min(cuts, key=lambda i: abs(i - mid)) if cuts else int(mid)
        lines = [text[:cut], text[cut:]]
    d.rectangle([0, SUB_TOP, W, H], fill=(17, 24, 39))
    y = SUB_TOP + (H - SUB_TOP - len(lines) * 56) / 2
    for ln in lines:
        d.text((W / 2 - d.textlength(ln, font=f) / 2, y), ln, font=f, fill=(255, 255, 255))
        y += 56


def caption_png(title, subtitle, path):
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font(52, True)
    tw = d.textlength(title, font=f)
    d.rounded_rectangle([50, 50, 50 + tw + 60, 50 + 52 + 40], 14, fill=(17, 24, 39, 210))
    d.text((80, 68), title, font=f, fill=(255, 255, 255))
    fs = font(44, True)
    sw = d.textlength(subtitle, font=fs)
    d.rectangle([0, H - 110, W, H], fill=(17, 24, 39, 200))
    d.text((W / 2 - sw / 2, H - 88), subtitle, font=fs, fill=(255, 255, 255))
    im.save(path)


# ----- narration -------------------------------------------------------------------------------------------
async def _say(text, path):
    import edge_tts
    await edge_tts.Communicate(text, VOICE, rate="+20%").save(str(path))


def say(text, path):
    asyncio.run(_say(text, path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(MEDIA / "step1_overview.mp4"))
    ap.add_argument("--keep", default=None, help="keep the work folder here")
    args = ap.parse_args()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="step1video_"))
    segs = []
    venc = ["-c:v", "libx264", "-preset", "medium", "-crf", "27", "-pix_fmt", "yuv420p", "-r", "30"]
    aenc = ["-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k"]
    n = 0
    for item in slides(tmp):
        if item[0] == "DEMO":
            k = item[1]
            video, _, title = DEMOS[k]
            caption_png(title, DEMO_SUBTITLES[k], tmp / f"cap{k}.png")
            seg = tmp / f"seg{n:03d}.mp4"; n += 1
            run(["-i", MEDIA / video, "-i", tmp / f"cap{k}.png", "-filter_complex",
                 f"[0:v]scale={W}:{H},fps=30,format=yuv420p[v];[v][1:v]overlay=0:0,format=yuv420p[o]",
                 "-map", "[o]", "-map", "0:a", *venc, *aenc, seg])
            segs.append(seg)
            continue
        png, lines = item
        for j, line in enumerate(lines):
            shown, _, spoken = line.partition("|")
            spoken = spoken or shown
            mp3 = tmp / f"n{n:03d}.mp3"
            say(spoken, mp3)
            frame = tmp / f"f{n:03d}.png"
            with_subtitle(png, shown, frame)
            lead = .25 if j == 0 else .05
            dur = lead + duration(mp3) + (.45 if j == len(lines) - 1 else .15)
            ms = int(lead * 1000)
            seg = tmp / f"seg{n:03d}.mp4"; n += 1
            run(["-loop", "1", "-framerate", "30", "-i", frame, "-i", mp3, "-filter_complex",
                 f"[1:a]adelay={ms}|{ms},apad[a]", "-map", "0:v", "-map", "[a]", "-t", f"{dur:.2f}",
                 *venc, "-tune", "stillimage", *aenc, seg])
            segs.append(seg)
        print(f"{png.name}: {len(lines)} lines", flush=True)
    (tmp / "list.txt").write_text("".join(f"file '{s.as_posix()}'\n" for s in segs), encoding="utf-8")
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    run(["-f", "concat", "-safe", "0", "-i", tmp / "list.txt", "-c", "copy", "-movflags", "+faststart", out])
    print(f"saved {out}: {duration(out):.1f} s")
    if args.keep:
        shutil.copytree(tmp, args.keep, dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
