"""PC: make the step1 overview video (about 3 minutes) from media/step1 and the demo records.

    python scripts/make_step1_video.py --out media/step1/step1_overview.mp4

Slides are drawn with Pillow / Matplotlib (Japanese in Yu Gothic), the narration is Microsoft's neural voice
ja-JP-NanamiNeural through edge-tts (the narration text goes to Microsoft's speech service), and ffmpeg from
imageio-ffmpeg cuts and joins everything.  The three demo videos keep their own sound (the flute); the graphs
come from the whistle-copy records of the demos (media/step1/records/, copied from the UNO Q).
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
FONT_B = "C:/Windows/Fonts/YuGothB.ttc"
FONT_M = "C:/Windows/Fonts/YuGothM.ttc"
VOICE = "ja-JP-NanamiNeural"
BG, FG, ACC, MUTE, CARD = (246, 247, 249), (29, 35, 48), (37, 99, 235), (107, 114, 128), (255, 255, 255)

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
        d.rectangle([0, 0, W, 150], fill=CARD)
        d.rectangle([0, 146, W, 150], fill=ACC)
        d.text((80, 38), title, font=font(60, True), fill=FG)
        if subtitle:
            d.text((80 + d.textlength(title, font=font(60, True)) + 40, 62), subtitle, font=font(34), fill=MUTE)
    return im, d


def photo(path, box):
    """The photo cropped to fill box (x0, y0, x1, y1)."""
    from PIL import Image, ImageOps
    im = Image.open(path).convert("RGB")
    im = ImageOps.exif_transpose(im)
    w, h = box[2] - box[0], box[3] - box[1]
    return ImageOps.fit(im, (w, h), Image.LANCZOS)


def label(d, xy, text, size=36):
    f = font(size, True)
    x, y = xy
    tw = d.textlength(text, font=f)
    d.rounded_rectangle([x, y, x + tw + 36, y + size + 26], 12, fill=(17, 24, 39))
    d.text((x + 18, y + 10), text, font=f, fill=(255, 255, 255))


def bullets(d, x, y, items, size=40, gap=26, width=1700):
    f = font(size)
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
            d.text((cx, y), rest, font=f, fill=FG)
        else:
            d.text((cx, y), head, font=f, fill=FG)
        y += size + gap
    return y


def box(d, rect, text, fill=CARD, edge=ACC, size=34, sub=None):
    d.rounded_rectangle(rect, 18, fill=fill, outline=edge, width=4)
    x0, y0, x1, y1 = rect
    f = font(size, True)
    lines = text.split("\n")
    total = len(lines) * (size + 10) + (30 if sub else 0)
    y = (y0 + y1) / 2 - total / 2
    for line in lines:
        d.text(((x0 + x1) / 2 - d.textlength(line, font=f) / 2, y), line, font=f, fill=FG)
        y += size + 10
    if sub:
        fs = font(24)
        d.text(((x0 + x1) / 2 - d.textlength(sub, font=fs) / 2, y + 4), sub, font=fs, fill=MUTE)


def arrow(d, a, b, color=ACC, width=8):
    import math
    d.line([a, b], fill=color, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    s = 26
    p1 = (b[0] - s * math.cos(ang - .45), b[1] - s * math.sin(ang - .45))
    p2 = (b[0] - s * math.cos(ang + .45), b[1] - s * math.sin(ang + .45))
    d.polygon([b, p1, p2], fill=color)


def graph(stamp, out, title):
    """Whistle heard (top) and the flute's copy (bottom) for one demo record."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("C:/Windows/Fonts/YuGothM.ttc")
    plt.rcParams["font.family"] = font_manager.FontProperties(fname="C:/Windows/Fonts/YuGothM.ttc").get_name()
    t = np.load(RECORDS / f"{stamp}_target.npz"); p = np.load(RECORDS / f"{stamp}_play.npz")
    tc, tv, wc = t["cents"], t["voice"], t["whistle_cents"]
    h, v = p["0/heard"][0], p["0/valid"][0]
    shift = float(p["shift_cents"]) if "shift_cents" in p.files else 0.0
    m = v & tv
    key = float(np.median((h - shift - tc)[m]))
    hz = lambda c: 440 * 2 ** (np.asarray(c) / 1200)
    lead = int(np.flatnonzero(tv)[0])                       # the whistle starts where the target's notes do
    fig, axes = plt.subplots(2, 1, figsize=(16, 7.6), sharex=True)
    tw = (np.arange(len(wc)) + 0) * .01
    first = np.flatnonzero(np.isfinite(wc))[0]
    axes[0].plot(tw - tw[first] + lead * .01, hz(wc), ".", ms=4, color="#6b7280")
    axes[0].set_ylabel("口笛 [Hz]", fontsize=15); axes[0].set_title("聞いた音（口笛）", fontsize=18, loc="left")
    tt = np.arange(len(tc)) * .01
    axes[1].plot(tt, np.where(tv, hz(tc + key), np.nan), color="#111827", lw=3, label="目標（口笛を笛の音域へ移したもの）")
    axes[1].plot(tt[:len(h)][v], hz(h[v] - shift), ".", ms=5, color="#ea580c", label="笛が鳴らした音")
    axes[1].set_ylabel("笛 [Hz]", fontsize=15); axes[1].set_xlabel("時間 [秒]", fontsize=15)
    axes[1].set_title("再生した音（笛）", fontsize=18, loc="left"); axes[1].legend(fontsize=13, loc="upper right")
    for ax in axes:
        ax.grid(alpha=.3); ax.tick_params(labelsize=13)
    fig.suptitle(title, fontsize=22, x=.06, ha="left")
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close(fig)


# ----- narration -------------------------------------------------------------------------------------------
async def _say(text, path):
    import edge_tts
    await edge_tts.Communicate(text, VOICE, rate="+14%").save(str(path))


def say(text, path):
    asyncio.run(_say(text, path))


# ----- the video -------------------------------------------------------------------------------------------
def slides(tmp):
    """[(png, narration or None)] in order; demo segments are marked with ("DEMO", index)."""
    out = []
    # 1 title
    im, d = canvas()
    im.paste(photo(MEDIA / "装置全景.jpg", (0, 0, W, H)), (0, 0))
    d.rectangle([0, 700, W, H], fill=(17, 24, 39))
    d.text((90, 740), "やまびこ1号　Step1", font=font(96, True), fill=(255, 255, 255))
    d.text((94, 880), "口笛を聞いて、AI が笛でまねて吹く", font=font(52), fill=(209, 213, 219))
    im.save(tmp / "s01.png")
    out.append((tmp / "s01.png", "やまびこ1号は、管の笛をAIが吹くロボットです。"
                "ステップ1では、口笛を聞き取り、AIが笛でまねて吹けるようになりました。"))
    # 2 hardware
    im, d = canvas("装置", "管の長さを変えて音程を作る")
    cells = [("笛部1.jpg", "青い管とプランジャー（音程）"), ("リニアアクチュエータ.jpg", "リニアアクチュエーター"),
             ("ファン、タンギングサーボ、笛吹き込み部.jpg", "ファン・弁のサーボ・吹き込み部"),
             ("Arduinoと自作基板.jpg", "Arduino UNO Q と自作基板")]
    for k, (name, cap) in enumerate(cells):
        x0 = 60 + (k % 2) * 910; y0 = 190 + (k // 2) * 440
        im.paste(photo(MEDIA / name, (0, 0, 880, 410)), (x0, y0))
        label(d, (x0 + 16, y0 + 340), cap, 32)
    im.save(tmp / "s02.png")
    out.append((tmp / "s02.png", "青い管の中のプランジャーをリニアアクチュエーターで動かし、管の長さで音の高さを変えます。"
                "息はファンで送り、サーボの弁で止めます。"
                "音はマイクで聞き、アルドゥイーノ・ウノ・キューと自作基板で、1秒に100回制御します。"))
    # 3 how it works
    im, d = canvas("仕組み", "自分の音を聞きながら、NN が駆動信号を出す")
    box(d, (70, 260, 400, 440), "口笛", sub="マイクで録音")
    box(d, (500, 260, 880, 440), "耳（YIN）", sub="10 ms ごとに音程")
    box(d, (980, 260, 1400, 440), "メロディーを\n笛の音域へ", sub="音程の間隔はそのまま", size=32)
    box(d, (1020, 560, 1480, 800), "NN（再帰型）", fill=(219, 234, 254), sub="目標と、聞こえた自分の音から")
    box(d, (1560, 560, 1860, 800), "リニア\nアクチュエーター", sub="PWM 駆動信号")
    box(d, (1560, 880, 1860, 1040), "笛", sub="管の長さ → 音程")
    box(d, (500, 600, 880, 780), "耳（YIN）", sub="笛の音を聞く")
    arrow(d, (400, 350), (500, 350)); arrow(d, (880, 350), (980, 350)); arrow(d, (1190, 440), (1250, 560))
    arrow(d, (1480, 680), (1560, 680)); arrow(d, (1710, 800), (1710, 880))
    d.line([(1560, 960), (690, 960), (690, 780)], fill=(234, 88, 12), width=8); arrow(d, (690, 960), (690, 780), (234, 88, 12))
    arrow(d, (880, 690), (1020, 690), (234, 88, 12))
    d.text((760, 975), "フィードバック（自分の音を聞いて直す）", font=font(30, True), fill=(234, 88, 12))
    im.save(tmp / "s03.png")
    out.append((tmp / "s03.png", "マイクの音から10ミリ秒ごとに音の高さを読み、口笛のメロディーを、音程の間隔を保って笛の音域に移します。"
                "そしてニューラルネットワークが、目標と、聞こえた自分の音を比べながら、"
                "アクチュエーターの駆動信号を直接出します。音程を直すフィードバックも、NNの仕事です。"))
    # 4 training
    im, d = canvas("モデルと学習", "デジタルツインの中で学び、実機で動かす")
    box(d, (80, 300, 470, 520), "実機で較正", sub="動かして、聞いた音を記録")
    box(d, (560, 300, 990, 520), "デジタルツイン", fill=(219, 234, 254), sub="静止摩擦・管の長さ・速さを当てはめ")
    box(d, (1080, 300, 1480, 520), "シミュレーターで\n学習（GPU）", sub="少しずつ違う機体で")
    box(d, (1560, 300, 1860, 520), "UNO Q へ", sub="NumPy だけで推論")
    for a, b in (((470, 410), (560, 410)), ((990, 410), (1080, 410)), ((1480, 410), (1560, 410))):
        arrow(d, a, b)
    y = bullets(d, 110, 620, ["学習の目標|音程の誤差 ＋ 駆動信号の滑らかさ",
                              "実機の癖|動き出しには PWM 0.5 以上が要る（静止摩擦）、モーター音で笛が聞こえにくい",
                              "まず手本をまねてから|シミュレーターの中で直接最適化（勾配は実機には流さない）",
                              "推論|隠れ層 64 の GRU、1 ステップ約 1 ms（10 ms の周期に収まる）"], size=40, gap=40)
    im.save(tmp / "s04.png")
    out.append((tmp / "s04.png", "NNは、実機に合わせたデジタルツインの中で学びます。"
                "静止摩擦や管の長さを実機から当てはめ、音程の誤差と動きの滑らかさを同時に学ばせました。"
                "学んだNNは、UNO Qの上で、1回1ミリ秒ほどで動きます。"))
    # 5 results
    im, d = canvas("結果", "滑らかに動き、1 オクターブを吹ける")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_M)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_M).get_name()
    fig, ax = plt.subplots(figsize=(8, 5.6))
    names = ["従来の制御\n（PD＋観測器）", "NN の制御\n（Step1）"]
    bars = ax.bar(names, [303, 9], color=["#9ca3af", "#2563eb"], width=.55)
    for b, v in zip(bars, [303, 9]):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v} 回", ha="center", fontsize=20)
    ax.set_title("1 曲の中で駆動信号の向きが反転した回数", fontsize=17); ax.tick_params(labelsize=15)
    ax.set_ylim(0, 360); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(tmp / "bars.png", dpi=110); plt.close(fig)
    from PIL import Image
    im.paste(Image.open(tmp / "bars.png").resize((880, 616)), (60, 220))
    bullets(d, 1000, 260, ["細かい上下が|約 30 分の 1",
                           "音程が取れた割合|99 %（従来 40〜80 %）",
                           "ドレミの音の間隔の誤差|14 セント",
                           "ファン 80 % で|裏返らずに 1 オクターブ",
                           "　430〜880 Hz"], size=40, gap=34)
    im.save(tmp / "s05.png")
    out.append((tmp / "s05.png", "従来の制御は、細かく上下を繰り返していましたが、NNでは向きの反転が約30分の1になり、滑らかに動きます。"
                "ファンを80%にすると、1オクターブの音域を使えることも分かりました。"))
    # 6 demos with their graphs
    im, d = canvas()
    d.text((W / 2 - d.textlength("デモ　口笛 → 笛", font=font(110, True)) / 2, 420), "デモ　口笛 → 笛",
           font=font(110, True), fill=FG)
    msg = "口笛を吹くと、聞いたメロディーを笛でまねて吹きます"
    d.text((W / 2 - d.textlength(msg, font=font(44)) / 2, 600), msg, font=font(44), fill=MUTE)
    im.save(tmp / "s06.png")
    out.append((tmp / "s06.png", "では、デモです。"))
    from PIL import Image
    for k, (_, stamp, title) in enumerate(DEMOS):
        graph(stamp, tmp / f"g{k}.png", title)
    out.append(("DEMO", 0))
    im, d = canvas()
    g = Image.open(tmp / "g0.png"); g.thumbnail((W - 80, H - 60))
    im.paste(g, ((W - g.width) // 2, (H - g.height) // 2))
    im.save(tmp / "s07_0.png")
    out.append((tmp / "s07_0.png", "上が聞いた口笛、下が笛の音です。黒い目標を、オレンジの点がなめらかに追っています。"))
    out.append(("DEMO", 1))
    out.append(("DEMO", 2))
    im, d = canvas()
    for j, k in enumerate((1, 2)):
        g = Image.open(tmp / f"g{k}.png"); g.thumbnail((W - 60, H // 2 - 20))
        im.paste(g, ((W - g.width) // 2, 10 + j * (H // 2)))
    im.save(tmp / "s07_12.png")
    out.append((tmp / "s07_12.png", "うぐいすの声も、跳ぶ音の多いメロディーも、聞いた形を写して吹いています。"))
    # 7 next steps
    im, d = canvas("これから", "部品ごとに NN にして、最後に合体する")
    y = bullets(d, 100, 220, ["Step2　タンギング（弁）の NN|",
                              "　休符・同じ音の連打・大きな跳躍で、空気を止めるタイミングを NN が決める|",
                              "Step3　耳の NN|",
                              "　20 ms の生の波形から音程を聞く。YIN と同じ精度で、誤差は半分（1.9 セント）|",
                              "　モーター音の中でも聞き取れる割合　YIN 41 % → NN 73 %|",
                              "その後|耳・手・弁の NN を合わせて、全体を学び直す"], size=38, gap=30)
    im.save(tmp / "s08.png")
    out.append((tmp / "s08.png", "ステップ2では、空気を止めるタンギングを、ステップ3では、音を聞く耳を、それぞれNNにします。"
                "耳のNNは、モーター音の中でも、従来の倍近く音程を聞き取れました。"
                "部品ごとに完成させてから、最後に一つにまとめます。"))
    # 8 physical AI
    im, d = canvas("フィジカル AI としてのポイント")
    bullets(d, 100, 230, ["体の癖と付き合う|静止摩擦、空気の遅れ、ファンの唸り、モーターの音",
                          "自分の音を聞いて直す|閉ループ。音程の誤差を NN 自身が修正する",
                          "デジタルツインで学ぶ|実機は少し動かすだけ。シミュレーターで大量に練習",
                          "実機のデータで育てる|較正と演奏の記録でツインを直し、学び直す",
                          "小さな計算機でリアルタイム|UNO Q の上で 10 ms ごとに推論",
                          "部品ごとの NN|耳・手・弁を交換しながら確かめ、最後に合体"], size=40, gap=34)
    im.save(tmp / "s09.png")
    out.append((tmp / "s09.png", "フィジカルAIとしてのポイントは、癖のある体で、AIが自分の音を聞いてその場で直すこと。"
                "デジタルツインで学び、実機のデータで育て直すこと。"
                "そして小さな計算機の上で、リアルタイムに動くことです。"))
    # 9 end
    im, d = canvas()
    im.paste(photo(MEDIA / "笛部2.jpg", (0, 0, W, H)), (0, 0))
    d.rectangle([0, 760, W, H], fill=(17, 24, 39))
    d.text((90, 800), "やまびこ1号　Step1", font=font(80, True), fill=(255, 255, 255))
    d.text((94, 920), "次は Step2（タンギング）、Step3（耳の NN）へ", font=font(44), fill=(209, 213, 219))
    im.save(tmp / "s10.png")
    out.append((tmp / "s10.png", "やまびこ1号、ステップ1でした。"))
    return out


def caption_png(text, path):
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font(52, True)
    tw = d.textlength(text, font=f)
    d.rounded_rectangle([50, 50, 50 + tw + 60, 50 + 52 + 40], 14, fill=(17, 24, 39, 210))
    d.text((80, 68), text, font=f, fill=(255, 255, 255))
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(MEDIA / "step1_overview.mp4"))
    ap.add_argument("--keep", default=None, help="keep the work folder here")
    args = ap.parse_args()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="step1video_"))
    segs = []
    for k, item in enumerate(slides(tmp)):
        seg = tmp / f"seg{k:02d}.mp4"
        if item[0] == "DEMO":
            video, _, title = DEMOS[item[1]]
            caption_png(title, tmp / f"cap{item[1]}.png")
            run(["-i", MEDIA / video, "-i", tmp / f"cap{item[1]}.png", "-filter_complex",
                 f"[0:v]scale={W}:{H},fps=30,format=yuv420p[v];[v][1:v]overlay=0:0,format=yuv420p[o]",
                 "-map", "[o]", "-map", "0:a", "-c:v", "libx264", "-preset", "medium", "-crf", "27",
                 "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k", seg])
        else:
            png, text = item
            mp3 = tmp / f"n{k:02d}.mp3"
            say(text, mp3)
            dur = duration(mp3) + .6
            run(["-loop", "1", "-framerate", "30", "-i", png, "-i", mp3, "-filter_complex",
                 "[1:a]adelay=200|200,apad[a]", "-map", "0:v", "-map", "[a]", "-t", f"{dur:.2f}",
                 "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage", "-crf", "22", "-pix_fmt", "yuv420p",
                 "-r", "30", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k", seg])
        segs.append(seg)
        print(f"segment {k:2d}: {duration(seg):5.1f} s", flush=True)
    (tmp / "list.txt").write_text("".join(f"file '{s.as_posix()}'\n" for s in segs), encoding="utf-8")
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    run(["-f", "concat", "-safe", "0", "-i", tmp / "list.txt", "-c", "copy", "-movflags", "+faststart", out])
    print(f"saved {out}: {duration(out):.1f} s")
    if args.keep:
        shutil.copytree(tmp, args.keep, dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
