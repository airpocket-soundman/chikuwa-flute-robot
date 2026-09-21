"""Build the GitHub Pages report from the staged evaluation manifest."""
from __future__ import annotations

import argparse
import html
import json
import pathlib


def n(value, digits=1):
    return "—" if value is None else f"{value:.{digits}f}"


def audio(src, label):
    return f'<div class="audio"><span>{html.escape(label)}</span><audio controls preload="none" src="e2e-results/{html.escape(src)}"></audio></div>'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="docs/e2e-results/manifest.json")
    ap.add_argument("--out", default="docs/e2e-training-report.html")
    args = ap.parse_args(); data = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    s, examples = data["summary"], data["audio"]
    representative = next((x for x in examples if x["case"] == "step_up_down"), examples[0])

    stages = [
        ("1", "Neural Ear", "生の20 ms波形 → 音程・発音状態", s["ear"],
         representative["ear_before"], representative["ear_after"],
         "afterはNNが推定した音程を診断用音色で再合成。外部の音程解析器は推論入力に使わない。"),
        ("2", "Reference Memory", "Ear系列 → 保存メモリ → 音程系列", s["memory"],
         representative["memory_before"], representative["memory_after"],
         "raw音声を捨て、NNが生成したmemory tensorだけから復元した音。direct bypassは禁止。"),
        ("3", "Feed-forward Policy", "記憶したお手本 → PWM・バルブ → 物理シミュレータ", s["feedforward"],
         representative["ff_before"], representative["ff_after"],
         "afterだけがアクチュエータと笛のシミュレーションを通った実演奏相当。自己音は入力していない。"),
        ("4", "Error Comparator", "目標Ear + 自己音Ear → 符号付き誤差", s["comparator"],
         representative["comparator_before"], representative["comparator_after"],
         "beforeは意図的に外した自己音、afterは推定誤差を足し戻した診断用再合成。物理演奏ではない。"),
    ]
    cards = []
    for number, title, flow, metrics, before, after, note in stages:
        status = "PASS" if metrics.get("pass") else "FAIL"
        metric_lines = []
        for key, value in metrics.items():
            if key == "pass": continue
            metric_lines.append(f'<div><dt>{html.escape(key.replace("_", " "))}</dt><dd>{n(value, 3)}</dd></div>')
        cards.append(f'''<section class="card">
          <div class="stage"><b>{number}</b><span>{status}</span></div>
          <h2>{title}</h2><p class="flow">{flow}</p>
          <div class="listen">{audio(before, "NNの前")}{audio(after, "NNの後")}</div>
          <dl>{''.join(metric_lines)}</dl><p class="note">{note}</p>
        </section>''')
    details = []
    for stage, rows in data["details"].items():
        heads = [k for k in rows[0] if k != "case"]
        header = "".join(f"<th>{html.escape(k.replace('_', ' '))}</th>" for k in heads)
        body = "".join("<tr><th>" + html.escape(row["case"]) + "</th>" +
                       "".join(f"<td>{n(row[k], 3)}</td>" for k in heads) + "</tr>" for row in rows)
        details.append(f"<h3>{html.escape(stage)}</h3><div class='table'><table><thead><tr><th>case</th>{header}</tr></thead><tbody>{body}</tbody></table></div>")
    old = {"mae": 264.1299841855391, "corr": -0.995092265219766,
           "direction": 1.0, "explanation": "旧方向指標は300 ms以内の微小な符号一致で1.0となるため無効化"}
    feedback = data["feedback"]
    doc = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yamabiko E2E 分離NN 学習レポート</title>
<style>
:root{{--bg:#081018;--panel:#101d29;--ink:#edf6fb;--muted:#9eb2c0;--cyan:#4ee1d1;--red:#ff7b72;--line:#284052}}*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(circle at 80% 0,#163047 0,transparent 42%),var(--bg);color:var(--ink);font:15px/1.65 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:48px 20px 80px}}h1{{font-size:clamp(2rem,5vw,4.5rem);line-height:1.02;margin:.2em 0}}h2{{margin:.2rem 0}}h3{{margin-top:2rem}}.eyebrow,.flow{{color:var(--cyan);letter-spacing:.05em}}.lead{{font-size:1.15rem;max-width:820px;color:#c7d6df}}
.verdict{{border:1px solid var(--red);background:#28171a;padding:18px 22px;border-radius:14px;margin:28px 0}}.verdict b{{color:var(--red)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}}.card{{position:relative;background:linear-gradient(145deg,#132433,#0d1822);border:1px solid var(--line);border-radius:18px;padding:22px}}
.stage{{display:flex;justify-content:space-between;align-items:center}}.stage b{{display:grid;place-items:center;width:36px;height:36px;border-radius:50%;background:var(--cyan);color:#071016}}.stage span{{font-weight:800;color:var(--red)}}.card:has(.stage span:first-child){{border-color:var(--cyan)}}
.listen{{display:grid;gap:10px;margin:18px 0}}.audio{{display:grid;grid-template-columns:64px 1fr;align-items:center;gap:8px}}audio{{width:100%;height:36px}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}dl div{{background:#0a141d;padding:9px;border-radius:8px}}dt{{font-size:.72rem;color:var(--muted)}}dd{{margin:0;font-size:1.05rem}}.note{{color:var(--muted);font-size:.9rem}}
.pipeline{{display:flex;flex-wrap:wrap;gap:8px;margin:24px 0}}.pipeline span{{border:1px solid var(--line);padding:8px 12px;border-radius:999px}}.pipeline i{{color:var(--cyan);font-style:normal;padding:8px 0}}
.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.82rem}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}code{{color:var(--cyan)}}
footer{{color:var(--muted);margin-top:50px;border-top:1px solid var(--line);padding-top:18px}}@media(max-width:600px){{main{{padding-top:28px}}dl{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="eyebrow">PHYSICAL AI / OBJECTIVE GATED DEVELOPMENT</p><h1>お手本は、<br>本当に演奏になったか。</h1>
<p class="lead">単一の総合スコアで隠さず、聴覚・記憶・フィードフォワード演奏・誤差判定を独立したNNに分け、未知の固定課題で評価した。各カードの音声は同じ <code>step_up_down</code> 課題の「NN前 / NN後」である。</p>
<div class="verdict"><b>現在の総合判定: 未完成</b> — 全段がPASSするまでE2E成功とは呼ばない。Feedback Residualは trained={str(feedback['trained']).lower()} / pass={str(feedback['pass']).lower()}（{html.escape(feedback['reason'])}）。</div>
<div class="pipeline"><span>raw reference</span><i>→</i><span>Neural Ear</span><i>→</i><span>Reference Memory</span><i>→</i><span>FF Policy</span><i>→</i><span>plant + self audio</span><i>→</i><span>Error Comparator</span><i>→</i><span>Feedback Residual</span></div>
<div class="grid">{''.join(cards)}</div>
<h2>評価を厳格化した理由</h2><p>旧一体モデルはMAE {old['mae']:.1f} cent、軌跡相関 {old['corr']:.3f} で実際には逆方向へ追従した。一方、旧方向指標だけは {old['direction']:.1f} だった。{old['explanation']}。新評価は発音できない目標区間へ1200 cent罰を与え、P90、発音率、休符漏れ、遷移ゲイン、整定誤差をケース別に残す。</p>
<h2>学習の分離と再統合</h2><p>前段が合格したら凍結して次段を学習する。FFは名目rig・自己音なしから開始し、同じ参照でもrigごとに異なるOracle操作を回帰する不可能問題を避けた。次に閉ループDAggerでFeedback Residualを学習し、最後に未知rigと複数takeのRig Adapterを追加する。最終配備時は各NNを一つのforward graphとcheckpointへ束ねられるため、分離評価とE2E推論は両立する。</p>
<h2>全ケース詳細</h2>{''.join(details)}
<footer>seed {data['seed']} / checkpoint {html.escape(data['checkpoint'])}<br>{html.escape(data['sonification_note'])}</footer>
</main></body></html>'''
    path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(doc, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__": main()
