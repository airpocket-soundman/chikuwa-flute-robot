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


def beat_audio(src, label):
    return f'<div class="audio"><span>{html.escape(label)}</span><audio controls preload="none" src="e2e-beat-results/{html.escape(src)}"></audio></div>'


def doc_audio(src, label):
    return f'<div class="audio"><span>{html.escape(label)}</span><audio controls preload="none" src="{html.escape(src)}"></audio></div>'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="docs/e2e-results/manifest.json")
    ap.add_argument("--out", default="docs/e2e-training-report.html")
    args = ap.parse_args(); data = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    s, examples = data["summary"], data["audio"]
    representative = next((x for x in examples if x["case"] == "step_up_down"), examples[0])
    beat_path = pathlib.Path(args.manifest).parent.parent / "e2e-beat-results" / "manifest.json"
    beat_section = "<p>Tempo/Beat Gateは未実行です。</p>"
    if beat_path.exists():
        beat = json.loads(beat_path.read_text(encoding="utf-8")); bm = beat["tempo_beat"]
        bc = next((x for x in beat["cases"] if x["case"] == "bpm_137"), beat["cases"][0])
        status = "PASS" if bm.get("pass") else "FAIL"
        beat_section = f'''<div class="grid"><section class="card"><div class="stage"><b>B</b><span>{status}</span></div>
        <h2>Tempo / Beat NN</h2><p class="flow">raw reference → BPM・連続beat phase</p>
        <div class="listen">{beat_audio(bc['tempo_before'], 'お手本')}{beat_audio(bc['tempo_after'], '予測拍click')}</div>
        <dl><div><dt>tempo median relative error</dt><dd>{n(bm['tempo_relative_error_median'], 4)}</dd></div>
        <div><dt>phase MAE cycle</dt><dd>{n(bm['phase_circular_mae_cycle'], 4)}</dd></div></dl>
        <p class="note">afterはTempo NNの拍だけを鳴らした診断clickで、演奏音ではない。official path: {html.escape(beat['official_path'])}</p>
        </section>'''
        if beat.get("musical_memory_trained") and beat.get("musical_memory"):
            mm = beat["musical_memory"]; ms = "PASS" if mm.get("pass") else "FAIL"
            beat_section += f'''<section class="card"><div class="stage"><b>M</b><span>{ms}</span></div>
            <h2>Musical Memory NN</h2><p class="flow">予測beat + 音響表現 → 音程・休符・onset/offsetセル</p>
            <div class="listen">{beat_audio(bc['memory_before'], 'NNの前')}{beat_audio(bc['memory_after'], '記憶から再合成')}</div>
            <dl><div><dt>pitch MAE cents</dt><dd>{n(mm['pitch_mae_cents'])}</dd></div>
            <div><dt>voice F1</dt><dd>{n(mm['voice_f1'], 3)}</dd></div><div><dt>onset F1</dt><dd>{n(mm['onset_f1'], 3)}</dd></div>
            <div><dt>correlation</dt><dd>{n(mm['trajectory_correlation'], 3)}</dd></div></dl>
            <p class="note">raw音声と正解セル数を捨て、予測BPM・予測位相から求めたセル数で再生した診断音。</p></section>'''
        if beat.get("temporal_aligner_trained") and beat.get("temporal_aligner"):
            am = beat["temporal_aligner"]; ast = "PASS" if am.get("pass") else "FAIL"
            beat_section += f'''<section class="card"><div class="stage"><b>A</b><span>{ast}</span></div>
            <h2>Neural Clock / Temporal Aligner</h2><p class="flow">記憶セル + BPM + 100 Hz tick → 時刻付き演奏目標</p>
            <div class="listen">{beat_audio(bc['aligner_before'], '拍セル記憶')}{beat_audio(bc['aligner_after'], '100 Hz再現')}</div>
            <dl><div><dt>pointer MAE cells</dt><dd>{n(am['clock_pointer_mae_cells'], 3)}</dd></div>
            <div><dt>EOS MAE ms</dt><dd>{n(am['clock_eos_mae_ms'])}</dd></div>
            <div><dt>pitch MAE cents</dt><dd>{n(am['pitch_mae_cents'])}</dd></div>
            <div><dt>voice F1</dt><dd>{n(am['voice_f1'], 3)}</dd></div>
            <div><dt>onset F1 exact</dt><dd>{n(am['onset_f1_exact_frame'], 3)}</dd></div></dl>
            <p class="note">これはOracle拍セルでの単独Gate。100 Hz tick以外の正解時刻は推論入力に与えていない。afterは診断再合成で、物理演奏ではない。</p></section>'''
            if beat.get("temporal_connected"):
                cm = beat["temporal_connected"]; cst = "PASS" if cm.get("pass") else "FAIL"
                beat_section += f'''<section class="card"><div class="stage"><b>E</b><span>{cst}</span></div>
                <h2>Connected audio → 100 Hz Gate</h2><p class="flow">raw audio → Ear → Beat → Memory → Clock → Aligner</p>
                <div class="listen">{beat_audio(bc['tempo_before'], '生のお手本')}{beat_audio(bc['aligner_after'], '全段後の再現')}</div>
                <dl><div><dt>pitch MAE cents</dt><dd>{n(cm['pitch_mae_cents'])}</dd></div>
                <div><dt>correlation</dt><dd>{n(cm['trajectory_correlation'], 3)}</dd></div>
                <div><dt>voice F1</dt><dd>{n(cm['voice_f1'], 3)}</dd></div>
                <div><dt>rest false positive</dt><dd>{n(cm['rest_false_positive_rate'], 3)}</dd></div>
                <div><dt>onset F1 30 ms</dt><dd>{n(cm['onset_f1_30ms'], 3)}</dd></div></dl>
                <p class="note">正解BPM・正解セル・正解時刻を使わない frozen predicted-upstream 最終試験。単独PASSでも接続誤差が累積するため、現在はFAIL。</p></section>'''
        if beat.get("timing_profile_trained") and beat.get("timing_profile"):
            tm = beat["timing_profile"]; tst = "PASS" if tm.get("pass") else "FAIL"
            beat_section += f'''<section class="card"><div class="stage"><b>T</b><span>{tst}</span></div>
            <h2>Stored Timing Profile NN</h2><p class="flow">Tempo内部系列 → 単調な絶対セル位置列</p>
            <div class="listen">{beat_audio(bc['timing_before'], '生のお手本')}{beat_audio(bc['timing_after'], '保存profile click')}</div>
            <dl><div><dt>pointer MAE cells</dt><dd>{n(tm['pointer_mae_cells'], 3)}</dd></div>
            <div><dt>pointer P95 cells</dt><dd>{n(tm['pointer_p95_cells'], 3)}</dd></div>
            <div><dt>backward jumps</dt><dd>{n(tm['backward_jumps'], 0)}</dd></div></dl>
            <p class="note">afterは保存されたTiming Profileの拍だけを鳴らす診断click。物理演奏音ではない。</p></section>'''
        if beat.get("timeline_memory_trained") and beat.get("timeline_memory"):
            lm = beat["timeline_memory"]; lst = "PASS" if lm.get("pass") else "FAIL"
            source_note = " / ".join(f"{html.escape(k)} {n(v['pitch_mae_cents'])}c" for k, v in lm.get("by_source", {}).items())
            beat_section += f'''<section class="card"><div class="stage"><b>L</b><span>{lst}</span></div>
            <h2>Beat-conditioned Timeline Memory</h2><p class="flow">Beat内部系列 + Ear → 保存100 Hz profile → 再現</p>
            <div class="listen">{beat_audio(bc['timeline_before'], '生のお手本')}{beat_audio(bc['timeline_after'], '保存tensorから再現')}</div>
            <dl><div><dt>pitch MAE cents</dt><dd>{n(lm['pitch_mae_cents'])}</dd></div>
            <div><dt>correlation</dt><dd>{n(lm['trajectory_correlation'], 3)}</dd></div>
            <div><dt>voice F1</dt><dd>{n(lm['voice_f1'], 3)}</dd></div>
            <div><dt>rest false positive</dt><dd>{n(lm['rest_false_positive_rate'], 4)}</dd></div>
            <div><dt>onset F1 30 ms</dt><dd>{n(lm['onset_f1_30ms'], 3)}</dd></div></dl>
            <p class="note">remember後はraw音声とEar入力を破棄し、保存NN tensorだけでdecode。BPM/セル表現と併存するドリフトなし実行profile。音源別: {source_note}</p></section>'''
        if beat.get("timeline_position"):
            pm = beat["timeline_position"]; pst = "PASS" if pm.get("pass") else "FAIL"
            beat_section += f'''<section class="card"><div class="stage"><b>P</b><span>{pst}</span></div>
            <h2>Position Planner NN</h2><p class="flow">100 Hz音程・発音 → 正規化プランジャ位置</p>
            <div class="listen">{beat_audio(bc['position_before'], 'Timeline入力')}{beat_audio(bc['position_after'], '位置→定常笛音')}</div>
            <dl><div><dt>oracle position MAE %</dt><dd>{n(pm['position_mae_percent_stroke'], 3)}</dd></div>
            <div><dt>oracle steady pitch MAE</dt><dd>{n(pm['steady_pitch_mae_cents'])}</dd></div>
            <div><dt>connected position MAE %</dt><dd>{n(pm['connected_position_mae_percent_stroke'], 3)}</dd></div>
            <div><dt>connected steady pitch MAE</dt><dd>{n(pm['connected_steady_pitch_mae_cents'])}</dd></div></dl>
            <p class="note">単独GateはOracle音程で判定してPASS。Timeline接続値は前段誤差を含みFAIL。afterは定常位置の診断再合成で、モータ動特性は次Gate。</p></section>'''
        beat_section += "</div>"

    history_path = pathlib.Path(args.manifest).parent.parent / "e2e-beat-results" / "attempt-history.json"
    history_section = "<p>試行履歴はまだ生成されていません。</p>"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))["attempts"]
        failed = sum(not row["pass"] for row in history); passed = len(history) - failed; groups = []
        for stage in dict.fromkeys(row["stage"] for row in history):
            rows = [row for row in history if row["stage"] == stage]
            cards_history = []
            for row in rows:
                status = "PASS" if row["pass"] else "FAIL"
                metric_text = " / ".join(f"{html.escape(k.replace('_', ' '))} {n(v, 3)}" for k, v in list(row["metrics"].items())[:6])
                listening = ""
                if row.get("audio_manifest"):
                    manifest_path = pathlib.Path(args.manifest).parent.parent / row["audio_manifest"]
                    if manifest_path.exists():
                        am = json.loads(manifest_path.read_text(encoding="utf-8")); case = am["cases"][0]
                        pairs = (("position_before", "position_after"), ("timeline_before", "timeline_after"),
                                 ("timing_before", "timing_after"), ("aligner_before", "aligner_after"),
                                 ("memory_before", "memory_after"), ("tempo_before", "tempo_after"))
                        pair = next((p for p in pairs if p[0] in case and p[1] in case), None)
                        if pair:
                            prefix = pathlib.PurePosixPath(row["audio_manifest"]).parent.as_posix() + "/"
                            listening = f'<div class="listen">{doc_audio(prefix + case[pair[0]], "before")}{doc_audio(prefix + case[pair[1]], "after")}</div>'
                cards_history.append(f'''<article class="attempt"><div class="stage"><b>試</b><span>{status}</span></div>
                <h3>{html.escape(row['id'])}</h3><p>{html.escape(row['reason'])}</p><p class="metrics">{metric_text or '数値なし'}</p>
                <p class="note">split: {html.escape(str(row['split']))} / seed: {html.escape(str(row.get('seed')))}</p>{listening}</article>''')
            groups.append(f'''<details {'open' if stage in ('Temporal Aligner', 'Timeline Memory') else ''}><summary>{html.escape(stage)} — {len(rows)}試行</summary><div class="attempt-grid">{''.join(cards_history)}</div></details>''')
        history_section = f'''<p>全 {len(history)} 試行（PASS {passed} / FAIL {failed}）。FAILも削除せず、固定レポート値と残存checkpointの代表WAVを掲載する。</p>{''.join(groups)}'''

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
    attempts = []
    attempt_root = pathlib.Path(args.manifest).parent / "attempts"
    for attempt_manifest in sorted(attempt_root.glob("*/manifest.json")):
        attempt = json.loads(attempt_manifest.read_text(encoding="utf-8"))
        ff = attempt["summary"]["feedforward"]
        sample = next((x for x in attempt["audio"] if x["case"] == "step_up_down"), attempt["audio"][0])
        prefix = attempt_manifest.parent.relative_to(pathlib.Path(args.manifest).parent).as_posix() + "/"
        attempts.append(f'''<section class="card"><div class="stage"><b>試</b><span>FAIL</span></div>
        <h2>{html.escape(attempt_manifest.parent.name)}</h2>
        <p>MAE {n(ff['mae'])} cent / correlation {n(ff['correlation'], 3)} / direction {n(ff['direction'], 3)} / gain {n(ff['gain'], 3)}</p>
        <div class="listen">{audio(prefix + sample['ff_before'], "お手本")}{audio(prefix + sample['ff_after'], "試行後")}</div></section>''')
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
details{{margin:12px 0;border:1px solid var(--line);border-radius:14px;background:#0b1620}}summary{{cursor:pointer;padding:14px 18px;color:var(--cyan);font-weight:700}}.attempt-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;padding:0 12px 12px}}.attempt{{background:#101d29;border:1px solid var(--line);border-radius:12px;padding:14px}}.attempt h3{{font-size:.95rem;overflow-wrap:anywhere;margin:.5rem 0}}.attempt .metrics{{font-size:.78rem;color:#c7d6df}}
footer{{color:var(--muted);margin-top:50px;border-top:1px solid var(--line);padding-top:18px}}@media(max-width:600px){{main{{padding-top:28px}}dl{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="eyebrow">PHYSICAL AI / OBJECTIVE GATED DEVELOPMENT</p><h1>お手本は、<br>本当に演奏になったか。</h1>
<p class="lead">単一の総合スコアで隠さず、聴覚・記憶・フィードフォワード演奏・誤差判定を独立したNNに分け、未知の固定課題で評価した。各カードの音声は同じ <code>step_up_down</code> 課題の「NN前 / NN後」である。</p>
<h2>拍グリッド再設計</h2><p>お手本をBPMと拍位相へ割り当て、その拍セル上に連続音程・休符・onset・offset・傾斜を記憶する。BPMラベルは任意秒長の旧データへ後付けせず、BPMから生成した専用データだけで評価する。</p>{beat_section}
<div class="verdict"><b>現在の総合判定: 未完成</b> — 全段がPASSするまでE2E成功とは呼ばない。Feedback Residualは trained={str(feedback['trained']).lower()} / pass={str(feedback['pass']).lower()}（{html.escape(feedback['reason'])}）。</div>
<h2>全試行履歴（失敗モデルを含む）</h2>{history_section}
<div class="pipeline"><span>raw reference</span><i>→</i><span>Neural Ear</span><i>→</i><span>Reference Memory</span><i>→</i><span>FF Policy</span><i>→</i><span>plant + self audio</span><i>→</i><span>Error Comparator</span><i>→</i><span>Feedback Residual</span></div>
<div class="grid">{''.join(cards)}</div>
<h2>評価を厳格化した理由</h2><p>旧一体モデルはMAE {old['mae']:.1f} cent、軌跡相関 {old['corr']:.3f} で実際には逆方向へ追従した。一方、旧方向指標だけは {old['direction']:.1f} だった。{old['explanation']}。新評価は発音できない目標区間へ1200 cent罰を与え、P90、発音率、休符漏れ、遷移ゲイン、整定誤差をケース別に残す。</p>
<h2>学習の分離と再統合</h2><p>前段が合格したら凍結して次段を学習する。FFは名目rig・自己音なしから開始し、同じ参照でもrigごとに異なるOracle操作を回帰する不可能問題を避けた。次に閉ループDAggerでFeedback Residualを学習し、最後に未知rigと複数takeのRig Adapterを追加する。最終配備時は各NNを一つのforward graphとcheckpointへ束ねられるため、分離評価とE2E推論は両立する。</p>
<h2>今後の実装計画</h2>
<div class="grid">
  <section class="card">
    <div class="stage"><b>M</b><span>PLAN</span></div>
    <h2>交換可能な個別NNと統合E2E</h2>
    <p class="flow">modular learned / joint E2E / deterministic hybrid</p>
    <p>Ear、Tempo/Beat、Musical Memory、Aligner、Planner、Motor State、Controller、Comparator、Feedbackに共通Tensor契約を定義する。開発時は個別checkpointと中間評価を維持し、配備時は単一のComposite graphへ統合する。</p>
    <ul>
      <li>個別NN接続: 合格済みブロックを凍結して接続</li>
      <li>joint E2E: 補助損失を残して全体を低学習率で微調整</li>
      <li>hybrid: 任意ブロックを固定BPM、物理Planner、dead reckoning、PID等へ交換</li>
      <li>oracle input / predicted upstream / closed loopの3段階評価</li>
      <li>同一曲のWAV、精度、P99レイテンシ、モデルサイズを比較</li>
    </ul>
    <p><a href="modular-hybrid-model-plan.html">HTML詳細計画</a> / <a href="modular-hybrid-model-plan.md">Markdown原文</a></p>
  </section>
  <section class="card">
    <div class="stage"><b>A</b><span>PLAN</span></div>
    <h2>8小節文脈・コード理解・アドリブ</h2>
    <p class="flow">waveform + multi-resolution STFT → long musical memory → improvisation</p>
    <p>生波形と時間×周波数のSTFTを統合し、拍同期トークンへ圧縮する。8小節を16分音符単位なら約128セルとして保持し、和声を一点断定せず候補分布と潜在表現で扱う。</p>
    <ul>
      <li>masked bar reconstructionと次4小節予測で長期文脈を学習</li>
      <li>Imitation HeadとImprovisation Headが共通Memoryを利用</li>
      <li>continuation、variation、call-and-response、soloを分離評価</li>
      <li>Playability NNで音域、到達時間、発音余裕、危険率を評価</li>
      <li>理想生成WAVと物理シミュレータ演奏WAVを別々に公開</li>
    </ul>
    <p><a href="improvisation-extension-plan.html">HTML詳細計画</a> / <a href="improvisation-extension-plan.md">Markdown原文</a></p>
  </section>
</div>
<h2>不採用試行の音</h2><p>改善しなかった学習も消さず、同じ固定課題のWAVと指標を残す。</p><div class="grid">{''.join(attempts) if attempts else '<p>まだ記録なし</p>'}</div>
<h2>全ケース詳細</h2>{''.join(details)}
<footer>seed {data['seed']} / checkpoint {html.escape(data['checkpoint'])}<br>{html.escape(data['sonification_note'])}</footer>
</main></body></html>'''
    path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(doc, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__": main()
