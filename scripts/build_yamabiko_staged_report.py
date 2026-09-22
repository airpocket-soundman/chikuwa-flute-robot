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


def figure(src, caption, prefix=""):
    if not src:
        return ""
    return (f'<figure class="pitch-plot"><img loading="lazy" src="{html.escape(prefix + src)}" '
            f'alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')


def flow_node(title, state, detail):
    labels = {"pass": "PASS", "fail": "FAIL", "partial": "単独PASS / 接続FAIL",
              "pending": "未学習・未評価", "neutral": "INPUT"}
    return (f'<li class="flow-node {state}"><span>{html.escape(labels[state])}</span>'
            f'<b>{html.escape(title)}</b><small>{html.escape(detail)}</small></li>')


def flow_arrow(state="neutral", label=""):
    return (f'<li class="flow-arrow {state}" aria-hidden="true"><i>→</i>'
            f'<small>{html.escape(label)}</small></li>')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="docs/e2e-results/manifest.json")
    ap.add_argument("--out", default="docs/e2e-training-report.html")
    args = ap.parse_args(); data = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    s, examples, feedback = data["summary"], data["audio"], data["feedback"]
    representative = next((x for x in examples if x["case"] == "step_up_down"), examples[0])
    docs_root = pathlib.Path(args.manifest).parent.parent
    history_path = docs_root / "e2e-beat-results" / "attempt-history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))["attempts"] if history_path.exists() else []
    beat_path = pathlib.Path(args.manifest).parent.parent / "e2e-beat-results" / "manifest.json"
    beat = {}
    beat_cards = {"perception": [], "memory": [], "planning": [], "feedback": []}
    if beat_path.exists():
        beat = json.loads(beat_path.read_text(encoding="utf-8")); bm = beat["tempo_beat"]
        bc = next((x for x in beat["cases"] if x["case"] == "bpm_137"), beat["cases"][0])
        status = "PASS" if bm.get("pass") else "FAIL"
        beat_cards["perception"].append(f'''<section class="card"><div class="stage"><b>B</b><span class="{status.lower()}">{status}</span></div>
        <h2>Tempo / Beat NN</h2><p class="flow">raw reference → BPM・連続beat phase</p>
        <div class="listen">{beat_audio(bc['tempo_before'], 'お手本')}{beat_audio(bc['tempo_after'], '予測拍click')}</div>
        {figure(bc.get('tempo_plot'), '横軸: 時間 / 縦軸: 音程。橙の縦線は予測拍（このNNは音程列を出力しない）', 'e2e-beat-results/')}
        <dl><div><dt>tempo median relative error</dt><dd>{n(bm['tempo_relative_error_median'], 4)}</dd></div>
        <div><dt>phase MAE cycle</dt><dd>{n(bm['phase_circular_mae_cycle'], 4)}</dd></div></dl>
        <p class="note">afterはTempo NNの拍だけを鳴らした診断clickで、演奏音ではない。official path: {html.escape(beat['official_path'])}</p>
        </section>''')
        if beat.get("musical_memory_trained") and beat.get("musical_memory"):
            mm = beat["musical_memory"]; ms = "PASS" if mm.get("pass") else "FAIL"
            beat_cards["memory"].append(f'''<section class="card legacy"><div class="stage"><b>M</b><span class="{ms.lower()}">{ms}</span></div>
            <h2>Musical Memory NN</h2><p class="flow">予測beat + 音響表現 → 音程・休符・onset/offsetセル</p>
            <div class="listen">{beat_audio(bc['memory_before'], 'NNの前')}{beat_audio(bc['memory_after'], '記憶から再合成')}</div>
            {figure(bc.get('memory_plot'), '横軸: 時間 [s] / 縦軸: 音程 [cent] — お手本と記憶出力', 'e2e-beat-results/')}
            <dl><div><dt>pitch MAE cents</dt><dd>{n(mm['pitch_mae_cents'])}</dd></div>
            <div><dt>voice F1</dt><dd>{n(mm['voice_f1'], 3)}</dd></div><div><dt>onset F1</dt><dd>{n(mm['onset_f1'], 3)}</dd></div>
            <div><dt>correlation</dt><dd>{n(mm['trajectory_correlation'], 3)}</dd></div></dl>
            <p class="note">旧セル方式。現行本線ではTimeline Memoryへ置換。raw音声と正解セル数を捨て、予測BPM・予測位相から求めたセル数で再生した診断音。</p></section>''')
        if beat.get("temporal_aligner_trained") and beat.get("temporal_aligner"):
            am = beat["temporal_aligner"]; ast = "PASS" if am.get("pass") else "FAIL"
            beat_cards["memory"].append(f'''<section class="card legacy"><div class="stage"><b>A</b><span class="{ast.lower()}">{ast}</span></div>
            <h2>Neural Clock / Temporal Aligner</h2><p class="flow">記憶セル + BPM + 100 Hz tick → 時刻付き演奏目標</p>
            <div class="listen">{beat_audio(bc['aligner_before'], '拍セル記憶')}{beat_audio(bc['aligner_after'], '100 Hz再現')}</div>
            {figure(bc.get('aligner_plot'), '横軸: 時間 [s] / 縦軸: 音程 [cent] — 記憶セルと100 Hz再現', 'e2e-beat-results/')}
            <dl><div><dt>pointer MAE cells</dt><dd>{n(am['clock_pointer_mae_cells'], 3)}</dd></div>
            <div><dt>EOS MAE ms</dt><dd>{n(am['clock_eos_mae_ms'])}</dd></div>
            <div><dt>pitch MAE cents</dt><dd>{n(am['pitch_mae_cents'])}</dd></div>
            <div><dt>voice F1</dt><dd>{n(am['voice_f1'], 3)}</dd></div>
            <div><dt>onset F1 exact</dt><dd>{n(am['onset_f1_exact_frame'], 3)}</dd></div></dl>
            <p class="note">旧セル方式の単独Gate。現行本線は保存100 Hz Timelineを使用。afterは診断再合成で、物理演奏ではない。</p></section>''')
            if beat.get("temporal_connected"):
                cm = beat["temporal_connected"]; cst = "PASS" if cm.get("pass") else "FAIL"
                beat_cards["memory"].append(f'''<section class="card"><div class="stage"><b>E</b><span class="{cst.lower()}">{cst}</span></div>
                <h2>Connected audio → 100 Hz Gate</h2><p class="flow">raw audio → Ear → Beat → Memory → Clock → Aligner</p>
                <div class="listen">{beat_audio(bc['tempo_before'], '生のお手本')}{beat_audio(bc['aligner_after'], '全段後の再現')}</div>
                <dl><div><dt>pitch MAE cents</dt><dd>{n(cm['pitch_mae_cents'])}</dd></div>
                <div><dt>correlation</dt><dd>{n(cm['trajectory_correlation'], 3)}</dd></div>
                <div><dt>voice F1</dt><dd>{n(cm['voice_f1'], 3)}</dd></div>
                <div><dt>rest false positive</dt><dd>{n(cm['rest_false_positive_rate'], 3)}</dd></div>
                <div><dt>onset F1 30 ms</dt><dd>{n(cm['onset_f1_30ms'], 3)}</dd></div></dl>
                <p class="note">正解BPM・正解セル・正解時刻を使わない frozen predicted-upstream 最終試験。単独PASSでも接続誤差が累積するため、現在はFAIL。</p></section>''')
        if beat.get("timing_profile_trained") and beat.get("timing_profile"):
            tm = beat["timing_profile"]; tst = "PASS" if tm.get("pass") else "FAIL"
            beat_cards["memory"].append(f'''<section class="card legacy"><div class="stage"><b>T</b><span class="{tst.lower()}">{tst}</span></div>
            <h2>Stored Timing Profile NN</h2><p class="flow">Tempo内部系列 → 単調な絶対セル位置列</p>
            <div class="listen">{beat_audio(bc['timing_before'], '生のお手本')}{beat_audio(bc['timing_after'], '保存profile click')}</div>
            {figure(bc.get('timing_plot'), '横軸: 時間 / 縦軸: 音程。橙の縦線は保存profileの拍位置', 'e2e-beat-results/')}
            <dl><div><dt>pointer MAE cells</dt><dd>{n(tm['pointer_mae_cells'], 3)}</dd></div>
            <div><dt>pointer P95 cells</dt><dd>{n(tm['pointer_p95_cells'], 3)}</dd></div>
            <div><dt>backward jumps</dt><dd>{n(tm['backward_jumps'], 0)}</dd></div></dl>
            <p class="note">不採用のTiming Profile方式。afterは保存された拍だけを鳴らす診断clickで、物理演奏音ではない。</p></section>''')
        if beat.get("timeline_memory_trained") and beat.get("timeline_memory"):
            lm = beat["timeline_memory"]; lst = "PASS" if lm.get("pass") else "FAIL"
            source_note = " / ".join(f"{html.escape(k)} {n(v['pitch_mae_cents'])}c" for k, v in lm.get("by_source", {}).items())
            beat_cards["memory"].append(f'''<section class="card"><div class="stage"><b>L</b><span class="{lst.lower()}">{lst}</span></div>
            <h2>Beat-conditioned Timeline Memory</h2><p class="flow">Beat内部系列 + Ear → 保存100 Hz profile → 再現</p>
            <div class="listen">{beat_audio(bc['timeline_before'], '生のお手本')}{beat_audio(bc['timeline_after'], '保存tensorから再現')}</div>
            {figure(bc.get('timeline_plot'), '横軸: 時間 [s] / 縦軸: 音程 [cent] — お手本とTimeline再現', 'e2e-beat-results/')}
            <dl><div><dt>pitch MAE cents</dt><dd>{n(lm['pitch_mae_cents'])}</dd></div>
            <div><dt>correlation</dt><dd>{n(lm['trajectory_correlation'], 3)}</dd></div>
            <div><dt>voice F1</dt><dd>{n(lm['voice_f1'], 3)}</dd></div>
            <div><dt>rest false positive</dt><dd>{n(lm['rest_false_positive_rate'], 4)}</dd></div>
            <div><dt>onset F1 30 ms</dt><dd>{n(lm['onset_f1_30ms'], 3)}</dd></div></dl>
            <p class="note">現行本線。remember後はraw音声とEar入力を破棄し、保存NN tensorだけでdecode。音源別: {source_note}</p></section>''')
        if beat.get("timeline_position"):
            pm = beat["timeline_position"]; pst = "PASS" if pm.get("pass") else "FAIL"
            beat_cards["planning"].append(f'''<section class="card"><div class="stage"><b>P</b><span class="partial">単独 PASS / 接続 FAIL</span></div>
            <h2>Position Planner NN</h2><p class="flow">100 Hz音程・発音 → 正規化プランジャ位置</p>
            <div class="listen">{beat_audio(bc['position_before'], 'Timeline入力')}{beat_audio(bc['position_after'], '位置→定常笛音')}</div>
            {figure(bc.get('position_plot'), '横軸: 時間 [s] / 縦軸: 音程 [cent] — Timeline入力と位置計画後', 'e2e-beat-results/')}
            <dl><div><dt>oracle position MAE %</dt><dd>{n(pm['position_mae_percent_stroke'], 3)}</dd></div>
            <div><dt>oracle steady pitch MAE</dt><dd>{n(pm['steady_pitch_mae_cents'])}</dd></div>
            <div><dt>connected position MAE %</dt><dd>{n(pm['connected_position_mae_percent_stroke'], 3)}</dd></div>
            <div><dt>connected steady pitch MAE</dt><dd>{n(pm['connected_steady_pitch_mae_cents'])}</dd></div></dl>
            <p class="note">単独GateはOracle音程で判定してPASS。Timeline接続値は前段誤差を含みFAIL。afterは定常位置の診断再合成で、モータ動特性は次Gate。</p></section>''')

    history_section = "<p>試行履歴はまだ生成されていません。</p>"
    if history:
        failed = sum(not row["pass"] for row in history); passed = len(history) - failed
        history_processes = (
            ("1", "聞く・拍を取る", "音声からBPMと連続拍位相を推定", ("Tempo / Beat",)),
            ("2", "現行の記憶", "拍条件付き100 Hz Timelineを保存して再生", ("Timeline Memory",)),
            ("3", "位置へ変換", "音程・発音列からプランジャ位置を計画", ("Position Planner",)),
            ("R", "不採用・研究中の時間方式", "セル記憶、Clock、Aligner、Timing Profileと接続失敗を保存", ("Musical Memory", "Duration Clock", "Temporal Aligner", "Timing Profile", "Connected Temporal")),
        )
        process_blocks = []
        for process_number, process_title, process_note, process_stages in history_processes:
            stage_groups = []
            for stage in process_stages:
                rows = sorted((row for row in history if row["stage"] == stage),
                              key=lambda row: (not row["pass"], row["id"]))
                if not rows: continue
                cards_history = []
                for row in rows:
                    attempt_status = "PASS" if row["pass"] else "FAIL"
                    metric_text = " / ".join(f"{html.escape(k.replace('_', ' '))} {n(v, 3)}" for k, v in list(row["metrics"].items())[:6])
                    listening = ""
                    if row.get("audio_manifest"):
                        manifest_path = docs_root / row["audio_manifest"]
                        if manifest_path.exists():
                            am = json.loads(manifest_path.read_text(encoding="utf-8")); case = am["cases"][0]
                            pairs = (("position_before", "position_after", "position_plot"),
                                     ("timeline_before", "timeline_after", "timeline_plot"),
                                     ("timing_before", "timing_after", "timing_plot"),
                                     ("aligner_before", "aligner_after", "aligner_plot"),
                                     ("memory_before", "memory_after", "memory_plot"),
                                     ("tempo_before", "tempo_after", "tempo_plot"))
                            pair = next((p for p in pairs if p[0] in case and p[1] in case), None)
                            if pair:
                                prefix = pathlib.PurePosixPath(row["audio_manifest"]).parent.as_posix() + "/"
                                plot_caption = ("横軸: 時間 [s] / 縦軸: 音程 [cent, A4=0] — シアン: before / 橙: after"
                                                if pair[2] not in ("tempo_plot", "timing_plot") else
                                                "横軸: 時間 / 縦軸: お手本音程 — 橙の縦線: 拍イベント（音程出力なし）")
                                listening = (f'<div class="listen">{doc_audio(prefix + case[pair[0]], "before")}'
                                             f'{doc_audio(prefix + case[pair[1]], "after")}</div>'
                                             f'{figure(case.get(pair[2]), plot_caption, prefix)}')
                    cards_history.append(f'''<article class="attempt"><div class="stage"><b>試</b><span class="{attempt_status.lower()}">{attempt_status}</span></div>
                    <h3>{html.escape(row['id'])}</h3><p>{html.escape(row['reason'])}</p><p class="metrics">{metric_text or '数値なし'}</p>
                    <p class="note">split: {html.escape(str(row['split']))} / seed: {html.escape(str(row.get('seed')))}</p>{listening}</article>''')
                stage_passed = sum(row["pass"] for row in rows)
                stage_groups.append(f'''<details {'open' if stage_passed else ''}><summary>{html.escape(stage)} — PASS {stage_passed} / FAIL {len(rows)-stage_passed}</summary><div class="attempt-grid">{''.join(cards_history)}</div></details>''')
            process_blocks.append(f'''<section class="history-process"><header><b>{process_number}</b><div><h3>工程 {process_number}: {process_title}</h3><p>{process_note}</p></div></header>{''.join(stage_groups)}</section>''')
        history_section = f'''<p>全 {len(history)} 試行（PASS {passed} / FAIL {failed}）。現行工程と不採用方式を分け、FAILも削除せず掲載する。</p>{''.join(process_blocks)}'''

    stages = [
        ("1", "Neural Ear", "生の20 ms波形 → 音程・発音状態", s["ear"],
         representative["ear_before"], representative["ear_after"],
         "afterはNNが推定した音程を診断用音色で再合成。外部の音程解析器は推論入力に使わない。"),
        ("2", "Legacy Reference Memory", "Ear系列 → 保存メモリ → 音程系列", s["memory"],
         representative["memory_before"], representative["memory_after"],
         "旧参照メモリ方式。現行本線はBeat-conditioned Timeline Memory。raw音声を捨て、memory tensorだけから復元。"),
        ("3", "Legacy Physical Feed-forward Policy", "記憶したお手本 → PWM・バルブ → 物理シミュレータ", s["feedforward"],
         representative["ff_before"], representative["ff_after"],
         "旧物理演奏経路。afterはアクチュエータと笛のシミュレーションを通るが、現在のPosition Plannerとは未接続。"),
        ("4", "Error Comparator", "目標Ear + 自己音Ear → 符号付き誤差", s["comparator"],
         representative["comparator_before"], representative["comparator_after"],
         "beforeは意図的に外した自己音、afterは推定誤差を足し戻した診断用再合成。物理演奏ではない。"),
    ]
    for number, title, flow, metrics, before, after, note in stages:
        status = "PASS" if metrics.get("pass") else "FAIL"
        metric_lines = []
        for key, value in metrics.items():
            if key == "pass": continue
            metric_lines.append(f'<div><dt>{html.escape(key.replace("_", " "))}</dt><dd>{n(value, 3)}</dd></div>')
        bucket = "perception" if number == "1" else ("memory" if number == "2" else ("planning" if number == "3" else "feedback"))
        legacy_class = " legacy" if number in ("2", "3") else ""
        card_html = f'''<section class="card{legacy_class}">
          <div class="stage"><b>{number}</b><span class="{status.lower()}">{status}</span></div>
          <h2>{title}</h2><p class="flow">{flow}</p>
          <div class="listen">{audio(before, "NNの前")}{audio(after, "NNの後")}</div>
          <dl>{''.join(metric_lines)}</dl><p class="note">{note}</p>
        </section>'''
        beat_cards[bucket].insert(0, card_html) if number == "1" else beat_cards[bucket].append(card_html)
    beat_cards["planning"].append('''<section class="card pending"><div class="stage"><b>4</b><span class="pending">未評価</span></div>
      <h2>Motor / Controller / Flute</h2><p class="flow">目標位置 → モータ動特性 → バルブ → 笛</p>
      <p>現行Timeline→Position経路とは未統合。現在のafter WAVは定常位置からの診断再合成で、モータ動特性や実機を通していない。</p></section>''')
    beat_cards["feedback"].append(f'''<section class="card pending"><div class="stage"><b>F</b><span class="pending">未学習</span></div>
      <h2>Feedback Residual</h2><p class="flow">推定誤差 + 状態 → 操作量補正</p>
      <p>{html.escape(feedback['reason'])}。まだ学習・評価していない。</p></section>''')

    process_specs = (
        ("1", "聞く・拍を取る", "生音声から音程・発音、BPM、拍位相を推定する。", "perception"),
        ("2", "覚えて時間軸へ戻す", "お手本を保存し、演奏時刻に沿った100 Hz音程・休符列へ戻す。", "memory"),
        ("3", "身体で演奏する", "音程目標を位置へ変換し、モータと笛で音にする。", "planning"),
        ("4", "聴いて差を直す", "自己音と目標を比較し、次の操作を補正する。", "feedback"),
    )
    process_results = "".join(f'''<section class="process-section"><header><span>{number}</span><div><h2>工程 {number}: {title}</h2><p>{description}</p></div></header><div class="grid">{''.join(beat_cards[key])}</div></section>'''
                              for number, title, description, key in process_specs)

    tempo_metrics = beat.get("tempo_beat", {})
    timeline_metrics = beat.get("timeline_memory", {}) or {}
    position_metrics = beat.get("timeline_position", {}) or {}
    position_connected_pass = (position_metrics.get("connected_position_mae_percent_stroke", float("inf")) <= 1.0 and
                               position_metrics.get("connected_steady_pitch_mae_cents", float("inf")) <= 30.0)
    flow_items = [
        flow_node("お手本音声", "neutral", "raw waveform"), flow_arrow(),
        flow_node("Neural Ear", "pass" if s["ear"].get("pass") else "fail", f"MAE {n(s['ear'].get('mae'))} cent"), flow_arrow(),
        flow_node("Tempo / Beat", "pass" if tempo_metrics.get("pass") else "fail", f"BPM誤差 {n(tempo_metrics.get('tempo_relative_error_median', 0)*100, 2)}%"), flow_arrow(),
        flow_node("Timeline Memory", "pass" if timeline_metrics.get("pass") else "fail", f"MAE {n(timeline_metrics.get('pitch_mae_cents'))} cent"),
        flow_arrow("pass" if position_connected_pass else "fail", f"接続 {n(position_metrics.get('connected_steady_pitch_mae_cents'))} cent"),
        flow_node("Position Planner", "pass" if position_metrics.get("pass") and position_connected_pass else "partial",
                  f"単独 {n(position_metrics.get('steady_pitch_mae_cents'))} / 接続 {n(position_metrics.get('connected_steady_pitch_mae_cents'))} cent"),
        flow_arrow("fail", "現行経路は未統合"),
        flow_node("Motor / Flute", "fail", f"旧simulation MAE {n(s['feedforward'].get('mae'))} cent・実機未評価"), flow_arrow("fail"),
        flow_node("Error Comparator", "pass" if s["comparator"].get("pass") else "fail", f"error MAE {n(s['comparator'].get('error_mae'))} cent"), flow_arrow("pending"),
        flow_node("Feedback Residual", "pending", "未学習・反復適応も未統合"),
    ]
    flow_diagram = f'''<section class="flow-overview" aria-labelledby="flow-title"><div class="flow-heading"><div><h2 id="flow-title">全体フローと現在地</h2><p>緑は固定Gate PASS、赤はFAIL、橙は単独PASSだが接続FAIL、灰は未学習・未評価。</p></div><strong>全体 E2E: FAIL / 未完成</strong></div>
      <ol class="system-flow" aria-label="AI演奏システムの工程と合否">{''.join(flow_items)}</ol>
      <div class="feedback-return"><b>↩ Feedback loop</b><span>自己音の誤差からPosition / Controllerへ補正を戻す経路 — 未学習</span></div>
      <p class="flow-caveat">Neural Earは合成・音響乱数化ホールドアウトでのPASSで、実録音は未評価。Position PlannerのPASSは正解音程入力での単独Gateであり、Timeline接続はFAIL。</p></section>'''
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
    doc = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yamabiko E2E 分離NN 学習レポート</title>
<style>
:root{{--bg:#081018;--panel:#101d29;--ink:#edf6fb;--muted:#9eb2c0;--cyan:#4ee1d1;--red:#ff7b72;--line:#284052}}*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(circle at 80% 0,#163047 0,transparent 42%),var(--bg);color:var(--ink);font:15px/1.65 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:48px 20px 80px}}h1{{font-size:clamp(2rem,5vw,4.5rem);line-height:1.02;margin:.2em 0}}h2{{margin:.2rem 0}}h3{{margin-top:2rem}}.eyebrow,.flow{{color:var(--cyan);letter-spacing:.05em}}.lead{{font-size:1.15rem;max-width:820px;color:#c7d6df}}
.verdict{{border:1px solid var(--red);background:#28171a;padding:18px 22px;border-radius:14px;margin:28px 0}}.verdict b{{color:var(--red)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,310px),1fr));gap:18px}}.card{{position:relative;background:linear-gradient(145deg,#132433,#0d1822);border:1px solid var(--line);border-radius:18px;padding:22px}}
.stage{{display:flex;justify-content:space-between;align-items:center}}.stage b{{display:grid;place-items:center;width:36px;height:36px;border-radius:50%;background:var(--cyan);color:#071016}}.stage span{{font-weight:800;color:var(--red)}}.card:has(.stage span:first-child){{border-color:var(--cyan)}}
.listen{{display:grid;gap:10px;margin:18px 0}}.audio{{display:grid;grid-template-columns:64px 1fr;align-items:center;gap:8px}}audio{{width:100%;height:36px}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}dl div{{background:#0a141d;padding:9px;border-radius:8px}}dt{{font-size:.72rem;color:var(--muted)}}dd{{margin:0;font-size:1.05rem}}.note{{color:var(--muted);font-size:.9rem}}
.pitch-plot{{margin:14px 0 18px}}.pitch-plot img{{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#0b1620}}.pitch-plot figcaption{{margin-top:7px;color:var(--muted);font-size:.78rem}}.stage span.pass{{color:#70f0ac}}.stage span.fail{{color:#ff8c78}}.stage span.partial{{color:#ffc96b}}.stage span.pending{{color:#9eb2c0}}.legacy{{border-style:dashed;opacity:.86}}.pending{{border-color:#526675}}
.flow-overview{{margin:28px 0;padding:24px;background:#0b1620;border:1px solid var(--line);border-radius:18px}}.flow-heading{{display:flex;justify-content:space-between;gap:20px;align-items:start}}.flow-heading strong{{color:var(--red);border:1px solid var(--red);padding:8px 12px;border-radius:9px;white-space:nowrap}}.system-flow{{list-style:none;margin:22px 0;padding:0;display:flex;align-items:stretch;overflow-x:auto;gap:8px}}.flow-node{{min-width:145px;display:flex;flex-direction:column;gap:6px;padding:14px;border:2px solid var(--line);border-radius:12px;background:#101d29}}.flow-node span{{font-size:.72rem;font-weight:900}}.flow-node small{{color:var(--muted)}}.flow-node.pass{{border-color:#46c987}}.flow-node.pass span{{color:#70f0ac}}.flow-node.fail{{border-color:var(--red)}}.flow-node.fail span{{color:#ff8c78}}.flow-node.partial{{border-color:#d69c3b}}.flow-node.partial span{{color:#ffc96b}}.flow-node.pending,.flow-node.neutral{{border-color:#526675}}.flow-node.pending span,.flow-node.neutral span{{color:#b3c0c8}}.flow-arrow{{min-width:82px;display:grid;place-items:center;align-content:center;text-align:center;color:var(--muted)}}.flow-arrow i{{font-size:1.6rem;font-style:normal}}.flow-arrow.fail{{color:var(--red)}}.flow-arrow.pass{{color:#70f0ac}}.flow-arrow.pending{{color:#9eb2c0}}.feedback-return{{display:flex;gap:12px;align-items:center;border:1px dashed #526675;border-radius:10px;padding:10px 14px;color:var(--muted)}}.feedback-return b{{color:#9eb2c0}}.flow-caveat{{color:var(--muted);font-size:.88rem}}
.process-section{{margin:46px 0}}.process-section>header,.history-process>header{{display:flex;gap:14px;align-items:center;margin-bottom:16px}}.process-section>header>span,.history-process>header>b{{display:grid;place-items:center;flex:0 0 44px;height:44px;border-radius:12px;background:var(--cyan);color:#071016;font-size:1.15rem}}.process-section>header p,.history-process>header p{{margin:0;color:var(--muted)}}.history-process{{margin:26px 0;padding:18px;border-left:3px solid var(--line);background:#0a141d;border-radius:0 14px 14px 0}}
.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.82rem}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}code{{color:var(--cyan)}}
details{{margin:12px 0;border:1px solid var(--line);border-radius:14px;background:#0b1620}}summary{{cursor:pointer;padding:14px 18px;color:var(--cyan);font-weight:700}}.attempt-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:12px;padding:0 12px 12px}}.attempt{{background:#101d29;border:1px solid var(--line);border-radius:12px;padding:14px}}.attempt h3{{font-size:.95rem;overflow-wrap:anywhere;margin:.5rem 0}}.attempt .metrics{{font-size:.78rem;color:#c7d6df}}summary:focus-visible,a:focus-visible,audio:focus-visible{{outline:3px solid #ffc96b;outline-offset:3px}}
footer{{color:var(--muted);margin-top:50px;border-top:1px solid var(--line);padding-top:18px}}@media(max-width:700px){{main{{padding-top:28px}}dl{{grid-template-columns:1fr}}.flow-heading{{display:block}}.flow-heading strong{{display:inline-block;margin-top:8px}}.system-flow{{display:grid;overflow:visible}}.flow-arrow{{min-width:0;min-height:42px}}.flow-arrow i{{transform:rotate(90deg)}}.audio{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="eyebrow">PHYSICAL AI / OBJECTIVE GATED DEVELOPMENT</p><h1>お手本は、<br>本当に演奏になったか。</h1>
<p class="lead">単一の総合スコアで隠さず、聴覚・記憶・フィードフォワード演奏・誤差判定を独立したNNに分け、未知の固定課題で評価した。各カードの音声は同じ <code>step_up_down</code> 課題の「NN前 / NN後」である。</p>
{flow_diagram}
<div class="verdict"><b>現在の総合判定: 未完成</b> — 全段がPASSするまでE2E成功とは呼ばない。Feedback Residualは trained={str(feedback['trained']).lower()} / pass={str(feedback['pass']).lower()}（{html.escape(feedback['reason'])}）。</div>
<h2>工程別の最新結果</h2><p>現行の採用経路と旧方式をカード内で明記し、入力→出力の順に並べた。</p>{process_results}
<h2>全試行履歴（失敗モデルを含む）</h2>{history_section}
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
<h2>工程3・旧Physical Feed-forward試行</h2><p>現行Position Planner以前の物理シミュレーション試行。改善しなかった結果も、同じ固定課題のWAVと指標を残す。</p><div class="grid">{''.join(attempts) if attempts else '<p>まだ記録なし</p>'}</div>
<h2>全ケース詳細</h2>{''.join(details)}
<footer>seed {data['seed']} / checkpoint {html.escape(data['checkpoint'])}<br>{html.escape(data['sonification_note'])}</footer>
</main></body></html>'''
    path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(doc, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__": main()
