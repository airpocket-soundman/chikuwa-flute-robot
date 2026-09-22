"""Build the GitHub Pages report from the staged evaluation manifest."""
from __future__ import annotations

import argparse
import html
import json
import pathlib


def n(value, digits=1):
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value, digits=1):
    return "—" if value is None else f"{value * 100:.{digits}f}"


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


def metric_value(metrics, *keys):
    """Return the first available metric, keeping the report schema backwards compatible."""
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def gate_state(metrics, *keys, default="pending"):
    value = metric_value(metrics, *keys)
    if value is None:
        return default
    return "pass" if bool(value) else "fail"


def flow_node(title, state, detail, href=None):
    labels = {"pass": "PASS", "fail": "FAIL", "partial": "単独PASS / 接続FAIL",
              "pending": "未学習・未評価", "neutral": "INPUT"}
    content = (f'<span>{html.escape(labels[state])}</span><b>{html.escape(title)}</b>'
               f'<small>{html.escape(detail)}</small>')
    if href:
        content = f'<a href="{html.escape(href)}">{content}<em>工程の結果を見る ↓</em></a>'
    return f'<li class="flow-node {state}">{content}</li>'


def flow_arrow(state="neutral", label=""):
    return (f'<li class="flow-arrow {state}" aria-hidden="true"><i>→</i>'
            f'<small>{html.escape(label)}</small></li>')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="docs/e2e-results/manifest.json")
    ap.add_argument("--physical-manifest", default="docs/e2e-physical-results/manifest.json",
                    help="Optional motor/flute/feedback evaluation manifest")
    ap.add_argument("--composite-manifest", default="docs/e2e-composite-results/manifest.json",
                    help="Optional fully connected neural/deterministic evaluation manifest")
    ap.add_argument("--out", default="docs/e2e-training-report.html")
    args = ap.parse_args(); data = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    s, examples, feedback = data["summary"], data["audio"], data["feedback"]
    representative = next((x for x in examples if x["case"] == "step_up_down"), examples[0])
    docs_root = pathlib.Path(args.manifest).parent.parent
    physical_path = pathlib.Path(args.physical_manifest)
    physical = json.loads(physical_path.read_text(encoding="utf-8")) if physical_path.exists() else {}
    physical_summary = physical.get("summary", physical)
    if not physical_summary:
        fallback_reports = list((docs_root.parent / "runs").glob("yamabiko_physical_control*_report.json"))
        if fallback_reports:
            latest_report = max(fallback_reports, key=lambda path: path.stat().st_mtime)
            physical_summary = json.loads(latest_report.read_text(encoding="utf-8"))
    physical_prefix = ""
    if physical_path.exists():
        try:
            physical_prefix = physical_path.parent.relative_to(docs_root).as_posix().rstrip("/") + "/"
        except ValueError:
            physical_prefix = physical_path.parent.as_posix().rstrip("/") + "/"
    composite_path = pathlib.Path(args.composite_manifest)
    composite = json.loads(composite_path.read_text(encoding="utf-8")) if composite_path.exists() else {}
    composite_summary = composite.get("summary", {})
    composite_audio = composite.get("audio", {})
    uno_q_report_path = composite_path.parent / "uno_q_onnx_report.json"
    uno_q_report = json.loads(uno_q_report_path.read_text(encoding="utf-8")) if uno_q_report_path.exists() else {}
    hybrid_lab_path = composite_path.parent / "hybrid-lab.json"
    hybrid_lab = json.loads(hybrid_lab_path.read_text(encoding="utf-8")) if hybrid_lab_path.exists() else {}
    if hybrid_lab.get("samples"):
        hybrid_samples = hybrid_lab["samples"]
    elif hybrid_lab.get("combinations"):
        hybrid_samples = [{"id": "legacy", "title": hybrid_lab.get("sample", "4音ステップ"),
                           "description": "従来の単一評価サンプル",
                           "reference_wav": "e2e-composite-results/reference.wav",
                           "combinations": hybrid_lab["combinations"]}]
    else:
        hybrid_samples = []
    composite_prefix = ""
    if composite_path.exists():
        try:
            composite_prefix = composite_path.parent.relative_to(docs_root).as_posix().rstrip("/") + "/"
        except ValueError:
            composite_prefix = composite_path.parent.as_posix().rstrip("/") + "/"
    history_path = docs_root / "e2e-beat-results" / "attempt-history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))["attempts"] if history_path.exists() else []
    beat_path = pathlib.Path(args.manifest).parent.parent / "e2e-beat-results" / "manifest.json"
    beat = {}
    beat_cards = {"perception": [], "memory": [], "planning": [], "feedback": []}
    legacy_cards = []
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
            pm = beat["timeline_position"]
            oracle_pass = pm.get("oracle_input_pass", pm.get("pass", False))
            transform_pass = pm.get("real_input_transform_pass", False)
            output_pass = pm.get("connected_output_pass", False)
            inherited_mae = pm.get("inherited_pitch_mae_cents", float("nan"))
            added_mae = pm.get("model_added_pitch_mae_cents", float("nan"))
            total_mae = pm.get("output_total_pitch_mae_cents", pm.get("connected_steady_pitch_mae_cents"))
            gate_pass = oracle_pass and transform_pass and output_pass
            pst = "PASS" if gate_pass else "FAIL"
            beat_cards["planning"].append(f'''<section class="card"><div class="stage"><b>P</b><span class="{pst.lower()}">Position Gate {pst}</span></div>
            <h2>Position Planner NN = Feedforward</h2><p class="flow">100 Hz音程・発音 → 先読みした正規化プランジャ位置</p>
            <div class="listen">{beat_audio(bc['position_before'], 'Timeline入力')}{beat_audio(bc['position_after'], '位置→定常笛音')}</div>
            {figure(bc.get('position_plot'), '横軸: 時間 [s] / 縦軸: 音程 [cent] — Timeline入力と位置計画後', 'e2e-beat-results/')}
            <dl><div><dt>① 持ち込み誤差</dt><dd>{n(inherited_mae)} cent</dd></div>
            <div><dt>② Planner追加誤差</dt><dd>{n(added_mae)} cent — {'PASS' if transform_pass else 'FAIL'}</dd></div>
            <div><dt>③ 出口の総誤差</dt><dd>{n(total_mae)} cent — {'PASS' if output_pass else 'FAIL'}</dd></div>
            <div><dt>参考: 正解入力時</dt><dd>{n(pm['steady_pitch_mae_cents'])} cent — {'PASS' if oracle_pass else 'FAIL'}</dd></div>
            <div><dt>Planner追加位置MAE</dt><dd>{n(pm.get('model_added_position_mae_percent_stroke'), 3)}%</dd></div>
            <div><dt>入口からの正味MAE変化</dt><dd>{n(pm.get('net_absolute_change_cents'))} cent</dd></div>
            <div><dt>分解式の最大残差</dt><dd>{n(pm.get('attribution_residual_max_cents'), 6)} cent</dd></div></dl>
            <p class="note">このNNが音楽的フィードフォワードであり、別のFeedforward Policyは置かない。判定は正解入力・実Timeline入力でのPlanner固有変換・接続出口の3つを分離。同一曲・同一フレームの符号付き誤差は、持ち込み＋Planner追加＝出口総誤差。MAE同士は符号を失うため足し算にはならない。afterは定常位置の診断再合成で、モータ動特性は次Gate。</p></section>''')

    def history_attempt_card(row):
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
        return f'''<article class="attempt"><div class="stage"><b>試</b><span class="{attempt_status.lower()}">{attempt_status}</span></div>
        <h3>{html.escape(row['id'])}</h3><p>{html.escape(row['reason'])}</p><p class="metrics">{metric_text or '数値なし'}</p>
        <p class="note">split: {html.escape(str(row['split']))} / seed: {html.escape(str(row.get('seed')))}</p>{listening}</article>'''

    def render_history(selected):
        rows = [row for row in history if selected(row)]
        if not rows: return "<p>該当する試行記録はありません。</p>"
        stage_order = ("Tempo / Beat", "Musical Memory", "Duration Clock", "Temporal Aligner",
                       "Timing Profile", "Timeline Memory", "Position Planner", "Connected Temporal")
        groups = []
        for stage in stage_order:
            stage_rows = sorted((row for row in rows if row["stage"] == stage),
                                key=lambda row: (not row["pass"], row["id"]))
            if not stage_rows: continue
            passed = sum(row["pass"] for row in stage_rows)
            groups.append(f'''<details {'open' if passed else ''}><summary>{html.escape(stage)} — PASS {passed} / FAIL {len(stage_rows)-passed}</summary><div class="attempt-grid">{''.join(history_attempt_card(row) for row in stage_rows)}</div></details>''')
        passed = sum(row["pass"] for row in rows)
        return f'''<p class="history-count">{len(rows)}試行 — PASS {passed} / FAIL {len(rows)-passed}</p>{''.join(groups)}'''

    current_history = render_history(lambda row: row["stage"] in ("Tempo / Beat", "Timeline Memory", "Position Planner"))
    clock_history = render_history(lambda row: (row["stage"] in ("Musical Memory", "Temporal Aligner") or
                                                   (row["stage"] == "Connected Temporal" and "profile" not in row["id"])))
    timing_history = render_history(lambda row: (row["stage"] in ("Duration Clock", "Timing Profile") or
                                                  (row["stage"] == "Connected Temporal" and "profile" in row["id"])))

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
        if number == "1":
            stage_plot = figure(representative.get("ear_plot"),
                                "横軸: 時間 [s] / 縦軸: 音程 [cent, A4=0] — シアン: お手本 / 橙: Neural Ear出力",
                                "e2e-results/")
        elif number == "4":
            stage_plot = figure(representative.get("comparator_plot"),
                                "上段: 目標・自己音・補正後音程。下段: 必要な補正量とComparator予測 [cent]",
                                "e2e-results/")
        else:
            stage_plot = ""
        stage_explanation = ('''<div class="model-explainer"><h3>このモデルは何をしている？</h3>
          <p>記憶した目標音程と、Neural Earが聴いた現在の自己音を比べ、「何cent上げる／下げるべきか」を出力する。正なら音程を上げ、負なら下げる指示になる。</p>
          <p>ここでは誤差を測るだけで、モータは直接動かさない。この出力を次段のFeedback Residualが操作量の補正へ変換する。</p></div>''' if number == "4" else "")
        card_html = f'''<section class="card{legacy_class}">
          <div class="stage"><b>{number}</b><span class="{status.lower()}">{status}</span></div>
          <h2>{title}</h2><p class="flow">{flow}</p>{stage_explanation}
          <div class="listen">{audio(before, "NNの前")}{audio(after, "NNの後")}</div>{stage_plot}
          <dl>{''.join(metric_lines)}</dl><p class="note">{note}</p>
        </section>'''
        if number in ("2", "3"):
            legacy_cards.append(card_html)
        elif number == "1":
            beat_cards[bucket].insert(0, card_html)
        else:
            beat_cards[bucket].append(card_html)
    # Physical-control results are optional so the report can be generated before the
    # first run.  A missing gate is shown as untested, never silently treated as PASS.
    world_metrics = physical_summary.get("motor_audio_world_model", {})
    controller_metrics = physical_summary.get("motor_controller", physical_summary)
    feedback_metrics = physical_summary.get("feedback", physical_summary)
    world_state = gate_state(world_metrics, "pass") if world_metrics else "pending"
    controller_state = (gate_state(controller_metrics, "pass") if controller_metrics is not physical_summary else
                        gate_state(physical_summary, "motor_controller_pass"))
    physics_state = gate_state(physical_summary, "motor_physics_pass")
    flute_state = gate_state(physical_summary, "linear_flute_pass")
    residual_state = (gate_state(feedback_metrics, "pass") if feedback_metrics is not physical_summary else
                      gate_state(physical_summary, "feedback_pass"))

    def state_label(state):
        return {"pass": "PASS", "fail": "FAIL", "pending": "未評価"}[state]

    beat_cards["planning"].extend([
        f'''<section class="card {world_state}"><div class="stage"><b>W</b><span class="{world_state}">{state_label(world_state)}</span></div>
        <h2>Motor Audio World Model NN</h2><p class="flow">PWM履歴 → 次に聞こえる音程</p>
        <p>物理パラメータを推定せず、決定論的シミュレータを入出力だけから同定する。位置・速度・トルク・摩擦は教師にも入力にも使わない。</p>
        <dl><div><dt>holdout audio MAE</dt><dd>{n(metric_value(world_metrics, 'audio_mae_cents'), 1)} cent</dd></div>
        <div><dt>real rig</dt><dd>未検証</dd></div></dl></section>''',
        f'''<section class="card {controller_state}"><div class="stage"><b>C</b><span class="{controller_state}">{state_label(controller_state)}</span></div>
        <h2>Motor Controller NN</h2><p class="flow">Position Planner目標 + 過去PWMの潜在履歴 → PWM・valve</p>
        <p>Position Plannerが先読みした位置目標を追跡する低レベル制御器。音楽的なフィードフォワードを別に重複実装しない。</p>
        <dl><div><dt>position MAE</dt><dd>{n(metric_value(controller_metrics, 'position_mae_percent_stroke', 'motor_position_mae_percent_stroke'), 3)} % stroke</dd></div>
        <div><dt>feedforward pitch MAE</dt><dd>{n(metric_value(controller_metrics, 'feedforward_pitch_mae_cents'), 1)} cent</dd></div></dl></section>''',
        f'''<section class="card {physics_state}"><div class="stage"><b>P</b><span class="{physics_state}">{state_label(physics_state)}</span></div>
        <h2>Motor Physics Simulator</h2><p class="flow">PWM → torque rise・摩擦・慣性 → 実変位</p>
        <p>学習対象NNではなく、遅れ・オーバーシュート・摩擦を再現する訓練環境。乱数化した物理条件で制御器を評価する。</p></section>''',
        f'''<section class="card {flute_state}"><div class="stage"><b>F</b><span class="{flute_state}">{state_label(flute_state)}</span></div>
        <h2>Linear Flute Simulator</h2><p class="flow">実変位 → 線形音程 / valve → 発音ON・OFF</p>
        <p>笛の変位と音程を線形対応させた解析モデル。学習対象ではなく、モータの実変位を評価可能な音へ変換する。</p></section>''',
    ])

    physical_cases = physical.get("cases", [])
    if physical_cases:
        physical_case = next((case for case in physical_cases if case.get("case") == "step_up_down"), physical_cases[0])
        target_wav = metric_value(physical_case, "target_wav", "reference_wav", "before_wav")
        base_wav = metric_value(physical_case, "feedforward_wav", "base_wav", "open_loop_wav")
        closed_wav = metric_value(physical_case, "feedback_wav", "closed_wav", "closed_loop_wav", "after_wav")
        listens = "".join(doc_audio(physical_prefix + src, label) for src, label in
                          ((target_wav, "目標"), (base_wav, "Feedforward"), (closed_wav, "Feedback後")) if src)
        plots = "".join((
            figure(metric_value(physical_case, "pitch_plot"), "横軸: 時間 [s] / 縦軸: 音程 [cent] — 目標・Feedforward・Feedback後", physical_prefix),
            figure(metric_value(physical_case, "position_plot"), "横軸: 時間 [s] / 縦軸: 正規化変位 — 目標位置と物理シミュレータ変位", physical_prefix),
            figure(metric_value(physical_case, "control_plot", "pwm_plot"), "横軸: 時間 [s] / 縦軸: PWM — Feedforward出力とFeedback補正", physical_prefix),
        ))
        beat_cards["planning"].append(f'''<section class="card"><div class="stage"><b>W</b><span class="{residual_state}">{state_label(residual_state)}</span></div>
        <h2>Physical simulation WAV / plots</h2><p class="flow">目標 → Feedforward物理演奏 → Feedback補正後</p>
        <div class="listen">{listens}</div>{plots}
        <p class="note">これは内部シミュレータの音であり、実機録音ではない。real_rig_validated={str(bool(physical_summary.get('real_rig_validated', False))).lower()}。</p></section>''')

    beat_cards["feedback"].append(f'''<section class="card {residual_state}"><div class="stage"><b>R</b><span class="{residual_state}">{state_label(residual_state)}</span></div>
      <h2>Adaptive Feedback NN</h2><p class="flow">目標音程 − 自己音 + PWM/音程変化履歴 → PWM補正</p>
        <p>Position PlannerのフィードフォワードPWMを置き換えず、自己音のずれだけを小さな補正として加える。再生テンポは機構速度に合わせて0.5倍、音符境界では40 msだけバルブを閉じる。</p>
      <dl><div><dt>closed-loop pitch MAE</dt><dd>{n(metric_value(feedback_metrics, 'closed_pitch_mae_cents', 'feedback_pitch_mae_cents'), 1)} cent</dd></div>
      <div><dt>improvement</dt><dd>{pct(metric_value(feedback_metrics, 'improvement_fraction', 'feedback_improvement_fraction'), 1)} %</dd></div>
      <div><dt>non-worse rigs</dt><dd>{pct(metric_value(feedback_metrics, 'nonworse_rig_fraction', 'feedback_nonworse_rig_fraction'), 1)} %</dd></div></dl>
      <p class="note">評価は乱数化シミュレータ。実機適応と反復練習はまだ検証していない。</p></section>''')

    process_specs = (
        ("1", "聞く・拍を取る", "生音声から音程・発音、BPM、拍位相を推定する。", "perception"),
        ("2", "覚えて時間軸へ戻す", "お手本を保存し、演奏時刻に沿った100 Hz音程・休符列へ戻す。", "memory"),
        ("3", "身体で演奏する", "音程目標を位置へ変換し、モータと笛で音にする。", "planning"),
        ("4", "聴いて差を直す", "自己音と目標を比較し、次の操作を補正する。", "feedback"),
    )
    process_results = "".join(f'''<section class="process-section" id="process-{number}"><header><span>{number}</span><div><h2>工程 {number}: {title}</h2><p>{description}</p></div></header><div class="grid">{''.join(beat_cards[key])}</div></section>'''
                              for number, title, description, key in process_specs)

    tempo_metrics = beat.get("tempo_beat", {})
    timeline_metrics = beat.get("timeline_memory", {}) or {}
    position_metrics = beat.get("timeline_position", {}) or {}
    position_transform_pass = position_metrics.get("real_input_transform_pass", False)
    position_output_pass = position_metrics.get("connected_output_pass", False)
    position_gate_pass = (position_metrics.get("oracle_input_pass", False) and
                          position_transform_pass and position_output_pass)
    perception_flow = [
        flow_node("お手本音声", "neutral", "raw waveform", "#process-1"), flow_arrow(),
        flow_node("Neural Ear", "pass" if s["ear"].get("pass") else "fail", f"MAE {n(s['ear'].get('mae'))} cent", "#process-1"), flow_arrow(),
        flow_node("Tempo / Beat", "pass" if tempo_metrics.get("pass") else "fail", f"BPM誤差 {n(tempo_metrics.get('tempo_relative_error_median', 0)*100, 2)}%", "#process-1"),
    ]
    memory_flow = [
        flow_node("Timeline Memory", "pass" if timeline_metrics.get("pass") else "fail", f"MAE {n(timeline_metrics.get('pitch_mae_cents'))} cent", "#process-2"),
    ]
    performance_flow_primary = [
        flow_node("Position Planner = Feedforward", "pass" if position_gate_pass else "fail",
                  f"音程Timeline → 先読み位置 / 出口 {n(position_metrics.get('output_total_pitch_mae_cents'))} cent", "#process-3"),
        flow_arrow(controller_state, "位置目標"),
        flow_node("Motor Controller", controller_state,
                  f"位置MAE {n(metric_value(controller_metrics, 'position_mae_percent_stroke', 'motor_position_mae_percent_stroke'), 2)}% stroke", "#process-3"),
        flow_arrow(physics_state, "PWM・valve"),
        flow_node("Motor Physics", physics_state, "torque rise・摩擦・慣性 → 実変位", "#process-3"),
        flow_arrow(flute_state, "実変位"),
        flow_node("Linear Flute", flute_state, "実変位 → 音程 / valve → 発音", "#process-3"),
    ]
    performance_flow_secondary = [
        flow_node("Motor Audio World Model", world_state, "PWM履歴 → 次の可聴音程", "#process-3"),
        flow_arrow(world_state, "学習時だけ制御器へ勾配"),
        flow_node("Motor Controller", controller_state, "Planner目標 + 潜在履歴 → PWM", "#process-3"),
    ]
    comparator_metrics = physical_summary.get("comparator", s["comparator"])
    comparator_state = gate_state(comparator_metrics, "pass")
    correction_flow = [
        flow_node("Error Comparator", comparator_state,
                  f"error MAE {n(metric_value(comparator_metrics, 'error_mae', 'error_mae_cents'))} cent", "#process-4"),
        flow_arrow(residual_state, "符号付き音程差"),
        flow_node("Feedback Residual", residual_state,
                  f"閉ループMAE {n(metric_value(feedback_metrics, 'closed_pitch_mae_cents', 'feedback_pitch_mae_cents'))} cent", "#process-4"),
    ]
    performance_states = ["pass" if position_gate_pass else "fail", world_state,
                          controller_state, physics_state, flute_state]
    performance_gate_state = ("fail" if "fail" in performance_states else
                              ("pending" if "pending" in performance_states else "pass"))
    correction_states = [comparator_state, residual_state]
    correction_gate_state = ("fail" if "fail" in correction_states else
                             ("pending" if "pending" in correction_states else "pass"))
    flow_diagram = f'''<section class="flow-overview" aria-labelledby="flow-title"><div class="flow-heading"><div><h2 id="flow-title">全体フローと現在地</h2><p>緑は固定Gate PASS、赤はFAIL、橙は工程内に未評価部分あり、灰は未学習・未評価。</p></div><strong>全体 E2E: FAIL / 未完成</strong></div>
      <div class="phase-flow" aria-label="採用予定パイプラインの4工程">
        <section class="flow-phase"><header><span>1</span><div><b>聴く・拍を理解する</b><small>お手本から音程、休符、BPM、拍位置を取り出す</small></div><strong class="pass">工程 PASS</strong></header>
          <ol class="system-flow flow-row nodes-3">{''.join(perception_flow)}</ol></section>
        <div class="phase-connector pass"><span>↓</span><b>理解した音程と拍を保存する</b></div>
        <section class="flow-phase"><header><span>2</span><div><b>お手本をTimelineとして覚える</b><small>演奏時刻に沿った音程と休符の地図を保存する</small></div><strong class="pass">工程 PASS</strong></header>
          <ol class="system-flow flow-row nodes-1">{''.join(memory_flow)}</ol></section>
        <div class="phase-connector {'pass' if position_gate_pass else 'fail'}"><span>↓</span><b>同一フレーム再評価 — 持ち込み {n(position_metrics.get('inherited_pitch_mae_cents'))} ｜ Planner追加 {n(position_metrics.get('model_added_pitch_mae_cents'))} ｜ 出口 {n(position_metrics.get('output_total_pitch_mae_cents'))} cent（各MAEは非加算）</b></div>
        <section class="flow-phase"><header><span>3</span><div><b>身体で演奏する</b><small>Position Plannerがフィードフォワード目標を作り、PWM履歴の潜在表現でモータを制御する</small></div><strong class="{performance_gate_state if performance_gate_state != 'pending' else 'partial'}">Physical {state_label(performance_gate_state)}</strong></header>
          <ol class="system-flow flow-row nodes-4">{''.join(performance_flow_primary)}</ol>
          <ol class="system-flow flow-row nodes-2">{''.join(performance_flow_secondary)}</ol></section>
        <div class="phase-connector {correction_gate_state}"><span>↓</span><b>シミュレータ自己音と目標の差を閉ループ補正へ渡す</b></div>
        <section class="flow-phase"><header><span>4</span><div><b>自分の音を聴いて直す</b><small>目標との差を測り、Position Plannerの出力へPWM補正を足す</small></div><strong class="{correction_gate_state if correction_gate_state != 'pending' else 'partial'}">Correction {state_label(correction_gate_state)}</strong></header>
          <ol class="system-flow flow-row nodes-2">{''.join(correction_flow)}</ol></section>
      </div>
      <div class="feedback-return"><b>↩ Feedback loop</b><span>Feedback ResidualはPWMだけを補正。音楽的フィードフォワードはPosition Plannerが担当する。</span></div>
      <p class="flow-caveat">Motor PhysicsとLinear Fluteは決定論的な訓練環境。NNは内部の位置・速度・トルク・物理定数を受け取らず、World ModelはPWMと可聴音程の関係だけを学ぶ。ここで示す結果は内部シミュレーションであり、実機検証は {('済み' if physical_summary.get('real_rig_validated') else '未実施')}。</p></section>'''
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

    raw_e2e_cards = []
    for report_path in sorted((docs_root.parent / "runs").glob("yamabiko_e2e*_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        flags = [bool(value) for key, value in report.items() if key.endswith("_pass")]
        report_pass = bool(flags) and all(flags)
        report_status = "PASS" if report_pass else "FAIL"
        metrics = [(key, value) for key, value in report.items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
        metric_text = " / ".join(f"{html.escape(key.replace('_', ' '))} {n(value, 3)}" for key, value in metrics[:6])
        raw_e2e_cards.append(f'''<article class="attempt"><div class="stage"><b>E2E</b><span class="{report_status.lower()}">{report_status}</span></div>
        <h3>{html.escape(report_path.stem.removesuffix('_report'))}</h3><p class="metrics">{metric_text}</p>
        <p class="note">補助Gateの記録。最終演奏Gate 3は3試行すべてFAIL。</p></article>''')

    # Keep failed physical-control experiments visible.  A checked-in manifest may
    # carry richer artifact links; local run reports provide a useful fallback.
    physical_attempt_rows = list(physical.get("attempts", []))
    known_attempt_ids = {str(row.get("id")) for row in physical_attempt_rows}
    for report_path in sorted((docs_root.parent / "runs").glob("yamabiko_physical_control*_report.json")):
        attempt_id = report_path.stem.removesuffix("_report")
        if attempt_id in known_attempt_ids:
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        physical_attempt_rows.append({"id": attempt_id, "pass": report.get("pass", False),
                                      "reason": report.get("split", "physical simulation"),
                                      "metrics": report})

    physical_attempt_cards = []
    for row in physical_attempt_rows:
        metrics = row.get("metrics", row)
        row_pass = bool(row.get("pass", metrics.get("pass", False)))
        row_status = "PASS" if row_pass else "FAIL"
        metric_items = [(key, value) for key, value in metrics.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in ("seed",)]
        metric_text = " / ".join(f"{html.escape(key.replace('_', ' '))} {n(value, 3)}" for key, value in metric_items[:7])
        artifact = row.get("case", row.get("artifacts", {}))
        artifact_prefix = row.get("artifact_prefix", physical_prefix)
        artifact_listens = "".join(doc_audio(artifact_prefix + src, label) for src, label in (
            (metric_value(artifact, "target_wav", "reference_wav"), "目標"),
            (metric_value(artifact, "feedforward_wav", "base_wav"), "Feedforward"),
            (metric_value(artifact, "feedback_wav", "closed_wav"), "Feedback後"),
        ) if src)
        artifact_plot = figure(metric_value(artifact, "pitch_plot"),
                               "横軸: 時間 [s] / 縦軸: 音程 [cent] — 物理制御試行", artifact_prefix)
        physical_attempt_cards.append(f'''<article class="attempt"><div class="stage"><b>PHY</b><span class="{row_status.lower()}">{row_status}</span></div>
        <h3>{html.escape(str(row.get('id', 'physical-attempt')))}</h3><p>{html.escape(str(row.get('reason', 'physical simulation')))}</p>
        <p class="metrics">{metric_text or '数値なし'}</p>{f'<div class="listen">{artifact_listens}</div>' if artifact_listens else ''}{artifact_plot}
        <p class="note">シミュレーション評価。実機検証ではない。</p></article>''')
    physical_pass_count = sum(bool(row.get("pass", row.get("metrics", {}).get("pass", False))) for row in physical_attempt_rows)
    physical_history = (f'''<p class="history-count">{len(physical_attempt_rows)}試行 — PASS {physical_pass_count} / FAIL {len(physical_attempt_rows)-physical_pass_count}</p>
    <div class="attempt-grid raw-e2e-grid">{''.join(physical_attempt_cards)}</div>''' if physical_attempt_rows else
                        '<p>物理制御の試行記録はまだありません。</p>')

    composite_listens = "".join(doc_audio(composite_prefix + src, label) for src, label in (
        (composite_audio.get("reference"), "元のお手本"),
        (composite_audio.get("target"), "0.5倍速の演奏目標"),
        (composite_audio.get("neural"), "全NN接続後"),
        (composite_audio.get("deterministic"), "決定論的診断器"),
    ) if src)
    composite_status = "PASS" if composite_summary.get("pass") else "接続済み / 未合格"
    composite_state = "pass" if composite_summary.get("pass") else "fail"
    uno_q_takes = uno_q_report.get("takes", [])
    uno_q_take_text = " → ".join(f"{n(row.get('pitch_mae_cents'))}" for row in uno_q_takes)
    uno_q_card = f'''<section class="card {'pass' if uno_q_report.get('realtime_10ms_pass') else 'fail'}"><div class="stage"><b>UNO Q</b><span class="{'pass' if uno_q_report.get('realtime_10ms_pass') else 'fail'}">ONNX RUNTIME {'PASS' if uno_q_report.get('realtime_10ms_pass') else 'FAIL'}</span></div>
      <h2>UNO Q実機CPU + Hardware-in-the-loop</h2><p class="flow">ONNX Neural pipeline ↔ 決定論的Motor/Flute simulator（すべてUNO Q Linux上）</p>
      <dl><div><dt>お手本処理</dt><dd>{n(uno_q_report.get('listen_ms'))} ms / 曲ごと1回</dd></div>
      <div><dt>100 Hz制御 p99</dt><dd>{n(uno_q_report.get('step_p99_ms'), 3)} ms / 制限10 ms</dd></div>
      <div><dt>100 Hz制御 median</dt><dd>{n(uno_q_report.get('step_median_ms'), 3)} ms</dd></div>
      <div><dt>ピークRSS</dt><dd>{n(uno_q_report.get('peak_rss_kib') / 1024 if uno_q_report.get('peak_rss_kib') else None)} MiB</dd></div>
      <div><dt>ONNX容量</dt><dd>{n((uno_q_report.get('listen_onnx_bytes', 0) + uno_q_report.get('control_onnx_bytes', 0)) / 1048576, 2)} MiB</dd></div>
      <div><dt>反復MAE</dt><dd>{uno_q_take_text} cent</dd></div></dl>
      <p class="note">実測環境: {html.escape(str(uno_q_report.get('machine', 'unknown')))} / ONNX Runtime {html.escape(str(uno_q_report.get('onnxruntime', 'unknown')))}。マイク・モーターは使わず、物理装置だけを同じUNO Q内の決定論的シミュレータへ置換。したがって、これは実機CPU上の推論速度と接続のPASSであり、実モーター演奏の合格ではない。</p>
    </section>''' if uno_q_report else ''
    hybrid_sample_options = "".join(
        f'<option value="{html.escape(str(sample["id"]))}">{index:02d} {html.escape(str(sample["title"]))}</option>'
        for index, sample in enumerate(hybrid_samples, 1))
    first_hybrid_sample = hybrid_samples[0] if hybrid_samples else {}
    hybrid_lab_html = f'''<section class="hybrid-lab" id="hybrid-lab">
      <header><div><span>INTERACTIVE PIPELINE</span><h2>NN／決定論モジュール交換ラボ</h2></div><strong>{len(hybrid_samples)}サンプル × 各16構成</strong></header>
      <p>10種類のお手本を切り替え、各工程でNNか決定論モジュールを選ぶ。選んだ工程の出力が共通pitch／voice／position契約で次工程へ渡り、下流の音とグラフも作り直される。</p>
      <div class="hybrid-sample"><label><b>お手本サンプル</b><select data-hybrid-sample>{hybrid_sample_options}</select></label><span data-hybrid-description>{html.escape(str(first_hybrid_sample.get('description', '')))}</span><audio controls preload="none" data-hybrid-reference src="{html.escape(str(first_hybrid_sample.get('reference_wav', '')))}"></audio></div>
      <div class="hybrid-controls" aria-label="工程ごとの方式選択">
        <label><span>1 聴覚</span><select data-hybrid-stage="ear"><option value="nn">NN Neural Ear</option><option value="det">決定論 FFT Ear</option></select><small>raw波形 → 音程・発音</small></label><i>→</i>
        <label><span>2 記憶</span><select data-hybrid-stage="memory"><option value="nn">NN Timeline Memory</option><option value="det">決定論 Exact Memory</option></select><small>音程・発音 → 0.5倍Timeline</small></label><i>→</i>
        <label><span>3 計画</span><select data-hybrid-stage="planner"><option value="nn">NN Position Planner</option><option value="det">決定論 Linear Planner</option></select><small>Timeline → 位置</small></label><i>→</i>
        <label><span>4 制御</span><select data-hybrid-stage="control"><option value="nn">NN Controller＋Feedback</option><option value="det">決定論 PD＋PID</option></select><small>位置・自己音 → PWM・演奏</small></label>
      </div>
      <button type="button" class="hybrid-run">この構成で出力を作る</button>
      <div class="hybrid-live" aria-live="polite"><b data-hybrid-route></b><span>最終音程MAE <strong data-hybrid-mae>—</strong> cent</span><span>発音率 <strong data-hybrid-voice>—</strong></span></div>
      <canvas class="hybrid-chart" role="img" aria-label="選択したパイプラインの時間対音程グラフ"></canvas>
      <div class="hybrid-outputs">
        <section><b>1 聴覚の出力</b><small>音程・発音を次の記憶へ</small><audio controls preload="none" data-hybrid-audio="ear"></audio></section>
        <section><b>2 記憶の出力</b><small>0.5倍Timelineを次の計画へ</small><audio controls preload="none" data-hybrid-audio="memory"></audio></section>
        <section><b>3 計画の出力</b><small>位置を名目笛音へ変換した診断音</small><audio controls preload="none" data-hybrid-audio="planner"></audio></section>
        <section><b>4 最終演奏</b><small>選択した制御器による物理simulation音</small><audio controls preload="none" data-hybrid-audio="performance"></audio></section>
      </div>
      <p class="note">{html.escape(str(hybrid_lab.get('note', '')))} 音源は選択済み評価列からブラウザ内で合成する。これは自由な未評価組合せを推測した表示ではない。</p>
    </section>''' if hybrid_samples else ''
    composite_card = f'''<section class="card {composite_state}"><div class="stage"><b>ALL</b><span class="{composite_state}">{composite_status}</span></div>
      <h2>全工程 Connected Composite</h2>
      <p class="flow">お手本raw音声 → 記憶 → 0.5倍速計画 → PWM → 決定論的実変位・発音 → 自己音raw波形 → 同じNeural Ear → Comparator → Feedback</p>
      <div class="listen">{composite_listens}</div>
      {figure(composite.get('plot'), '横軸: 時間 [s] / 縦軸: 音程 [cent] — 目標・全NN接続・決定論的診断器', composite_prefix)}
      <dl><div><dt>Timeline出口 MAE</dt><dd>{n(composite_summary.get('timeline_pitch_mae_cents'))} cent</dd></div>
      <div><dt>全NN接続 演奏MAE</dt><dd>{n(composite_summary.get('neural_e2e_pitch_mae_cents'))} cent</dd></div>
      <div><dt>全NN 無音欠落率</dt><dd>{pct(composite_summary.get('neural_e2e_missing_voice_fraction'))} %</dd></div>
      <div><dt>決定論的診断器 MAE</dt><dd>{n(composite_summary.get('deterministic_e2e_pitch_mae_cents'))} cent</dd></div></dl>
      <p class="note"><b>接続の存在は確認できたが、性能合格ではない。</b> 凍結済みの各NNを1ケースで接続した結果で、結合学習・実機検証は未実施。Position Plannerと物理plantの較正差、Comparatorの学習分布差が残る。決定論版はplantの真値を読める診断用oracleで、実機へ搭載する制御器ではない。</p>
      <p class="note">各工程には決定論的な対応モジュール（FFT Ear、自己相関Tempo、完全Timeline Memory、線形Position、PD Motor Control、差分Comparator、PID Feedback）も実装し、NNの故障箇所を切り分けられる。</p>
    </section>''' if composite_summary else '<p>全工程Compositeの評価記録はまだありません。</p>'

    pipeline_tabs = f'''<section class="pipeline-lab" aria-labelledby="pipeline-tabs-title"><h2 id="pipeline-tabs-title">パイプライン別の結果と試行記録</h2>
    <p>採用予定と代替案を混在させず、同じパイプラインのフロー・結果・WAV・グラフ・失敗試行を一つのタブへまとめた。</p>
    <div class="tab-list" role="tablist" aria-label="パイプライン別結果">
      <button type="button" role="tab" id="tab-current" aria-controls="pipeline-current" aria-selected="true">採用予定 Timeline＋Physical <small>{len(physical_attempt_rows)}物理試行 / 実機未検証</small></button>
      <button type="button" role="tab" id="tab-clock" aria-controls="pipeline-clock" aria-selected="false" tabindex="-1">代替A Beat-cell＋Clock <small>26試行 / 接続FAIL</small></button>
      <button type="button" role="tab" id="tab-timing" aria-controls="pipeline-timing" aria-selected="false" tabindex="-1">代替B Duration／Timing <small>10試行 / 全体FAIL</small></button>
      <button type="button" role="tab" id="tab-legacy" aria-controls="pipeline-legacy" aria-selected="false" tabindex="-1">旧 Staged制御 <small>3系統 / 全体FAIL</small></button>
      <button type="button" role="tab" id="tab-raw-e2e" aria-controls="pipeline-raw-e2e" aria-selected="false" tabindex="-1">単一 Raw-audio E2E <small>7記録 / 演奏FAIL</small></button>
    </div>
    <section class="tab-panel" role="tabpanel" id="pipeline-current" aria-labelledby="tab-current">
      <header class="pipeline-summary adopted"><div><span>採用予定</span><h2>Beat-conditioned Timeline + Physical Control</h2></div><strong>Physical simulation {state_label(performance_gate_state)} / 実機未検証</strong></header>
      <div class="pipeline-explainer"><h3>この方式は何をしている？</h3>
        <p>お手本を「時間に沿った音程と休符の地図」として覚え、Position Plannerが先読み位置を作ります。これがフィードフォワードです。Motor ControllerはPWM履歴から作る潜在状態で位置を追い、実変位から合成した自己音をNeural Earで再び聴き、Feedback ResidualがPWMへ補正を足します。</p>
        <dl><div><dt>Feedforward</dt><dd>Position Planner（別Policyは置かない）</dd></div><div><dt>関係学習</dt><dd>PWM履歴 → 可聴音程のWorld Model</dd></div><div><dt>訓練環境</dt><dd>決定論的なtorque rise・摩擦・慣性 + 線形笛</dd></div><div><dt>検証範囲</dt><dd>内部simulationのみ / 実機未検証</dd></div></dl>
      </div>
      <p class="pipeline-route">raw audio → Neural Ear → Tempo/Beat → Timeline Memory → Position Planner (= Feedforward) → Motor Controller → Motor Physics → Linear Flute → deterministic waveform → Neural Ear (self) → Comparator → Adaptive Feedback ↩ PWM<br>学習時: PWM/audio → Motor Audio World Model → Motor Controller</p>
      {process_results}{hybrid_lab_html}<h2>全工程を実際に接続した結果</h2><div class="grid">{composite_card}{uno_q_card}</div><h2>知覚・記憶・Positionの試行履歴</h2>{current_history}
      <h2>Physical controlの試行履歴</h2><p>失敗試行も削除せず、モデルサイズ・学習方法の変更と結果を並べる。WAVとグラフがmanifestにある試行はカード内で再生・表示する。</p>{physical_history}
    </section>
    <section class="tab-panel" role="tabpanel" id="pipeline-clock" aria-labelledby="tab-clock">
      <header class="pipeline-summary alternative"><div><span>代替案 A</span><h2>Beat-cell + Neural Clock / Aligner</h2></div><strong>単独PASS / 接続FAIL</strong></header>
      <div class="pipeline-explainer"><h3>この方式は何をしている？</h3>
        <p>曲を16分音符のような小さな「拍のマス」に分け、各マスへ音程や休符を覚えます。演奏時には別のAI時計が今読むべきマスを決め、連続した音へ戻します。</p>
        <dl><div><dt>覚え方</dt><dd>拍ごとのセル</dd></div><div><dt>時間の扱い</dt><dd>Neural Clockが再生位置を進める</dd></div><div><dt>狙い</dt><dd>BPMを変えても同じ曲を再生</dd></div><div><dt>現在の課題</dt><dd>各AI単独では成功しても接続時に時刻がずれる</dd></div></dl>
      </div>
      <p class="pipeline-route">shared Ear/Tempo → Musical Memory cells → Neural Clock → Temporal Aligner → 100 Hz target</p>
      <p>Memory、Clock、AlignerはOracle条件の単独GateでPASSしたが、predicted-upstream接続では誤差が累積したため採用予定から外している。</p>
      {clock_history}
    </section>
    <section class="tab-panel" role="tabpanel" id="pipeline-timing" aria-labelledby="tab-timing">
      <header class="pipeline-summary rejected"><div><span>代替案 B</span><h2>Duration-conditioned / Stored Timing Profile</h2></div><strong>全体 FAIL</strong></header>
      <div class="pipeline-explainer"><h3>この方式は何をしている？</h3>
        <p>拍の内容だけでなく、「曲が何秒続くか」または「各時刻で何番目のマスを読むか」も一緒に覚える方式です。時計のずれを、参照時に保存した時間情報で抑えようとしました。</p>
        <dl><div><dt>覚え方</dt><dd>拍セル + 長さ／絶対位置</dd></div><div><dt>時間の扱い</dt><dd>Durationまたは保存Pointer</dd></div><div><dt>狙い</dt><dd>曲の終端と音符位置を正確に合わせる</dd></div><div><dt>現在の課題</dt><dd>Timing Profile自体と接続再生が未合格</dd></div></dl>
      </div>
      <p class="pipeline-route">Beat-cell memory → duration-conditioned clock または stored absolute pointer → Aligner</p>
      <p>演奏時間または絶対セル位置を記憶する案。Duration単独にはPASSがあるが接続はFAIL、Timing Profileは単独・接続ともFAIL。</p>
      {timing_history}
    </section>
    <section class="tab-panel" role="tabpanel" id="pipeline-legacy" aria-labelledby="tab-legacy">
      <header class="pipeline-summary legacy-summary"><div><span>旧方式</span><h2>Fixed 100 Hz Staged Control</h2></div><strong>全体 FAIL</strong></header>
      <div class="pipeline-explainer"><h3>この方式は何をしている？</h3>
        <p>BPMや拍を明示せず、お手本を最初から100 Hzの時間列として覚え、そのままモータ操作へ変換する初期方式です。構造は分かりやすい一方、テンポの理解と身体動作の誤差を切り分けにくい方式でした。</p>
        <dl><div><dt>覚え方</dt><dd>固定速度の100 Hz系列</dd></div><div><dt>時間の扱い</dt><dd>録音時の時刻をそのまま使用</dd></div><div><dt>狙い</dt><dd>短い経路でお手本から操作を生成</dd></div><div><dt>現在の課題</dt><dd>物理演奏と誤差比較がFAIL</dd></div></dl>
      </div>
      <p class="pipeline-route">raw audio → Neural Ear → Reference Memory → Feed-forward/Motor inverse → Plant → Comparator → Feedback</p>
      <div class="grid">{''.join(legacy_cards)}</div>
      <h2>旧Physical Feed-forward試行</h2><p>root checkpointと2つの派生試行を分けて保存。いずれも物理演奏GateはFAIL。</p>
      <div class="grid">{''.join(attempts) if attempts else '<p>派生試行なし</p>'}</div>
    </section>
    <section class="tab-panel" role="tabpanel" id="pipeline-raw-e2e" aria-labelledby="tab-raw-e2e">
      <header class="pipeline-summary research"><div><span>研究中</span><h2>Single Raw-audio E2EImitator</h2></div><strong>演奏 Gate 3: 0 / 3 PASS</strong></header>
      <div class="pipeline-explainer"><h3>この方式は何をしている？</h3>
        <p>お手本の音と自分の演奏音を一つの大きなAIへ直接入れ、途中の音符表現を人が決めずにモータ操作までまとめて学ばせる方式です。最も純粋なE2Eですが、どこで失敗したかを調べにくい難しさがあります。</p>
        <dl><div><dt>覚え方</dt><dd>GRU内部の連続した記憶</dd></div><div><dt>時間の扱い</dt><dd>Attentionと制御GRUが暗黙に学習</dd></div><div><dt>狙い</dt><dd>知覚・記憶・制御を一体で最適化</dd></div><div><dt>現在の課題</dt><dd>最終演奏誤差が大きく、反復適応も未達</dd></div></dl>
      </div>
      <p class="pipeline-route">reference raw + self raw → shared CNN / GRU memory / attention / control GRU → PWM・valve・done</p>
      <p>単一forward graphとして最もE2Eだが、演奏誤差237〜268 centで未合格。補助知覚Gateの成功と最終演奏成功を分けて表示する。</p>
      <div class="attempt-grid raw-e2e-grid">{''.join(raw_e2e_cards)}</div>
      <p><a href="e2e.md">単一E2Eの構造・複数take評価を読む</a></p>
    </section>
    </section>'''
    old = {"mae": 264.1299841855391, "corr": -0.995092265219766,
           "direction": 1.0, "explanation": "旧方向指標は300 ms以内の微小な符号一致で1.0となるため無効化"}
    verdict_detail = (f"Physical simulationはMotor Controller {state_label(controller_state)} / "
                      f"Feedback Residual {state_label(residual_state)}。実機検証は"
                      f"{'済み' if physical_summary.get('real_rig_validated') else '未実施'}。")
    hybrid_data_json = json.dumps(hybrid_lab, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/") if hybrid_lab else "null"
    doc = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yamabiko E2E 分離NN 学習レポート</title>
<style>
:root{{--bg:#081018;--panel:#101d29;--ink:#edf6fb;--muted:#9eb2c0;--cyan:#4ee1d1;--red:#ff7b72;--line:#284052}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}
body{{margin:0;background:radial-gradient(circle at 80% 0,#163047 0,transparent 42%),var(--bg);color:var(--ink);font:15px/1.65 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:48px 20px 80px}}h1{{font-size:clamp(2rem,5vw,4.5rem);line-height:1.02;margin:.2em 0}}h2{{margin:.2rem 0}}h3{{margin-top:2rem}}.eyebrow,.flow{{color:var(--cyan);letter-spacing:.05em}}.lead{{font-size:1.15rem;max-width:820px;color:#c7d6df}}
.verdict{{border:1px solid var(--red);background:#28171a;padding:18px 22px;border-radius:14px;margin:28px 0}}.verdict b{{color:var(--red)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,310px),1fr));gap:18px}}.card{{position:relative;background:linear-gradient(145deg,#132433,#0d1822);border:1px solid var(--line);border-radius:18px;padding:22px}}
.stage{{display:flex;justify-content:space-between;align-items:center}}.stage b{{display:grid;place-items:center;width:36px;height:36px;border-radius:50%;background:var(--cyan);color:#071016}}.stage span{{font-weight:800;color:var(--red)}}.card:has(.stage span:first-child){{border-color:var(--cyan)}}
.listen{{display:grid;gap:10px;margin:18px 0}}.audio{{display:grid;grid-template-columns:64px 1fr;align-items:center;gap:8px}}audio{{width:100%;height:36px}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}dl div{{background:#0a141d;padding:9px;border-radius:8px}}dt{{font-size:.72rem;color:var(--muted)}}dd{{margin:0;font-size:1.05rem}}.note{{color:var(--muted);font-size:.9rem}}
.model-explainer{{margin:14px 0;padding:14px;border-left:3px solid var(--cyan);border-radius:0 10px 10px 0;background:#0a1721}}.model-explainer h3{{margin:0 0 5px;color:var(--cyan);font-size:1rem}}.model-explainer p{{margin:.35rem 0;color:#d2e0e7;font-size:.92rem}}
.pitch-plot{{margin:14px 0 18px}}.pitch-plot img{{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#0b1620}}.pitch-plot figcaption{{margin-top:7px;color:var(--muted);font-size:.78rem}}.stage span.pass{{color:#70f0ac}}.stage span.fail{{color:#ff8c78}}.stage span.partial{{color:#ffc96b}}.stage span.pending{{color:#9eb2c0}}.legacy{{border-style:dashed;opacity:.86}}.pending{{border-color:#526675}}
.flow-overview{{margin:28px 0;padding:24px;background:#0b1620;border:1px solid var(--line);border-radius:18px}}.flow-heading{{display:flex;justify-content:space-between;gap:20px;align-items:start}}.flow-heading strong{{color:var(--red);border:1px solid var(--red);padding:8px 12px;border-radius:9px;white-space:nowrap}}.phase-flow{{display:grid;gap:0;margin:22px 0}}.flow-phase{{padding:16px;border:1px solid var(--line);border-radius:14px;background:#0d1923}}.flow-phase>header{{display:grid;grid-template-columns:42px minmax(0,1fr) auto;gap:12px;align-items:center}}.flow-phase>header>span{{display:grid;place-items:center;width:42px;height:42px;border-radius:50%;background:var(--cyan);color:#071016;font-size:1.1rem;font-weight:900}}.flow-phase>header div{{display:grid}}.flow-phase>header small{{color:var(--muted)}}.flow-phase>header strong{{padding:5px 9px;border-radius:8px;font-size:.8rem}}.flow-phase>header strong.pass{{color:#70f0ac;border:1px solid #46c987}}.flow-phase>header strong.fail{{color:#ff8c78;border:1px solid var(--red)}}.flow-phase>header strong.partial{{color:#ffc96b;border:1px solid #d69c3b}}.system-flow{{list-style:none;margin:14px 0 0;padding:0}}.flow-row{{display:grid;align-items:stretch;gap:8px}}.flow-row.nodes-4{{grid-template-columns:minmax(0,1fr) 42px minmax(0,1fr) 42px minmax(0,1fr) 42px minmax(0,1fr)}}.flow-row.nodes-3{{grid-template-columns:minmax(0,1fr) 54px minmax(0,1fr) 54px minmax(0,1fr)}}.flow-row.nodes-2{{grid-template-columns:minmax(0,1fr) 54px minmax(0,1fr)}}.flow-row.nodes-1{{grid-template-columns:minmax(0,1fr)}}.flow-node{{min-width:0;display:flex;flex-direction:column;border:2px solid var(--line);border-radius:12px;background:#101d29}}.flow-node>span,.flow-node>b,.flow-node>small{{margin-left:14px;margin-right:14px}}.flow-node>a{{display:flex;flex:1;flex-direction:column;gap:6px;padding:14px;color:inherit;text-decoration:none}}.flow-node>a span,.flow-node>span{{font-size:.72rem;font-weight:900}}.flow-node small{{color:var(--muted)}}.flow-node em{{margin-top:auto;padding-top:8px;color:var(--cyan);font-size:.75rem;font-style:normal}}.flow-node.pass{{border-color:#46c987}}.flow-node.pass span{{color:#70f0ac}}.flow-node.fail{{border-color:var(--red)}}.flow-node.fail span{{color:#ff8c78}}.flow-node.partial{{border-color:#d69c3b}}.flow-node.partial span{{color:#ffc96b}}.flow-node.pending,.flow-node.neutral{{border-color:#526675}}.flow-node.pending span,.flow-node.neutral span{{color:#b3c0c8}}.flow-arrow{{min-width:0;display:grid;place-items:center;align-content:center;text-align:center;color:var(--muted)}}.flow-arrow i{{font-size:1.6rem;font-style:normal}}.flow-arrow small{{font-size:.68rem}}.flow-arrow.fail,.phase-connector.fail{{color:var(--red)}}.flow-arrow.pass,.phase-connector.pass{{color:#70f0ac}}.flow-arrow.pending,.phase-connector.pending{{color:#9eb2c0}}.phase-connector{{display:flex;justify-content:center;gap:12px;align-items:center;min-height:58px;text-align:center;font-size:.85rem}}.phase-connector span{{font-size:1.6rem}}.feedback-return{{display:flex;gap:12px;align-items:center;border:1px dashed #526675;border-radius:10px;padding:10px 14px;color:var(--muted)}}.feedback-return b{{color:#9eb2c0}}.flow-caveat{{color:var(--muted);font-size:.88rem}}
.process-section{{margin:46px 0;scroll-margin-top:20px}}.process-section>header,.history-process>header{{display:flex;gap:14px;align-items:center;margin-bottom:16px}}.process-section>header>span,.history-process>header>b{{display:grid;place-items:center;flex:0 0 44px;height:44px;border-radius:12px;background:var(--cyan);color:#071016;font-size:1.15rem}}.process-section>header p,.history-process>header p{{margin:0;color:var(--muted)}}.history-process{{margin:26px 0;padding:18px;border-left:3px solid var(--line);background:#0a141d;border-radius:0 14px 14px 0}}
.pipeline-lab{{margin:42px 0}}.tab-list{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr));gap:8px;margin:18px 0}}.tab-list button{{min-height:58px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:#0b1620;color:var(--ink);font:inherit;font-weight:750;text-align:left;cursor:pointer}}.tab-list button small{{display:block;color:var(--muted);font-weight:500}}.tab-list button[aria-selected="true"]{{border-color:var(--cyan);background:#12303a;box-shadow:inset 0 -3px 0 var(--cyan)}}.tab-panel{{padding:22px;border:1px solid var(--line);border-radius:16px;background:#09131c;scroll-margin-top:18px}}.tab-panel[hidden]{{display:none}}.pipeline-summary{{display:flex;justify-content:space-between;gap:18px;align-items:start;padding-bottom:14px;border-bottom:1px solid var(--line)}}.pipeline-summary span{{color:var(--cyan);font-size:.78rem;font-weight:900;letter-spacing:.08em}}.pipeline-summary strong{{padding:7px 10px;border-radius:8px;white-space:nowrap}}.pipeline-summary.adopted strong{{color:#ffc96b;border:1px solid #d69c3b}}.pipeline-summary.alternative strong{{color:#ffc96b;border:1px solid #d69c3b}}.pipeline-summary.rejected strong,.pipeline-summary.legacy-summary strong,.pipeline-summary.research strong{{color:var(--red);border:1px solid var(--red)}}.pipeline-explainer{{margin:18px 0;padding:18px;border:1px solid #315065;border-radius:14px;background:linear-gradient(135deg,#102333,#0c1924)}}.pipeline-explainer h3{{margin:0 0 6px;color:var(--cyan)}}.pipeline-explainer>p{{margin:.3rem 0 1rem;font-size:1.02rem;color:#d8e5eb}}.pipeline-explainer dl{{margin:0}}.pipeline-explainer dd{{font-size:.9rem}}.pipeline-route{{padding:12px 14px;border-radius:10px;background:#101d29;color:var(--cyan);font-family:ui-monospace,monospace;overflow-wrap:anywhere}}.history-count{{color:var(--muted);font-weight:700}}.raw-e2e-grid{{padding:0}}
.hybrid-lab{{margin:46px 0;padding:22px;border:1px solid #315065;border-radius:16px;background:linear-gradient(145deg,#102333,#09131c)}}.hybrid-lab>header{{display:flex;justify-content:space-between;gap:18px;align-items:start;border-bottom:1px solid var(--line);padding-bottom:12px}}.hybrid-lab>header span{{color:var(--cyan);font-size:.75rem;font-weight:900;letter-spacing:.08em}}.hybrid-lab>header strong{{color:#70f0ac;border:1px solid #46c987;border-radius:8px;padding:6px 9px}}.hybrid-sample{{display:grid;grid-template-columns:auto minmax(140px,1fr) minmax(240px,1fr);align-items:center;gap:12px;margin:16px 0;padding:12px;background:#0a141d;border-radius:10px}}.hybrid-sample span{{color:var(--muted)}}.hybrid-controls{{display:grid;grid-template-columns:minmax(0,1fr) 24px minmax(0,1fr) 24px minmax(0,1fr) 24px minmax(0,1fr);gap:7px;align-items:stretch}}.hybrid-controls label{{display:flex;flex-direction:column;gap:6px;padding:12px;background:#0a141d;border:1px solid var(--line);border-radius:10px}}.hybrid-controls label>span{{font-weight:800;color:var(--cyan)}}.hybrid-controls label>small,.hybrid-outputs small{{color:var(--muted)}}.hybrid-controls select{{width:100%;padding:8px;border:1px solid #526675;border-radius:7px;background:#101d29;color:var(--ink);font:inherit}}.hybrid-controls>i{{display:grid;place-items:center;color:var(--cyan);font-size:1.4rem;font-style:normal}}.hybrid-run{{display:block;margin:16px 0 12px;padding:10px 16px;border:0;border-radius:9px;background:var(--cyan);color:#071016;font:inherit;font-weight:850;cursor:pointer}}.hybrid-live{{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:12px 0}}.hybrid-live>b{{color:var(--cyan);font-family:ui-monospace,monospace}}.hybrid-live span{{padding:5px 8px;background:#0a141d;border-radius:7px}}.hybrid-chart{{display:block;width:100%;height:310px;border:1px solid var(--line);border-radius:10px;background:#07111a}}.hybrid-outputs{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:12px 0}}.hybrid-outputs section{{display:grid;gap:5px;min-width:0;padding:10px;background:#0a141d;border-radius:10px}}.hybrid-outputs audio{{width:100%}}
.hybrid-sample{{grid-template-columns:minmax(180px,.8fr) minmax(180px,1fr) minmax(240px,1fr)}}.hybrid-sample label{{display:grid;gap:5px}}.hybrid-sample select{{width:100%;padding:8px;border:1px solid #526675;border-radius:7px;background:#101d29;color:var(--ink);font:inherit}}
.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.82rem}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}code{{color:var(--cyan)}}
details{{margin:12px 0;border:1px solid var(--line);border-radius:14px;background:#0b1620}}summary{{cursor:pointer;padding:14px 18px;color:var(--cyan);font-weight:700}}.attempt-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:12px;padding:0 12px 12px}}.attempt{{background:#101d29;border:1px solid var(--line);border-radius:12px;padding:14px}}.attempt h3{{font-size:.95rem;overflow-wrap:anywhere;margin:.5rem 0}}.attempt .metrics{{font-size:.78rem;color:#c7d6df}}summary:focus-visible,a:focus-visible,audio:focus-visible,.tab-list button:focus-visible{{outline:3px solid #ffc96b;outline-offset:3px}}
footer{{color:var(--muted);margin-top:50px;border-top:1px solid var(--line);padding-top:18px}}@media(max-width:860px){{main{{padding-top:28px}}dl{{grid-template-columns:1fr}}.flow-heading,.pipeline-summary,.hybrid-lab>header{{display:block}}.flow-heading strong,.pipeline-summary strong,.hybrid-lab>header strong{{display:inline-block;margin-top:8px;white-space:normal}}.flow-row.nodes-4,.flow-row.nodes-3,.flow-row.nodes-2,.flow-row.nodes-1,.hybrid-controls{{grid-template-columns:1fr}}.flow-arrow{{min-width:0;min-height:42px}}.flow-arrow i,.hybrid-controls>i{{transform:rotate(90deg)}}.flow-phase>header{{grid-template-columns:42px minmax(0,1fr)}}.flow-phase>header strong{{grid-column:1/-1;justify-self:start}}.audio,.hybrid-sample{{grid-template-columns:1fr}}.hybrid-outputs{{grid-template-columns:repeat(2,minmax(0,1fr))}}.tab-panel{{padding:14px}}}}@media(max-width:540px){{.hybrid-outputs{{grid-template-columns:1fr}}.hybrid-chart{{height:260px}}}}
</style></head><body><main>
<p class="eyebrow">PHYSICAL AI / OBJECTIVE GATED DEVELOPMENT</p><h1>お手本は、<br>本当に演奏になったか。</h1>
<p class="lead">単一の総合スコアで隠さず、聴覚・記憶・フィードフォワード・状態推定・モータ制御・誤差補正を分けて評価する。Position Plannerが音楽的フィードフォワードを担い、Motor PhysicsとLinear Fluteは学習用シミュレータとして実変位と音を作る。</p>
{flow_diagram}
<div class="verdict"><b>現在の総合判定: 未完成</b> — 全段がPASSし、さらに実機で再検証するまでE2E成功とは呼ばない。{html.escape(verdict_detail)}</div>
{pipeline_tabs}
<h2>評価を厳格化した理由</h2><p>旧一体モデルはMAE {old['mae']:.1f} cent、軌跡相関 {old['corr']:.3f} で実際には逆方向へ追従した。一方、旧方向指標だけは {old['direction']:.1f} だった。{old['explanation']}。新評価は発音できない目標区間へ1200 cent罰を与え、P90、発音率、休符漏れ、遷移ゲイン、整定誤差をケース別に残す。</p>
<h2>学習の分離と再統合</h2><p>Position Plannerを音楽的フィードフォワードとして固定し、PWMと可聴音程の関係だけを学ぶWorld Model、位置計画を追うMotor Controller、速い運動状態と遅い個体文脈を持つAdaptive Feedbackを順に学習する。物理真値は学習入力・教師へ漏らさず、評価グラフの原因分析だけに使う。</p>
<h2>今後の実装計画</h2>
<div class="grid">
  <section class="card">
    <div class="stage"><b>M</b><span>PLAN</span></div>
    <h2>交換可能な個別NNと統合E2E</h2>
    <p class="flow">modular learned / joint E2E / deterministic hybrid</p>
    <p>Ear、Tempo/Beat、Musical Memory、Aligner、Planner、World Model、Controller、Comparator、Feedbackに共通Tensor契約を定義する。開発時は個別checkpointと中間評価を維持し、配備時はWorld Modelを外し、ControllerとAdaptive Feedbackを単一のComposite graphへ統合する。</p>
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
<h2>全ケース詳細</h2>{''.join(details)}
<footer>seed {data['seed']} / checkpoint {html.escape(data['checkpoint'])}<br>{html.escape(data['sonification_note'])}</footer>
</main><script>
(() => {{
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const panels = tabs.map(tab => document.getElementById(tab.getAttribute('aria-controls')));
  if (!tabs.length) return;
  document.documentElement.classList.add('tabs-ready');
  const activate = (index, updateHash = false) => {{
    tabs.forEach((tab, i) => {{
      const selected = i === index;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      panels[i].hidden = !selected;
    }});
    if (updateHash) history.replaceState(null, '', '#' + panels[index].id);
  }};
  const indexForHash = () => {{
    const id = location.hash.slice(1);
    if (!id) return 0;
    const target = document.getElementById(id);
    const index = panels.findIndex(panel => panel.id === id || (target && panel.contains(target)));
    return index < 0 ? 0 : index;
  }};
  tabs.forEach((tab, index) => {{
    tab.addEventListener('click', () => activate(index, true));
    tab.addEventListener('keydown', event => {{
      let next = index;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else return;
      event.preventDefault(); activate(next, true); tabs[next].focus();
    }});
  }});
  const activateHash = () => {{
    const index = indexForHash(); activate(index, false);
    const target = document.getElementById(location.hash.slice(1));
    if (target && target !== panels[index]) requestAnimationFrame(() => target.scrollIntoView());
  }};
window.addEventListener('hashchange', activateHash); activateHash();
}})();
(() => {{
  const data = {hybrid_data_json};
  const root = document.getElementById('hybrid-lab');
  if (!root || !data) return;
  const stages = ['ear', 'memory', 'planner', 'control'];
  const samples = data.samples || [{{id:'legacy', title:data.sample || '4音ステップ',
    description:'従来の単一評価サンプル', reference_wav:'e2e-composite-results/reference.wav',
    combinations:data.combinations}}];
  const sampleSelect = root.querySelector('[data-hybrid-sample]');
  const referenceAudio = root.querySelector('[data-hybrid-reference]');
  const selects = Object.fromEntries(stages.map(name => [name, root.querySelector(`[data-hybrid-stage="${{name}}"]`)]));
  const audios = Object.fromEntries(stages.map((name, index) => [stages[index], root.querySelector(`[data-hybrid-audio="${{stages[index] === 'control' ? 'performance' : stages[index]}}"]`)]));
  const urls = {{}};
  let current = null;
  const label = {{nn:'NN', det:'決定論'}};

  function wavBlob(track) {{
    const sr = 16000, perFrame = 160, count = track.pitch_cents.length * perFrame;
    const buffer = new ArrayBuffer(44 + count * 2), view = new DataView(buffer);
    const text = (offset, value) => [...value].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
    text(0,'RIFF'); view.setUint32(4,36 + count*2,true); text(8,'WAVE'); text(12,'fmt ');
    view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
    view.setUint32(24,sr,true); view.setUint32(28,sr*2,true); view.setUint16(32,2,true); view.setUint16(34,16,true);
    text(36,'data'); view.setUint32(40,count*2,true);
    let phase = 0, cursor = 44;
    for (let frame=0; frame<track.pitch_cents.length; frame++) {{
      const frequency = 440 * Math.pow(2, track.pitch_cents[frame] / 1200);
      const step = 2 * Math.PI * frequency / sr, sounding = Boolean(track.voice[frame]);
      for (let i=0; i<perFrame; i++) {{
        const sample = sounding ? .22*Math.sin(phase) + .045*Math.sin(2*phase) : 0;
        view.setInt16(cursor, Math.max(-32767,Math.min(32767,Math.round(sample*32767))), true);
        cursor += 2; phase = (phase + step) % (2*Math.PI);
      }}
    }}
    return new Blob([buffer], {{type:'audio/wav'}});
  }}

  function updateAudio(name, track) {{
    if (urls[name]) URL.revokeObjectURL(urls[name]);
    urls[name] = URL.createObjectURL(wavBlob(track)); audios[name].src = urls[name];
  }}

  function draw(entry) {{
    const canvas = root.querySelector('.hybrid-chart'), rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    const dpr = window.devicePixelRatio || 1, width = rect.width, height = rect.height;
    canvas.width = Math.round(width*dpr); canvas.height = Math.round(height*dpr);
    const ctx = canvas.getContext('2d'); ctx.scale(dpr,dpr);
    const styles = getComputedStyle(document.documentElement), ink = styles.getPropertyValue('--ink').trim();
    const muted = styles.getPropertyValue('--muted').trim(), line = styles.getPropertyValue('--line').trim();
    const cyan = styles.getPropertyValue('--cyan').trim(), orange = '#ffc96b';
    ctx.clearRect(0,0,width,height); const m={{l:58,r:18,t:34,b:42}}, lo=600, hi=2000;
    const x = i => m.l + i*(width-m.l-m.r)/Math.max(1,entry.memory.pitch_cents.length-1);
    const y = value => m.t + (hi-value)*(height-m.t-m.b)/(hi-lo);
    ctx.font='12px system-ui'; ctx.fillStyle=muted; ctx.strokeStyle=line; ctx.lineWidth=1;
    [600,900,1200,1500,1800].forEach(v=>{{ctx.beginPath();ctx.moveTo(m.l,y(v));ctx.lineTo(width-m.r,y(v));ctx.stroke();ctx.fillText(String(v),8,y(v)+4);}});
    ctx.fillText('音程 [cent]',8,18); ctx.fillText(`時間 [s]  0 — ${{(entry.memory.pitch_cents.length/100).toFixed(1)}}`,m.l,height-12);
    const plot = (track,color,dash=[]) => {{ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash);ctx.beginPath();let open=false;track.pitch_cents.forEach((v,i)=>{{if(!track.voice[i]){{open=false;return;}}const px=x(i),py=y(v);if(!open){{ctx.moveTo(px,py);open=true;}}else ctx.lineTo(px,py);}});ctx.stroke();ctx.setLineDash([]);}};
    plot(entry.memory,cyan); plot(entry.performance,orange,[6,4]);
    ctx.fillStyle=cyan;ctx.fillText('— 記憶Timeline（制御目標）',m.l,19);ctx.fillStyle=orange;ctx.fillText('--- 最終演奏',m.l+205,19);
    ctx.strokeStyle=ink;ctx.strokeRect(m.l,m.t,width-m.l-m.r,height-m.t-m.b);
  }}

  function render() {{
    const choice = Object.fromEntries(stages.map(name => [name,selects[name].value]));
    const key = stages.map(name => choice[name]).join('-');
    const sample = samples.find(item => item.id === sampleSelect.value) || samples[0];
    let entry;
    if (sample.outputs) {{
      const memoryKey = `${{choice.ear}}-${{choice.memory}}`;
      const plannerKey = `${{memoryKey}}-${{choice.planner}}`;
      const performance = sample.outputs.performance[key];
      entry = {{ear:sample.outputs.ear[choice.ear], memory:sample.outputs.memory[memoryKey],
        planner:sample.outputs.planner[plannerKey], performance, metrics:performance.metrics}};
    }} else {{
      entry = sample.combinations[key];
    }}
    if (!entry) return;
    current = entry;
    root.querySelector('[data-hybrid-description]').textContent = sample.description || '';
    if (referenceAudio.getAttribute('src') !== sample.reference_wav) referenceAudio.src = sample.reference_wav;
    root.querySelector('[data-hybrid-route]').textContent = stages.map(name => label[choice[name]]).join(' → ');
    root.querySelector('[data-hybrid-mae]').textContent = entry.metrics.final_pitch_mae_cents == null ? '—' : entry.metrics.final_pitch_mae_cents.toFixed(1);
    root.querySelector('[data-hybrid-voice]').textContent = (100*entry.metrics.voiced_fraction).toFixed(1) + '%';
    updateAudio('ear', entry.ear); updateAudio('memory', entry.memory);
    updateAudio('planner', {{pitch_cents:entry.planner.pitch_cents || entry.planner.nominal_cents,
                            voice:entry.planner.voice}});
    updateAudio('control', entry.performance); draw(entry);
  }}
  sampleSelect.addEventListener('change', render);
  stages.forEach(name => selects[name].addEventListener('change', render));
  root.querySelector('.hybrid-run').addEventListener('click', render);
  window.addEventListener('resize', () => current && draw(current)); render();
}})();
</script></body></html>'''
    path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(doc, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__": main()
