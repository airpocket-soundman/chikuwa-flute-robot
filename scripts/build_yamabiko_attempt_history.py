"""Collect every beat-grid development report into a committed Pages manifest."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs/e2e-beat-results/attempt-history.json"

GROUPS = (
    ("Tempo / Beat", "yamabiko_tempo_beat"),
    ("Musical Memory", "yamabiko_musical_memory"),
    ("Temporal Aligner", "yamabiko_temporal_aligner"),
    ("Connected Temporal", "yamabiko_temporal_connected"),
    ("Duration Clock", "yamabiko_temporal_duration"),
    ("Timing Profile", "yamabiko_timing_profile"),
    ("Timeline Memory", "yamabiko_timeline_memory"),
    ("Timeline Memory", "yamabiko_timeline_timbre"),
    ("Position Planner", "yamabiko_timeline_position"),
)

REASONS = {
    "aligned_v4": "固定BPM clockの累積ドリフト", "aligned_v5": "PLLが局所位相誤差を増幅",
    "temporal_aligner_v4": "鋭い区間attentionでClockまで崩壊", "temporal_aligner_v6": "平坦なevent教師でonsetピークが曖昧",
    "connected": "predicted-upstream接続で時刻誤差が累積", "duration_v1": "参照長追加だけではClock未収束",
    "timing_profile_direct": "絶対位置直接回帰が未知曲へ一般化せず", "timing_profile_indexed": "frame index追加後も積分誤差が残存",
    "timeline_memory_direct": "Ear skip除去でhum音源の絶対音程が悪化", "timeline_memory_final": "未知音源で音程一般化不足",
    "timeline_memory_v2": "音程MAEは改善したが軌跡相関不足", "timeline_memory_v3": "合格閾値直前だが未達",
    "timbre_v2": "音色再学習系。セルMemoryは失敗、Timelineは改善", "position": "単独とpredicted-upstream結合を分離評価",
}


def stage_for(stem):
    for stage, prefix in GROUPS:
        if stem.startswith(prefix): return stage
    return "Other"


def reason_for(stem, passed):
    if passed: return "採用候補または合格checkpoint"
    if stem.startswith("yamabiko_timeline_position_v3"):
        return "同一フレーム分解でPlanner自身の実入力変換誤差が大きくFAIL"
    for key, reason in REASONS.items():
        if key in stem: return reason
    return "固定Gateの少なくとも1指標を満たさず不採用"


def main():
    reports = []
    prefixes = tuple(prefix for _, prefix in GROUPS)
    for path in sorted(RUNS.glob("*.json")):
        stem = path.stem.removesuffix("_report")
        if not stem.startswith(prefixes): continue
        try: data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        passed = bool(data.get("pass", False))
        metrics = {k: v for k, v in data.items() if isinstance(v, (int, float, bool)) and k not in ("pass", "seed")}
        audio_manifest = ROOT / "docs/e2e-beat-attempts" / stem / "manifest.json"
        reports.append({"id": stem, "stage": stage_for(stem), "pass": passed,
                        "split": data.get("split", "development/legacy"), "seed": data.get("seed"),
                        "reason": reason_for(stem, passed), "metrics": metrics,
                        "audio_manifest": str(audio_manifest.relative_to(ROOT / "docs")).replace("\\", "/") if audio_manifest.exists() else None})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"format": "yamabiko-attempt-history-v1", "attempts": reports}, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(reports)} attempts)")


if __name__ == "__main__": main()
