"""Write the target and the robot's imitation as WAV files, for listening to trained policies.

Every model plays the same few pieces on the same simulated rigs (fixed
seeds), so snapshots from different generations can be compared by ear. The
target is rendered with a recorder-like timbre at the flute's own pitch; the
imitation is the robot's flute synthesised from the simulator trace, one
file per take. A manifest.json lists the files with each take's error.

    python scripts/export_imitation_audio.py --out runs/history_audio/gen050 \
        --model "記憶なし MLP=runs/snapshots/fb_mlp_gen050.npz" --model "GRU=runs/snapshots/fb_gru_gen050.npz"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from export_targets import write_wav  # noqa: E402

from flute_rl import FluteEnv, PhysicsPriorPolicy, rollout  # noqa: E402
from flute_rl.audio import synth_self, synth_source  # noqa: E402
from flute_rl.feedback import FeedbackPolicy, FeedbackResidualPolicy  # noqa: E402
from flute_rl.policy import load_net  # noqa: E402
from flute_rl.targets import bird_song, make_target  # noqa: E402

SR = 16000
PIECES = [
    ("legato", "つなげる曲(スラー・タイ)", lambda: make_target(np.random.default_rng(11), 3, articulation="legato")),
    ("detached", "切る曲(音の間に短い無音)", lambda: make_target(np.random.default_rng(12), 3, articulation="detached")),
    ("uguisu", "ウグイス(ホーホケキョ)", lambda: bird_song(np.random.default_rng(13), "uguisu")),
]
RIG_SEED = 4_000_000


def policy_for(path: str | None):
    if path == "prior":  # physics model only: no rig identification, no ILC, no listening
        return PhysicsPriorPolicy()
    if path is None:
        return FeedbackPolicy()
    net, meta = load_net(path)
    base = FeedbackPolicy(fb_gain=float(meta["fb_gain"]), ilc_gain=float(meta["ilc_gain"]))
    return FeedbackResidualPolicy(base, net, scale=float(meta["scale"]), horizon=int(meta["horizon"]),
                                  history=int(meta.get("history", 0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[],
                    help='"label=path.npz" (repeatable); "label=classic" for the hand-made feedback, "label=prior" for the physics model only')
    ap.add_argument("--out", required=True)
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--harsh", type=float, default=0.0, help="0..1: rig effects the controllers do not model")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"pieces": [], "models": []}
    targets = []
    for i, (pid, name, make) in enumerate(PIECES):
        tgt = make()
        targets.append(tgt)
        y, _ = synth_source(tgt, "recorder", np.random.default_rng(100 + i), sr=SR, shift=0.0)
        write_wav(out / f"target_{pid}.wav", y, SR)
        manifest["pieces"].append({"id": pid, "name": name, "target": f"target_{pid}.wav", "seconds": round(len(tgt) * 0.01, 2)})

    for spec in args.model:
        label, path = spec.split("=", 1)
        entry = {"label": label, "path": path, "pieces": []}
        for i, (pid, _, _) in enumerate(PIECES):
            env = FluteEnv(takes=args.takes, feedback=True, harsh=args.harsh)
            r = rollout(env, policy_for(None if path == "classic" else path), seed=RIG_SEED + i,
                        options={"target": targets[i]})
            takes = []
            for k in range(args.takes):
                m = r["take"] == k
                y = synth_self(r["cents"][m], r["sounding"][m], np.random.default_rng(200 + i), sr=SR)
                slug = "".join(ch if ch.isalnum() else "_" for ch in path.split("/")[-1].replace(".npz", ""))
                fname = f"{slug}_{pid}_take{k + 1}.wav"
                write_wav(out / fname, y, SR)
                pt = r["per_take"][k]
                takes.append({"wav": fname, "mean_abs_cents": round(float(pt["mean_abs_cents"]), 1),
                              "gap_leak": None if not np.isfinite(pt["gap_leak"]) else round(float(pt["gap_leak"]), 3),
                              "sounding_rate": round(float(pt["sounding_rate"]), 3)})
            entry["pieces"].append({"id": pid, "takes": takes})
        manifest["models"].append(entry)
        print(label, [[t["mean_abs_cents"] for t in p["takes"]] for p in entry["pieces"]], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(list(out.glob('*.wav')))} WAV files and manifest.json to {out}")


if __name__ == "__main__":
    main()
