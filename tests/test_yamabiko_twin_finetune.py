import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.residual_performer import ResidualConfig, ResidualTwinPerformer  # noqa: E402
from yamabiko_twin_finetune import SCALED, load_twin, rigs_around  # noqa: E402


def twin_json(path, plant):
    """A twin file shaped like yamabiko_twin_fit.py's report."""
    params = plant.parameters(1, "cpu", spread=1.0, generator=torch.Generator().manual_seed(5))
    params.hearing_delay_steps = torch.tensor([2])  # off by one either way stays at or above 0
    values = {name: float(value[0]) for name, value in vars(params).items() if value is not None}
    path.write_text(json.dumps({"twin": values, "temp_c": 24.0}), encoding="utf-8")
    return params


def test_rigs_stay_around_the_twin(tmp_path):
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    params = twin_json(tmp_path / "twin.json", plant)
    twin, temp_c = load_twin(tmp_path / "twin.json", "cpu")
    assert temp_c == 24.0 and twin.hearing_delay_steps.dtype == torch.long
    rigs = rigs_around(twin, 256, .12, torch.Generator().manual_seed(0))
    for name in SCALED:
        ratio = getattr(rigs, name) / getattr(params, name)
        assert ratio.min() >= .88 - 1e-5 and ratio.max() <= 1.12 + 1e-5 and ratio.std() > .03
    off = (rigs.hearing_delay_steps - params.hearing_delay_steps).abs()
    assert off.max() <= 1 and .15 < (off == 1).float().mean() < .45


def test_finetune_writes_a_runtime_never_worse_than_the_start(tmp_path):
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    twin_json(tmp_path / "twin.json", plant)
    torch.manual_seed(0)
    torch.save(ResidualTwinPerformer(plant, ResidualConfig(hidden=16)).checkpoint(), tmp_path / "start.pt")
    out = tmp_path / "tuned"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "yamabiko_twin_finetune.py"), str(tmp_path / "twin.json"),
                    "--out", str(out), "--checkpoint", str(tmp_path / "start.pt"), "--steps", "2", "--batch", "2",
                    "--song-steps", "40", "--eval-every", "1", "--eval-rigs", "2", "--device", "cpu"],
                   check=True, cwd=ROOT)
    report = json.loads((tmp_path / "tuned.json").read_text(encoding="utf-8"))
    assert report["after"]["play2"] <= report["before"]["play2"]
    runtime = np.load(str(out) + "_runtime.npz")
    assert "cell.w_ih" in runtime.files
    assert torch.load(str(out) + ".pt", weights_only=False)["format"] == "yamabiko-residual-twin-v1"
