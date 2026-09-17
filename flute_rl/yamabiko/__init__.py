"""Yamabiko No.1: encoder-less plunger control that uses the flute's own pitch as the position sensor.

See docs/yamabiko.md. Separate from the earlier edge-blown rig (flute_rl.sim / env), which
is kept unchanged so the results in the README stay reproducible.
"""
from .control import Encoder, ExternalFit, GRUPolicy, Observer, OpenLoop, Oracle
from .learn import TaskSpec, fitness
from .metrics import song_metrics
from .rig import OVERBLOW_CENTS, Rig, RigParams
from .session import Schedule, Swap, draw_pieces, make_schedule, run_session

__all__ = [
    "Encoder", "ExternalFit", "GRUPolicy", "OVERBLOW_CENTS", "Observer", "OpenLoop", "Oracle", "Rig", "RigParams",
    "Schedule", "Swap", "TaskSpec", "draw_pieces", "fitness", "make_schedule", "run_session", "song_metrics",
]
