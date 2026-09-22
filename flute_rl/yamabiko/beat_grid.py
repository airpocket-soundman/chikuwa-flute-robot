"""Learned metrical bottleneck for raw-audio imitation.

The demonstration is first mapped to a tempo/beat phase, then to a sequence
of continuous-pitch musical cells.  The cells are not MIDI notes: they retain
cent-valued pitch, rests, attacks and within-cell slope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BeatGridConfig:
    input_dim: int = 66
    beat_hidden: int = 96
    memory_hidden: int = 128
    subdivision: int = 4
    bpm_center: float = 120.0
    bpm_log_scale: float = 0.5


class TempoBeatNet(nn.Module):
    """Full-reference NN producing global BPM and beat phase at 100 Hz."""

    def __init__(self, config: BeatGridConfig):
        super().__init__(); self.config = config
        self.encoder = nn.GRU(config.input_dim, config.beat_hidden, batch_first=True, bidirectional=True)
        self.phase = nn.Sequential(nn.Linear(2 * config.beat_hidden, config.beat_hidden), nn.SiLU(),
                                   nn.Linear(config.beat_hidden, 3))
        self.tempo = nn.Sequential(nn.Linear(2 * config.beat_hidden, config.beat_hidden), nn.SiLU(),
                                   nn.Linear(config.beat_hidden, 1))

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        packed = nn.utils.rnn.pack_padded_sequence(features, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True,
                                                       total_length=features.shape[1])
        mask = torch.arange(features.shape[1], device=features.device)[None] < lengths[:, None]
        pooled = (encoded * mask[..., None]).sum(1) / lengths[:, None].clamp_min(1)
        bpm_log = self.tempo(pooled)[:, 0]
        phase = self.phase(encoded)
        # Normalising the two-vector makes phase decoding stable without a
        # non-differentiable angle in the model itself.
        phase_xy = phase[..., :2] / phase[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return bpm_log, torch.cat([phase_xy, phase[..., 2:3]], -1), encoded, mask

    def bpm(self, normalized_log_bpm: torch.Tensor) -> torch.Tensor:
        c = self.config
        return c.bpm_center * torch.exp(c.bpm_log_scale * normalized_log_bpm)

    def normalized_bpm(self, bpm: torch.Tensor) -> torch.Tensor:
        c = self.config
        return torch.log(bpm / c.bpm_center) / c.bpm_log_scale

    @staticmethod
    def phase_bpm(phase_outputs: torch.Tensor, valid_mask: torch.Tensor, gap: int = 10) -> torch.Tensor:
        """Derive tempo from the NN's continuous phase trajectory.

        This is clock arithmetic over a learned representation, not an audio
        pitch/onset algorithm.  A median rejects local phase glitches.
        """
        p0, p1 = phase_outputs[:, :-gap, :2], phase_outputs[:, gap:, :2]
        dot = (p0 * p1).sum(-1)
        cross = p1[..., 0] * p0[..., 1] - p1[..., 1] * p0[..., 0]
        angle = torch.remainder(torch.atan2(cross, dot), 2 * torch.pi)
        pair = valid_mask[:, :-gap] & valid_mask[:, gap:]
        values = []
        for row in range(len(angle)):
            good = angle[row][pair[row]]
            values.append(torch.median(good) if good.numel() else torch.zeros((), device=angle.device))
        radians = torch.stack(values)
        return (radians / (2 * torch.pi) * 60.0 / (gap / 100.0)).clamp(40.0, 240.0)


class MusicalMemoryNet(nn.Module):
    """Decode continuous-pitch/rest cells from beat-aligned reference memory."""

    OUTPUTS = 5  # pitch, voice logit, onset logit, offset logit, pitch slope

    def __init__(self, config: BeatGridConfig):
        super().__init__(); self.config = config
        source_dim = config.input_dim + 2 * config.beat_hidden + 3
        self.context = nn.GRU(source_dim, config.memory_hidden, batch_first=True, bidirectional=True)
        query_dim = 8
        self.query = nn.Sequential(nn.Linear(query_dim, config.memory_hidden), nn.SiLU(),
                                   nn.Linear(config.memory_hidden, 2 * config.memory_hidden))
        self.key = nn.Linear(2 * config.memory_hidden, 2 * config.memory_hidden, bias=False)
        self.decoder = nn.Sequential(nn.Linear(4 * config.memory_hidden, config.memory_hidden), nn.SiLU(),
                                     nn.Linear(config.memory_hidden, self.OUTPUTS))

    @staticmethod
    def cell_queries(batch: int, cells: int, device, dtype, lengths: torch.Tensor | None = None):
        index = torch.arange(cells, device=device, dtype=dtype)[None].expand(batch, -1)
        denom = (torch.full((batch, 1), max(cells - 1, 1), device=device, dtype=dtype)
                 if lengths is None else (lengths[:, None].to(dtype) - 1).clamp_min(1))
        u = index / denom
        return torch.stack([u, 1 - u, torch.sin(2 * torch.pi * u), torch.cos(2 * torch.pi * u),
                            torch.sin(4 * torch.pi * u), torch.cos(4 * torch.pi * u),
                            torch.sin(8 * torch.pi * u), torch.cos(8 * torch.pi * u)], -1)

    def forward(self, audio_features: torch.Tensor, beat_encoded: torch.Tensor,
                beat_outputs: torch.Tensor, frame_mask: torch.Tensor, cells: int,
                cell_lengths: torch.Tensor | None = None):
        source = torch.cat([audio_features, beat_encoded, beat_outputs], -1)
        contextual, _ = self.context(source)
        q = self.query(self.cell_queries(len(source), cells, source.device, source.dtype, cell_lengths))
        k = self.key(contextual)
        score = torch.einsum("bcd,btd->bct", q, k) / (k.shape[-1] ** .5)
        score = score.masked_fill(~frame_mask[:, None], torch.finfo(score.dtype).min)
        attention = torch.softmax(score, -1)
        attended = torch.einsum("bct,btd->bcd", attention, contextual)
        return self.decoder(torch.cat([q, attended], -1)), attention


class BeatAlignedMusicalMemoryNet(nn.Module):
    """Soft-resample neural audio memory onto the NN-predicted beat grid."""

    def __init__(self, config: BeatGridConfig, pitch_skip: bool = True, sharp_alignment: bool = True,
                 stable_clock: bool = True, phase_lock: bool = True,
                 alignment_sigma: float | None = None, pitch_residual_scale: float = .1,
                 wrap_clock: bool = False):
        super().__init__(); self.config = config; self.pitch_skip = pitch_skip
        self.sharp_alignment = sharp_alignment; self.stable_clock = stable_clock; self.phase_lock = phase_lock
        self.alignment_sigma = alignment_sigma; self.pitch_residual_scale = pitch_residual_scale
        self.wrap_clock = wrap_clock
        source_dim = config.input_dim + 2 * config.beat_hidden + 3
        self.source = nn.Sequential(nn.Linear(source_dim, config.memory_hidden), nn.SiLU(),
                                    nn.Linear(config.memory_hidden, config.memory_hidden))
        self.cells = nn.GRU(config.memory_hidden, config.memory_hidden, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(nn.Linear(2 * config.memory_hidden, config.memory_hidden), nn.SiLU(),
                                     nn.Linear(config.memory_hidden, MusicalMemoryNet.OUTPUTS))

    def forward(self, audio_features: torch.Tensor, beat_encoded: torch.Tensor,
                beat_outputs: torch.Tensor, frame_mask: torch.Tensor, cells: int,
                cell_lengths: torch.Tensor | None = None):
        source = self.source(torch.cat([audio_features, beat_encoded, beat_outputs], -1))
        xy = beat_outputs[..., :2]
        dot = (xy[:, 1:] * xy[:, :-1]).sum(-1)
        cross = xy[:, 1:, 0] * xy[:, :-1, 1] - xy[:, 1:, 1] * xy[:, :-1, 0]
        delta = torch.remainder(torch.atan2(cross, dot), 2 * torch.pi) / (2 * torch.pi)
        # Reject negative/noisy near-full-cycle jumps; at 100 Hz a physical
        # beat advances far less than half a cycle per frame.
        delta = torch.where(delta < .5, delta, torch.zeros_like(delta))
        confidence = torch.sigmoid(beat_outputs[..., 2])
        valid = frame_mask & (confidence >= .5)
        valid_delta = valid[:, 1:] & valid[:, :-1]
        if self.wrap_clock:
            raw_phase = torch.remainder(torch.atan2(xy[..., 0], xy[..., 1]) / (2 * torch.pi), 1.0)
            wraps = ((raw_phase[:, 1:] < .30) & (raw_phase[:, :-1] > .70) & valid_delta).to(source.dtype)
            count = torch.cat([torch.zeros(len(source), 1, device=source.device), torch.cumsum(wraps, 1)], 1)
            first = valid.float().argmax(1); row = torch.arange(len(source), device=source.device)
            phase0 = raw_phase[row, first]
            beat_position = count + raw_phase - phase0[:, None]
        elif self.phase_lock:
            bpm = TempoBeatNet.phase_bpm(beat_outputs, valid)
            raw_phase = torch.remainder(torch.atan2(xy[..., 0], xy[..., 1]) / (2 * torch.pi), 1.0)
            first = valid.float().argmax(1); row = torch.arange(len(source), device=source.device)
            current = raw_phase[row, first]; positions = []
            for t in range(source.shape[1]):
                expected = current + bpm / 6000.0
                candidate = raw_phase[:, t] + torch.round(expected - raw_phase[:, t])
                current = torch.where(valid[:, t], candidate, current)
                positions.append(current)
            beat_position = torch.stack(positions, 1)
        elif self.stable_clock:
            bpm = TempoBeatNet.phase_bpm(beat_outputs, valid)
            first = valid.float().argmax(1)
            row = torch.arange(len(source), device=source.device)
            phase0 = torch.remainder(torch.atan2(xy[row, first, 0], xy[row, first, 1]) / (2 * torch.pi), 1.0)
            time = torch.arange(source.shape[1], device=source.device)[None] - first[:, None]
            beat_position = phase0[:, None] + time * bpm[:, None] / 6000.0
        else:
            delta = delta * valid_delta
            beat_position = torch.cat([torch.zeros(len(source), 1, device=source.device), torch.cumsum(delta, 1)], 1)
        cell_position = beat_position * self.config.subdivision
        centers = torch.arange(cells, device=source.device, dtype=source.dtype)[None, :, None] + .5
        distance = cell_position[:, None, :] - centers
        sigma = self.alignment_sigma if self.alignment_sigma is not None else (.30 if self.sharp_alignment else .55)
        weights = torch.exp(-.5 * (distance / sigma) ** 2) * valid[:, None]
        if cell_lengths is not None:
            cell_mask = torch.arange(cells, device=source.device)[None] < cell_lengths[:, None]
            weights = weights * cell_mask[..., None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bct,btd->bcd", weights, source)
        contextual, _ = self.cells(pooled)
        decoded = self.decoder(contextual)
        # Preserve the calibrated Neural Ear pitch through the metrical
        # bottleneck.  The decoder learns only a bounded contextual residual;
        # it cannot replace cell alignment with an unrelated shortcut.
        if self.pitch_skip:
            pitch_weights = weights * torch.sigmoid(audio_features[..., -1])[:, None]
            pitch_weights = pitch_weights / pitch_weights.sum(-1, keepdim=True).clamp_min(1e-6)
            pooled_ear_pitch = torch.einsum("bct,bt->bc", pitch_weights, audio_features[..., -2])
            decoded = torch.cat([pooled_ear_pitch[..., None] + self.pitch_residual_scale * decoded[..., :1],
                                 decoded[..., 1:]], -1)
        return decoded, weights


def checkpoint(config: BeatGridConfig, tempo: TempoBeatNet, memory: MusicalMemoryNet, **metadata):
    return {"format": "yamabiko-beat-grid-v1", "config": asdict(config),
            "tempo_beat": tempo.state_dict(), "musical_memory": memory.state_dict(), **metadata}
