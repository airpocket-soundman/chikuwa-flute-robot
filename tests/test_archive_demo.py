import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import archive_demo  # noqa: E402
import archive_fetch  # noqa: E402

ROW = """<tr >
<td><audio src='x.mp3'></audio></td>
<td><span class="common-name"><a href='https://xeno-canto.org/species/Parus-minor'>Japanese Tit</a></span>
 <span class='sci-name'>Parus minor</span> </td>
<td> 0:36</td>
<td> <a href='https://xeno-canto.org/contributor/X'>Someone</a></td>
<td>2015-05-20</td>
<td> 10:00</td>
<td>Japan</td>
<td><a href="https://xeno-canto.org/location/map?x">Mount Fuji east side</a></td><td></td><td>song</td>
<td><a href='https://xeno-canto.org/285813/download'>dl</a></td>
<td><a href="https://xeno-canto.org/285813">XC285813 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">cc</a></td>
</tr>"""


def test_parse_page_reads_a_row():
    rows = archive_fetch.parse_page(ROW)
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == 285813 and r["species"] == "Parus minor" and r["type"] == "song"
    assert r["recordist"] == "Someone" and r["country"] == "Japan"
    assert r["licence"].endswith("/by-nc-nd/4.0/")


def test_fit_compresses_a_wide_phrase_and_stretches_time():
    c = np.r_[np.linspace(0, 2400, 50), np.full(10, np.nan), np.linspace(2400, 0, 40)]
    fitted, scale = archive_demo.fit(c, -10, 1200, "compress", 2.0)
    assert len(fitted) == 2 * len(c)
    assert 0.3 < scale < 0.5
    span = np.nanpercentile(fitted, 98) - np.nanpercentile(fitted, 2)
    assert span <= 1210 * .9 + 1
    assert np.isnan(fitted).sum() > 15                          # the gap is kept, twice as long


def test_listen_follows_a_sweep():
    sr = 16000
    t = np.arange(int(sr * .8)) / sr
    f = np.linspace(1000, 2000, len(t))
    x = .5 * np.sin(2 * np.pi * np.cumsum(f) / sr)
    raw, clean = archive_demo.listen(x, sr, "silbo")
    v = np.isfinite(clean)
    assert v.mean() > .8
    hz = 440 * 2 ** (clean[v] / 1200)
    assert abs(hz[2] - 1000) < 80 and abs(hz[-3] - 2000) < 120
    assert len(archive_demo.synth(clean, v)) == len(clean) * 160
