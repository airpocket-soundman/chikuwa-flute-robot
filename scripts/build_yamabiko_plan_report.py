"""Build the plan page (docs/yamabiko-plan.html) from the plan benchmark.

Only the current plan (fit an unknown rig -> digital twin -> rig-specific
control -> improve by repeating a song) is shown, with the deterministic
comparison and an interactive simulation.  Numbers come from
docs/plan-results/manifest.json; nothing is written by hand.
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib

CONTROLLERS = {
    "det_nominal": ("決定論（フィッティングなし）", "#9eb2c0"),
    "det_twin": ("決定論＋ツイン", "#ffc96b"),
    "det_true": ("決定論＋真の値（理想の当てはめ）", "#ff8c78"),
    "nn_play1": ("NN 1回目", "#7aa7ff"),
    "nn_play2": ("NN 2回目", "#5cc8ff"),
    "nn_play3": ("NN 3回目", "#70f0ac"),
    "sensor": ("位置センサー付き（参考）", "#d6a3ff"),
}


def pct(summary, key="hit_rate"):
    if not summary or summary.get(key, {}).get("mean") is None:
        return "—"
    v = summary[key]
    return f"{v['mean'] * 100:.0f} %<small> [{v['low'] * 100:.0f}–{v['high'] * 100:.0f}]</small>"


def verdict(state, text):
    return f'<span class="state {state}">{text}</span>'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="docs/plan-results/manifest.json")
    ap.add_argument("--lab", default="docs/plan-results/lab.json")
    ap.add_argument("--out", default="docs/yamabiko-plan.html")
    args = ap.parse_args()
    manifest_path, lab_path = pathlib.Path(args.manifest), pathlib.Path(args.lab)
    m = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    lab = json.loads(lab_path.read_text(encoding="utf-8")) if lab_path.exists() else {"samples": []}

    # Gate states from the benchmark.
    fit = m.get("fit", {})
    phrases = m.get("phrases", {})
    hit = lambda name: (phrases.get(name) or {}).get("hit_rate", {})
    gates = []
    if fit:
        ok = fit["calibration_heard_mae_median"] <= 10 and fit["calibration_steps"] <= 1000
        gates.append(("G1 フィッティング", "約10秒の較正で、較正の記録の聴こえを10 cent以内で再現する",
                      verdict("pass" if ok else "fail", "達成" if ok else "未達"),
                      f"較正 {fit['calibration_steps'] / 100:.1f} 秒、再現誤差 中央値 {fit['calibration_heard_mae_median']:.1f} cent"
                      f"（参考：保留した曲を指令だけで丸ごと予測 {fit['held_out_heard_mae_median']:.1f} cent）"))
    else:
        gates.append(("G1 フィッティング", "約10秒の較正で、較正の記録の聴こえを10 cent以内で再現する", verdict("pending", "未評価"), ""))
    if hit("det_twin") and hit("det_nominal") and hit("det_true"):
        better = hit("det_twin")["low"] > hit("det_nominal")["high"]
        close = hit("det_true")["mean"] - hit("det_twin")["mean"] <= .05
        state = "pass" if better and close else ("partial" if better or close else "fail")
        gates.append(("G2 決定論＋ツイン", "フィッティングなしより有意に高く、理想の当てはめから5ポイント以内",
                      verdict(state, {"pass": "達成", "partial": "一部達成", "fail": "未達"}[state]),
                      f"ツイン {hit('det_twin')['mean'] * 100:.0f} %／なし {hit('det_nominal')['mean'] * 100:.0f} %／理想 {hit('det_true')['mean'] * 100:.0f} %"))
    else:
        gates.append(("G2 決定論＋ツイン", "フィッティングなしより有意に高く、理想の当てはめから5ポイント以内", verdict("pending", "未評価"), ""))
    if hit("nn_twin_play1"):
        gates.append(("G3 NN＋ツイン", "", verdict("pending", "評価中"), ""))
    else:
        gates.append(("G3 NN＋ツイン", "ツインで仕上げたNNが決定論＋ツインから3ポイント以内（1回目）、3回目で決定論以上",
                      verdict("pending", "未着手"),
                      (f"参考：ツインなしのNN 1回目 {hit('nn_play1')['mean'] * 100:.0f} % → 3回目 {hit('nn_play3')['mean'] * 100:.0f} %"
                       if hit("nn_play1") else "")))
    best = max((hit(n)["mean"] for n in ("det_twin", "nn_play3") if hit(n)), default=None)
    gates.append(("G4 デモ", "参照10曲で、3回以内の繰り返しで命中率85 %以上",
                  verdict("pass" if best is not None and best >= .85 else ("fail" if best is not None else "pending"),
                          "達成" if best is not None and best >= .85 else ("未達" if best is not None else "未評価")),
                  f"現在の最良 {best * 100:.0f} %" if best is not None else ""))
    gate_rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in gates)

    def comparison(section):
        data = m.get(section, {})
        return "".join(f"<tr><td><i class='dot' style='background:{color}'></i>{label}</td><td>{pct(data.get(name))}</td>"
                       f"<td>{pct(data.get(name), 'in_tune')}</td>"
                       f"<td>{(data.get(name) or {}).get('reach_ms', {}).get('mean') or 0:.0f} ms</td></tr>"
                       for name, (label, color) in CONTROLLERS.items() if data.get(name))

    per_phrase_rows = "".join(
        f"<tr><td>{html.escape(p['title'])}</td>" + "".join(f"<td>{pct(p['scores'].get(name))}</td>" for name in CONTROLLERS)
        + "</tr>" for p in m.get("per_phrase", []))
    per_phrase_head = "".join(f"<th>{label}</th>" for label, _ in CONTROLLERS.values())

    recovery = fit.get("recovery", {})
    fit_html = (f"""<dl class="facts">
      <div><dt>較正動作の長さ</dt><dd>{fit['calibration_steps'] / 100:.1f} 秒</dd></div>
      <div><dt>較正記録への当てはまり（中央値）</dt><dd>{fit['calibration_heard_mae_median']:.1f} cent</dd></div>
      <div><dt>保留した曲の聴こえ予測（中央値）</dt><dd>{fit['held_out_heard_mae_median']:.1f} cent</dd></div>
      <div><dt>最高速度の推定誤差（中央値）</dt><dd>{recovery['speed_ratio_error'] * 100:.0f} %</dd></div>
      <div><dt>トルクの推定誤差（中央値）</dt><dd>{recovery['torque_ratio_error'] * 100:.0f} %</dd></div>
      <div><dt>不感帯の推定誤差（中央値）</dt><dd>{recovery['deadband_error']:.3f}</dd></div>
      <div><dt>管長ずれの推定誤差（中央値）</dt><dd>{recovery['tube_offset_error_mm']:.1f} mm</dd></div>
      <div><dt>聴こえの遅れの正解率</dt><dd>{recovery['delay_correct'] * 100:.0f} %</dd></div>
    </dl><p class="note">聴こえの誤差は実機でも測れる。「保留した曲の予測」は、音を聴かずに指令だけで1曲（約5秒）を予測したもので、小さなずれが積み重なるため大きく外れる機体がある。実際の制御は聴いた音程で位置を直しながら動くので、この値よりツインを使った制御の成績（G2）を重視する。推定誤差（真の値との比較）はシミュレーターでだけ分かる参考値。</p>"""
               if fit else "<p class='note'>未評価。</p>")

    options = "".join(f'<option value="{html.escape(s["id"])}">{html.escape(s["title"])}</option>' for s in lab["samples"])
    rig_options = "".join(f'<option value="{k}">{"標準的な機体" if k == "typical" else "遅い機体"}（モーター {v["motor_factor"]:.2f} 倍）</option>'
                          for k, v in lab.get("rigs", {}).items())
    checks = "".join(f'<label><input type="checkbox" data-series="{name}" {"checked" if name in ("det_twin", "nn_play3") else ""}>'
                     f'<i class="dot" style="background:{color}"></i>{label}</label>' for name, (label, color) in CONTROLLERS.items())
    audio_options = "".join(f'<option value="{name}">{label}</option>' for name, (label, _) in CONTROLLERS.items())
    scores_json = json.dumps({p["id"]: {n: (s["hit_rate"]["mean"] if s else None) for n, s in p["scores"].items()}
                              for p in m.get("per_phrase", [])})

    doc = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>やまびこ1号 開発計画と結果</title>
<style>
:root{{--bg:#081018;--panel:#0f1d29;--ink:#edf6fb;--muted:#a9bdc9;--line:#294657;--cyan:#4ee1d1;--red:#ff8c78}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 system-ui,"Hiragino Sans","Yu Gothic",sans-serif}}
main{{max-width:1100px;margin:auto;padding:36px 16px 80px}}h1{{font-size:clamp(1.8rem,5vw,3rem);line-height:1.15;margin:.2em 0}}
h2{{margin-top:2.6rem;color:var(--cyan);border-bottom:1px solid var(--line);padding-bottom:.3rem}}h3{{margin-top:1.6rem}}
.lead{{color:var(--muted);font-size:1.05rem}}.note{{color:var(--muted);font-size:.9rem}}small{{color:var(--muted)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:14px 0}}
.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}}
th{{color:var(--muted);font-weight:600}}.state{{padding:2px 8px;border-radius:8px;font-size:.8rem;white-space:nowrap}}
.state.pass{{color:#70f0ac;border:1px solid #46c987}}.state.fail{{color:var(--red);border:1px solid var(--red)}}
.state.partial{{color:#ffc96b;border:1px solid #d69c3b}}.state.pending{{color:#9eb2c0;border:1px solid #526675}}
.flow{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}}.flow div{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}}
.flow b{{color:var(--cyan)}}.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}}.facts div{{background:#0a141d;border-radius:8px;padding:10px}}
dt{{color:var(--muted);font-size:.78rem}}dd{{margin:0;font-size:1.1rem}}.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
.controls{{display:flex;flex-wrap:wrap;gap:12px;align-items:center}}select{{background:#0a141d;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px}}
.series{{display:flex;flex-wrap:wrap;gap:10px 16px;margin:10px 0}}canvas{{width:100%;height:340px;background:#0a141d;border-radius:10px;border:1px solid var(--line)}}
audio{{width:100%;max-width:420px}}a{{color:var(--cyan)}}
</style></head><body><main>
<p class="note">PHYSICAL AI / やまびこ1号 — 最新の計画と結果（この計画に関係する内容だけ）</p>
<h1>未知の機体に着いたら、まず機体を知り、<br>同じ曲を繰り返して上手くなる。</h1>
<p class="lead">実機のモーターと笛は、最初にフィッティングした後はほぼ変わらない。ただし最初の1台がどんなものかは分からない。そこで「短い較正動作で機体を知る → その機体のデジタルツインを作る → 機体に合わせた制御で演奏する → 同じ曲を繰り返して上手くなる」を開発の筋にする。どの段階もエンコーダなしの決定論的モジュールと同じ条件で比べる。</p>

<h2>1. 計画</h2>
<div class="flow">
  <div><b>F1 較正動作</b><br>原点合わせの後、バルブを開けて決まったPWM列を約8秒流し、聴こえた音程を記録する（実機で実行できる）。</div>
  <div><b>F2 デジタルツイン</b><br>記録にシミュレーターの機体の値を当てはめ、その機体専用のシミュレーターを作る（PC）。</div>
  <div><b>F3 機体に合わせた制御</b><br>決定論はツインの値を使う。NNはツインの周りだけで仕上げる。</div>
  <div><b>F4 繰り返しで上達</b><br>曲の記憶で、同じ曲を繰り返すほど上手くなる（デモの見せ場）。</div>
  <div><b>F5 評価</b><br>固定ベンチマークで命中率と95 %区間を比べる。</div>
</div>
<div class="card"><b>デジタルツイン（ツイン）とは</b>：実際の1台の機体をまねた、その機体専用のシミュレーター。実機に約7秒の決まった動き（小さく動かす、全力で往復する、端から端までゆっくり動かす）をさせ、送った指令と聴こえた音程だけを記録し、シミュレーターのモーターの速さ・力・摩擦・不感帯、笛の管の長さ、音が聴こえるまでの遅れを「同じ指令なら同じ音程が聴こえる」ように合わせて作る。決定論の制御は標準値の代わりにツインの値を使い、NNは実機の代わりにツインの中で練習してその機体向けに仕上げる。</div>
<h3>合格基準と現状</h3>
<div class="table"><table><thead><tr><th>段階</th><th>基準（暫定・実機計測後に見直す）</th><th>判定</th><th>現状</th></tr></thead><tbody>{gate_rows}</tbody></table></div>
<p class="note">命中率＝各音が鳴り始めから0.3秒以内に±25 centへ入った割合。平均誤差ではなくこれで判断し、95 %区間が重ならない差だけを改善と呼ぶ。</p>

<h2>2. シミュレーター</h2>
<div class="card"><div class="table"><table><thead><tr><th>要素</th><th>模擬しているもの</th></tr></thead><tbody>
<tr><td>笛</td><td>閉管 f = c/4(L−x)。変位に線形なのは周期で、音程は対数。原点は最低音より200 cent低く、どの機体でも全音域を出せる。管長±10 mm、温度±8 ℃、吹圧±10 cent。</td></tr>
<tr><td>モーター</td><td>トルクの立ち上がり、慣性、クーロン・粘性摩擦、最高速度、両端のストッパー、PWMの不感帯（0.20前後）。速度とトルクは実機未計測のため0.4〜2.5倍の幅で乱数化。</td></tr>
<tr><td>聴こえ</td><td>1±1ステップ（10 ms単位）の遅れ、3 centの音程ノイズ、2 %の欠落。</td></tr>
<tr><td>ブラックボックスの決まり</td><td>実機と同じく、送れるのはPWMとバルブ、受け取れるのは聴こえた音程だけ。機体の真の値や勾配は学習にも制御にも使わず、採点と原因分析にだけ使う。</td></tr>
</tbody></table></div>
<p class="note">ベンチマーク：未知の機体 {m.get('rigs', '—')} 台（乱数固定）、参照10曲（0.5倍速、正解の楽譜）と乱数曲。実機では未検証。</p></div>

<h2>3. 操作できるシミュレーション</h2>
<div class="card">
  <div class="controls"><label>曲 <select id="song">{options}</select></label><label>機体 <select id="rig">{rig_options}</select></label>
  <label>音を聴く <select id="listen">{audio_options}</select></label></div>
  <div class="series">{checks}</div>
  <canvas id="chart"></canvas>
  <p><audio id="target" controls preload="none"></audio> 目標 &nbsp; <audio id="played" controls preload="none"></audio> 選んだ方式</p>
  <p id="score" class="note"></p>
  <p class="note">選んだ機体での演奏をブラックボックスのシミュレーターで記録したもの。音はブラウザで合成した診断音。命中率はこの曲を{m.get('rigs', '—')}台で演奏した平均。</p>
</div>

<h2>4. 結果</h2>
<h3>F1〜F2 フィッティングの精度</h3>
<div class="card">{fit_html}</div>
<h3>制御方式の比較：参照10曲</h3>
<div class="table"><table><thead><tr><th>方式</th><th>命中率</th><th>合っている割合</th><th>届くまで（中央値）</th></tr></thead><tbody>{comparison('phrases')}</tbody></table></div>
<h3>制御方式の比較：乱数曲</h3>
<div class="table"><table><thead><tr><th>方式</th><th>命中率</th><th>合っている割合</th><th>届くまで（中央値）</th></tr></thead><tbody>{comparison('random')}</tbody></table></div>
<h3>曲ごとの命中率</h3>
<div class="table"><table><thead><tr><th>曲</th>{per_phrase_head}</tr></thead><tbody>{per_phrase_rows}</tbody></table></div>

<h2>5. 関連</h2>
<p><a href="fitting-twin-plan.md">計画書（Markdown）</a> ／ <a href="e2e-training-report.html">これまでの試行の記録（経緯を含む詳細レポート）</a></p>
</main>
<script>
const lab = {json.dumps(lab, ensure_ascii=False, separators=(",", ":"))};
const scores = {scores_json};
const names = {json.dumps({k: v[0] for k, v in CONTROLLERS.items()}, ensure_ascii=False)};
const colors = {json.dumps({k: v[1] for k, v in CONTROLLERS.items()})};
const $ = id => document.getElementById(id);
function wav(track, voice) {{
  const sr = 16000, per = 160, n = track.length * per, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const s = (o, t) => [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  s(0,'RIFF'); v.setUint32(4,36+n*2,true); s(8,'WAVE'); s(12,'fmt '); v.setUint32(16,16,true); v.setUint16(20,1,true);
  v.setUint16(22,1,true); v.setUint32(24,sr,true); v.setUint32(28,sr*2,true); v.setUint16(32,2,true); v.setUint16(34,16,true);
  s(36,'data'); v.setUint32(40,n*2,true); let ph = 0, c = 44;
  for (let f = 0; f < track.length; f++) {{ const step = 2*Math.PI*440*Math.pow(2, track[f]/1200)/sr, on = voice[f];
    for (let i = 0; i < per; i++) {{ const x = on ? .22*Math.sin(ph) + .045*Math.sin(2*ph) : 0;
      v.setInt16(c, Math.round(x*32767), true); c += 2; ph = (ph + step) % (2*Math.PI); }} }}
  return URL.createObjectURL(new Blob([buf], {{type:'audio/wav'}}));
}}
function draw() {{
  const sample = lab.samples.find(x => x.id === $('song').value) || lab.samples[0]; if (!sample) return;
  const tracks = sample.rigs[$('rig').value] || {{}};
  const cv = $('chart'), r = cv.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  cv.width = r.width*dpr; cv.height = r.height*dpr; const g = cv.getContext('2d'); g.scale(dpr, dpr);
  const W = r.width, H = r.height, m = {{l:52,r:12,t:12,b:28}}, lo = 600, hi = 2000, n = sample.target.length;
  const x = i => m.l + i*(W-m.l-m.r)/(n-1), y = c => m.t + (hi - Math.max(lo, Math.min(hi, c)))*(H-m.t-m.b)/(hi-lo);
  g.clearRect(0,0,W,H); g.font = '11px system-ui'; g.fillStyle = '#a9bdc9'; g.strokeStyle = '#294657';
  [700,1000,1300,1600,1900].forEach(c => {{ g.beginPath(); g.moveTo(m.l,y(c)); g.lineTo(W-m.r,y(c)); g.stroke(); g.fillText(c, 8, y(c)+4); }});
  g.fillText('時間 ' + (n/100).toFixed(1) + ' s', m.l, H-8);
  const line = (t, color, width) => {{ g.strokeStyle = color; g.lineWidth = width; g.beginPath(); let open = false;
    t.forEach((c, i) => {{ if (!sample.voice[i]) {{ open = false; return; }} if (!open) {{ g.moveTo(x(i), y(c)); open = true; }} else g.lineTo(x(i), y(c)); }}); g.stroke(); }};
  line(sample.target, '#f2f5f7', 3);
  document.querySelectorAll('[data-series]').forEach(cb => {{ if (cb.checked && tracks[cb.dataset.series]) line(tracks[cb.dataset.series], colors[cb.dataset.series], 1.6); }});
  const chosen = $('listen').value; $('target').src = wav(sample.target, sample.voice);
  if (tracks[chosen]) $('played').src = wav(tracks[chosen], sample.voice);
  const sc = scores[sample.id] || {{}};
  $('score').textContent = Object.keys(names).filter(k => sc[k] != null).map(k => names[k] + ' ' + Math.round(sc[k]*100) + '%').join(' ／ ');
}}
['song','rig','listen'].forEach(id => $(id).addEventListener('change', draw));
document.querySelectorAll('[data-series]').forEach(cb => cb.addEventListener('change', draw));
addEventListener('resize', draw); draw();
</script></body></html>"""
    pathlib.Path(args.out).write_text(doc, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
