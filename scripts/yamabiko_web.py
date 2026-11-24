"""UNO Q: a web page to try the servo, the actuator and the fan by hand.

    python3 scripts/yamabiko_web.py [--port 8080]      # then open http://<UNO Q address>:8080/

Talks to the command server (scripts/yamabiko_server.py) on the same board, as one connection: while the
page drives the outputs, other clients (yamabiko_remote.py, the button jog) are refused until it lets go.

Safety:
- the actuator runs only while a button on the page is held: the page repeats the command every 100 ms and
  the actuator stops when none came for 0.4 s (a dropped phone, a closed tab)
- when no page has asked for the status for 3 s, everything goes back to safe (actuator 0, valve shut,
  fan off) and the connection to the command server is closed
Only the Python standard library and the repository's remote.py are used.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.profiles import latest, load_profiles  # noqa: E402
from flute_rl.yamabiko.remote import Remote, RemoteError  # noqa: E402

ACT_HOLD_S = .4
IDLE_S = 3.0


class Bench:
    """The one connection to the command server, plus the dead-man timers."""

    def __init__(self, port):
        self.port = port
        self.rig: Remote | None = None
        self.lock = threading.RLock()
        self.act_until = 0.0
        self.act_pwm = 0.0
        self.last_seen = 0.0
        self.servo = {"open": 1800, "closed": 1200, "move_ms": 150}
        threading.Thread(target=self._watch, daemon=True).start()

    def _connect(self) -> Remote:
        if self.rig is None:
            self.rig = Remote(port=self.port, forward=False)
            self.rig.limit(1.0)
        return self.rig

    def call(self, fn):
        with self.lock:
            self.last_seen = time.monotonic()
            try:
                return fn(self._connect())
            except (OSError, RemoteError) as exc:
                if isinstance(exc, OSError) or "closed" in str(exc):
                    self._drop()
                raise

    def _drop(self):
        if self.rig is not None:
            try:
                self.rig.close()
            except Exception:
                pass
            self.rig = None
        self.act_pwm, self.act_until = 0.0, 0.0

    def _watch(self):
        while True:
            time.sleep(.05)
            with self.lock:
                now = time.monotonic()
                if self.rig is None:
                    continue
                if self.act_pwm and now > self.act_until:
                    self.act_pwm = 0.0
                    try:
                        self.rig.actuator(0.0)
                    except (OSError, RemoteError):
                        self._drop()
                if now - self.last_seen > IDLE_S:
                    self._drop()              # close() sends stop: actuator 0, valve shut, fan off

    # ----- page commands --------------------------------------------------------------------------
    def actuator(self, pwm):
        pwm = max(-1.0, min(1.0, float(pwm)))

        def go(rig):
            if pwm != self.act_pwm:
                rig.actuator(pwm)
            self.act_pwm = pwm
            self.act_until = time.monotonic() + ACT_HOLD_S
            return {"pwm": pwm}
        return self.call(go)

    def fan(self, on, duty):
        return self.call(lambda rig: rig.fan(bool(on), max(0.0, min(1.0, float(duty)))))

    def valve(self, open_):
        return self.call(lambda rig: rig.valve(bool(open_)))

    def servo_positions(self, open_us, closed_us, move_ms):
        self.servo = {"open": int(open_us), "closed": int(closed_us), "move_ms": int(move_ms)}
        return self.call(lambda rig: rig.valve_positions(self.servo["open"], self.servo["closed"], self.servo["move_ms"]))

    def stop(self):
        def go(rig):
            self.act_pwm = 0.0
            rig.actuator(0.0); rig.fan(False, 0.0); rig.valve(False)
            return {}
        return self.call(go)

    def status(self):
        def go(rig):
            st = rig.status()
            st["servo_cfg"] = self.servo
            return st
        return self.call(go)


PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>やまびこ1号 テスト</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mute:#6b7280;--line:#e3e6eb;--acc:#2563eb;--warn:#dc2626;--ok:#16a34a}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--card:#1b2027;--fg:#e8ebf0;--mute:#9aa3af;--line:#2b323c;--acc:#60a5fa;--warn:#f87171;--ok:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:16px}h1{font-size:20px;margin:4px 0 12px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:14px}
h2{font-size:16px;margin:0 0 10px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}
label{min-width:7.5em;color:var(--mute)}input[type=range]{flex:1;min-width:160px}
input[type=number]{width:6em;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
button{font:inherit;padding:12px 16px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer;touch-action:none;user-select:none;-webkit-user-select:none}
button.big{flex:1;min-width:130px;font-size:18px;padding:22px 10px}button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
button.stop{background:var(--warn);color:#fff;border-color:var(--warn);width:100%;font-size:18px}
.val{font-variant-numeric:tabular-nums;min-width:4.5em;text-align:right}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td{padding:4px 0;border-bottom:1px solid var(--line)}td:last-child{text-align:right}
#err{color:var(--warn);min-height:1.5em}.mute{color:var(--mute);font-size:14px}
</style></head><body><main>
<h1>やまびこ1号 テスト</h1>
<p><a href="/whistle">口笛コピー →</a></p>
<section><button class="stop" id="stop">すべて止める</button><div id="err"></div></section>

<section><h2>リニアアクチュエーター</h2>
<div class="row"><label>duty</label><input type="range" id="aduty" min="0" max="100" value="100"><span class="val" id="adutyv"></span></div>
<div class="row"><button class="big" id="ext">伸ばす（押している間）</button><button class="big" id="ret">縮める（押している間）</button></div>
<div class="mute">ボタンを離すか、0.4 秒通信が途切れると止まります。</div></section>

<section><h2>ファン</h2>
<div class="row"><label>duty</label><input type="range" id="fduty" min="0" max="100" value="50"><span class="val" id="fdutyv"></span></div>
<div class="row"><button class="big" id="fan">ファン OFF</button></div>
<div class="mute">ページを閉じて 3 秒たつと止まります。</div></section>

<section><h2>サーボ（SG90・弁）</h2>
<div class="row"><button class="big" id="vopen">開</button><button class="big" id="vclose">閉</button></div>
<div class="row"><label>開のパルス</label><input type="range" id="sopen" min="500" max="2500" step="10"><span class="val" id="sopenv"></span></div>
<div class="row"><label>閉のパルス</label><input type="range" id="sclose" min="500" max="2500" step="10"><span class="val" id="sclosev"></span></div>
<div class="row"><label>動く時間</label><input type="range" id="smove" min="0" max="1000" step="10"><span class="val" id="smovev"></span></div>
<div class="mute">パルスを変えると、今の位置（開か閉）にすぐ反映されます。電磁弁の出力（J5）も開閉に連動します。</div></section>

<section><h2>状態</h2><table id="st"></table></section>
</main>
<script>
const $=id=>document.getElementById(id);
const err=m=>{$('err').textContent=m||''};
async function api(path,body){
  try{const r=await fetch('/api/'+path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
    const j=await r.json();if(!r.ok)throw new Error(j.error||r.status);err('');return j}
  catch(e){err('エラー: '+e.message);return null}}
const pct=v=>v+' %';
function bindRange(id,fmt){const el=$(id),out=$(id+'v');const f=()=>out.textContent=fmt(el.value);el.addEventListener('input',f);f();return el}
const aduty=bindRange('aduty',pct), fduty=bindRange('fduty',pct);
const sopen=bindRange('sopen',v=>v+' µs'), sclose=bindRange('sclose',v=>v+' µs'), smove=bindRange('smove',v=>v+' ms');

// actuator: repeat while held
let holdTimer=null;
function hold(btn,sign){
  const send=()=>api('actuator',{pwm:sign*aduty.value/100});
  const start=e=>{e.preventDefault();if(holdTimer)return;btn.classList.add('on');try{btn.setPointerCapture(e.pointerId)}catch(_){};send();holdTimer=setInterval(send,100)};
  const end=()=>{if(!holdTimer)return;clearInterval(holdTimer);holdTimer=null;btn.classList.remove('on');api('actuator',{pwm:0})};
  btn.addEventListener('pointerdown',start);['pointerup','pointercancel','lostpointercapture'].forEach(t=>btn.addEventListener(t,end));
  btn.addEventListener('contextmenu',e=>e.preventDefault());}
hold($('ext'),1);hold($('ret'),-1);

// fan
let fanOn=false;
function fanSend(){api('fan',{on:fanOn,duty:fduty.value/100})}
$('fan').onclick=()=>{fanOn=!fanOn;fanShow();fanSend()};
function fanShow(){$('fan').textContent=fanOn?'ファン ON':'ファン OFF';$('fan').classList.toggle('on',fanOn)}
fduty.addEventListener('change',()=>{if(fanOn)fanSend()});

// servo
$('vopen').onclick=()=>api('valve',{open:true});$('vclose').onclick=()=>api('valve',{open:false});
const servoSend=()=>api('servo',{open:+sopen.value,closed:+sclose.value,move_ms:+smove.value});
[sopen,sclose,smove].forEach(el=>el.addEventListener('change',servoSend));

$('stop').onclick=async()=>{if(holdTimer){clearInterval(holdTimer);holdTimer=null}fanOn=false;fanShow();await api('stop',{})};

// status
let first=true;
async function poll(){
  const s=await api('status');
  if(s){
    if(first&&s.servo_cfg){sopen.value=s.servo_cfg.open;sclose.value=s.servo_cfg.closed;smove.value=s.servo_cfg.move_ms;[sopen,sclose,smove].forEach(el=>el.dispatchEvent(new Event('input')));first=false}
    if(fanOn!==!!s.fan){fanOn=!!s.fan;fanShow()}
    const rows=[['アクチュエーター（MCU が出している duty）',(s.applied_pwm*100).toFixed(0)+' %'],
      ['電源から流れる電流',s.isense_mv!=null?(s.isense_mv*10)+' mA（ピーク '+(s.isense_peak_mv*10)+'）':'—'],
      ['ファン',(s.fan_switch?'ON':'OFF')+' / duty '+Math.round(s.fan_duty*100)+' %'],['ファンの回転数',s.fan_rpm+' rpm'],
      ['弁',s.valve_open?'開':'閉'],['サーボのパルス',s.servo_pos+' µs'],
      ['EXEC ボタン',(s.button_held?'押している':'離している')+'（'+s.button_count+' 回）'],
      ['マイク',s.mic_db+' dBFS'+(s.pitch_hz?' / '+s.pitch_hz+' Hz':'')],
      ['指令の途絶（ファームウェア）',s.fw_timeout?'あり':'なし']];
    $('st').innerHTML=rows.map(r=>'<tr><td>'+r[0]+'</td><td>'+r[1]+'</td></tr>').join('');}
  setTimeout(poll,250)}
poll();
</script></body></html>
"""



# ----- whistle copy: record a whistle, play it back with a performer runtime ---------------------------

WHISTLE_DIR = pathlib.Path(__file__).resolve().parents[1] / "runs" / "real" / "whistle"


def common_ear(profile):
    """Server arguments for the profile's ear when it hears everything (step5: "common_ear": true), else {}."""
    if profile.get("common_ear") and profile.get("ear") == "nn":
        return {"ear": "nn", "ear_runtime": profile["ear_runtime_path"]}
    return {}
def wav_bytes(pcm, sr):
    import io
    import wave
    import numpy as np
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr))
        w.writeframes(np.asarray(pcm, "<i2").tobytes())
    return buf.getvalue()


VALVE_US = (1560, 1820)       # the SG90 pulses set on the bench page for the current rig (open, closed)


class WhistleJob:
    """One whistle copy at a time, in a thread: count-in, record, analyse (YIN), play, report."""

    def __init__(self, bench, server_port, profiles):
        self.bench, self.server_port = bench, server_port
        self.profiles = profiles                  # models/<step>/profile.json, any of them runnable
        self.step = latest(profiles)
        self.lock = threading.Lock()
        self.state = {"phase": "idle", "message": "スタートを押してください"}
        self.thread = None
        self.tunings = {}         # per step, the last calibration: shift_cents, home / top pitch, the sweep
        self.wavs = {}            # "whistle" / "flute": WAV bytes of the last copy

    def _launch(self, target, *args, first="準備中"):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError("まだ前の作業が動いています")
            self.state = {"phase": "count", "message": first}
            self.thread = threading.Thread(target=target, args=args, daemon=True)
            self.thread.start()
        return {"started": True}

    def _pick(self, step):
        if step:
            if step not in self.profiles:
                raise ValueError(f"{step} がありません")
            self.step = step
        return self.profiles[self.step]

    def start(self, seconds, quantize, step=None):
        profile = self._pick(step)
        return self._launch(self._run, float(seconds), bool(quantize), profile)

    def calibrate(self, step=None):
        profile = self._pick(step)
        return self._launch(self._calibrate, profile, first="キャリブレーション中")

    def calibrate_at_start(self, wait_s=180.0):
        """Once after the page starts (the board just booted): wait for the command server, then calibrate."""
        def go():
            until = time.time() + wait_s
            while time.time() < until:
                try:
                    Remote(port=self.server_port, forward=False).close()
                    break
                except OSError:
                    time.sleep(5)
            time.sleep(5)
            try:
                self.calibrate(self.step)
            except ValueError:
                pass
        threading.Thread(target=go, daemon=True).start()

    def status(self):
        with self.lock:
            return dict(self.state, tuning=self.tunings.get(self.step), has_audio=sorted(self.wavs), step=self.step,
                        steps=[{"name": n, "title": p.get("title", n)} for n, p in self.profiles.items()])

    def _tune(self, rig, profile):
        self._set(phase="tune", message="キャリブレーション中（縮み切りの音と、上下に 1 往復）")
        result = rig.tune(runtime=profile["runtime_path"], fan=profile["fan"], **common_ear(profile))
        tuning = {k: result[k] for k in ("shift_cents", "home_cents", "top_cents", "sweep_max_cents", "heard", "phase")}
        tuning["time"] = time.strftime("%H:%M")
        self.tunings[profile["name"]] = tuning

    def _calibrate(self, profile):
        rig = None
        try:
            with self.bench.lock:
                self.bench._drop()
            rig = Remote(port=self.server_port, forward=False)
            rig.valve_positions(*profile["valve_us"], move_ms=20)
            self._tune(rig, profile)
            self._set(phase="idle", message="キャリブレーションできました。スタートを押してください")
        except Exception as exc:                       # noqa: BLE001
            self._set(phase="error", message=f"キャリブレーションできませんでした: {exc}")
        finally:
            if rig is not None:
                try:
                    rig.stop(force=True)
                finally:
                    rig.close()

    def _set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def _run(self, seconds, quantize, profile):
        import numpy as np
        from flute_rl.yamabiko.whistle import clean_track, to_target, whistle_track
        rig = None
        try:
            with self.bench.lock:                      # the page's own connection lets go: one driver at a time
                self.bench._drop()
            rig = Remote(port=self.server_port, forward=False)
            rig.valve_positions(*profile["valve_us"], move_ms=20)
            if profile["name"] not in self.tunings:   # not calibrated since the page started
                self._tune(rig, profile)
            rig.stop(force=True)                       # fan off, valve shut: only the whistle in the mic
            for k in (3, 2, 1):
                self._set(phase="count", message=f"{k} …")
                time.sleep(1)
            self._set(phase="record", message=f"口笛をどうぞ（{seconds:.0f} 秒）")
            pcm, sr = rig.record(seconds)
            self.wavs = {"whistle": wav_bytes(pcm, sr)}
            stamp = time.strftime("%Y%m%d_%H%M%S")
            WHISTLE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(WHISTLE_DIR / f"{stamp}_whistle.npz", pcm=pcm, sr=sr)   # data for the ear (step5)
            self._set(phase="analyse", message="聞き取り中")
            try:
                if profile.get("memory_runtime_path"):      # step6: written into the memory, read back from it
                    from flute_rl.yamabiko.ear_nn import Ear
                    from flute_rl.yamabiko.score_memory import score_from_recording
                    target, voice, raw, report = score_from_recording(
                        pcm, Ear(dict(np.load(profile["ear_runtime_path"]))), dict(np.load(profile["memory_runtime_path"])))
                    cents = raw
                else:
                    if common_ear(profile):                # step5: the same ear network hears the whistle
                        from flute_rl.yamabiko.ear_nn import Ear
                        raw = Ear(dict(np.load(profile["ear_runtime_path"]))).track(pcm, sr)
                    else:
                        raw = whistle_track(pcm, sr)
                    cents = clean_track(raw, **profile.get("whistle", {}))
                    target, voice, report = to_target(cents, profile["low_cents"], profile["high_cents"],
                                                      quantize=quantize)
            except ValueError:
                level = 20 * np.log10(max(float(np.sqrt(np.mean((pcm / 32768.0) ** 2))), 1e-9))
                raise ValueError(f"口笛が聞こえませんでした（録音の大きさ {level:.0f} dB）。マイクの近くで吹いてください")
            path = WHISTLE_DIR / f"{stamp}_target.npz"
            np.savez(path, cents=target, voice=voice, whistle_cents=raw)

            def listed(a):
                return [None if not np.isfinite(v) else round(float(v), 1) for v in a]
            self._set(phase="play", message="演奏中", whistle=listed(raw), cleaned=listed(cents),
                      target=[round(float(v), 1) for v in target], voice=[bool(v) for v in voice], report=report)
            result = rig.play(profile["runtime_path"], str(path), out=str(WHISTLE_DIR / f"{stamp}_play.npz"),
                              repeats=1, fan=profile["fan"], shift_cents=self.tunings[profile["name"]]["shift_cents"],
                              warmup_s=2.0, ear=profile.get("ear", "yin"),
                              valve_runtime=profile.get("valve_runtime_path"),
                              ear_runtime=profile.get("ear_runtime_path"))
            played = np.load(WHISTLE_DIR / f"{stamp}_play.npz")
            if "0/pcm" in played:
                self.wavs["flute"] = wav_bytes(played["0/pcm"], int(played["sr"]))
            h, v, tc = played["0/heard"][0], played["0/valid"][0], played["0/target_cents"][0]
            m = v & voice
            key = float(np.median((h - tc)[m])) if m.any() else 0.0
            rel = float(np.abs(h - tc - key)[m].mean()) if m.any() else None
            shift = float(result.get("shift_cents", 0.0))
            self._set(phase="done", message="できました",
                      heard=[round(float(x - shift), 1) if ok else None for x, ok in zip(h, v)],
                      key=key, relative_mae=rel, shift=shift, stamp=stamp)
        except Exception as exc:                       # noqa: BLE001  (shown on the page)
            self._set(phase="error", message=str(exc))
        finally:
            if rig is not None:
                try:
                    rig.stop(force=True)
                finally:
                    rig.close()


WHISTLE_PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>口笛コピー</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mute:#6b7280;--line:#e3e6eb;--acc:#2563eb;--warn:#dc2626;--ok:#16a34a;--c1:#9ca3af;--c2:#2563eb;--c3:#111827;--c4:#ea580c}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--card:#1b2027;--fg:#e8ebf0;--mute:#9aa3af;--line:#2b323c;--acc:#60a5fa;--warn:#f87171;--ok:#4ade80;--c1:#6b7280;--c2:#60a5fa;--c3:#e5e7eb;--c4:#fb923c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:900px;margin:0 auto;padding:16px}h1{font-size:20px;margin:4px 0 12px}a{color:var(--acc)}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:14px}
h2{font-size:16px;margin:0 0 8px}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{font:inherit;font-size:20px;padding:18px 28px;border-radius:12px;border:0;background:var(--acc);color:#fff;cursor:pointer}
button:disabled{opacity:.45;cursor:default}button.sub{background:var(--bg);color:var(--fg);border:1px solid var(--line);font-size:16px;padding:14px 18px}
button.play{font-size:15px;padding:10px 14px;background:var(--bg);color:var(--fg);border:1px solid var(--line)}label{color:var(--mute)}select{font:inherit;padding:6px;border-radius:6px}
#msg{font-size:22px;font-weight:600;min-height:1.5em}.mute{color:var(--mute);font-size:14px}
svg{width:100%;height:auto;display:block}.legend span{display:inline-block;margin-right:14px;font-size:14px}
.sw{display:inline-block;width:14px;height:4px;vertical-align:middle;margin-right:5px;border-radius:2px}
.num{font-variant-numeric:tabular-nums}
</style></head><body><main>
<h1>口笛コピー <span class="mute">— 吹いたメロディーを、やまびこ1号がまねします</span></h1>
<p class="mute"><a href="/">← テストページへ</a></p>
<section>
<div class="row">
<button id="start">スタート</button>
<button id="cal" class="sub">キャリブレーション</button>
<label>使う版 <select id="step"></select></label>
<label>録音の長さ <select id="sec"><option>3</option><option>4</option><option selected>5</option><option>6</option><option>8</option></select> 秒</label>
<label><input type="checkbox" id="quant"> 半音にそろえる</label>
</div>
<div id="msg" style="margin-top:12px"></div>
<div class="mute">縦軸はセント（100 セント = 半音、0 = ラ 440 Hz）。メロディーは音程の間隔を保って笛の音域に移します。</div>
</section>
<section><h2>聞いた音（口笛）</h2>
<div class="legend"><span><i class="sw" style="background:var(--c1)"></i>口笛の音程</span><span><i class="sw" style="background:var(--c2)"></i>整えた後</span></div>
<svg id="g1" viewBox="0 0 860 260"></svg><div class="mute num" id="info1"></div>
<div class="row" style="margin-top:8px"><button class="play" id="pw" disabled>▶ 録音した口笛を聞く</button></div></section>
<section><h2>再生した音（笛）</h2>
<div class="legend"><span><i class="sw" style="background:var(--c3)"></i>目標（口笛を笛の音域へ）</span><span><i class="sw" style="background:var(--c4)"></i>笛が鳴らした音</span></div>
<svg id="g2" viewBox="0 0 860 260"></svg><div class="mute num" id="info2"></div>
<div class="row" style="margin-top:8px"><button class="play" id="pf" disabled>▶ 笛で再現した音を聞く</button></div></section>
<section><h2>キャリブレーション（その日の笛）</h2>
<div class="legend"><span><i class="sw" style="background:var(--c2)"></i>縮み切り → 伸び切り → 縮み切り（PWM 0.6）</span></div>
<svg id="g3" viewBox="0 0 860 200"></svg><div class="mute num" id="info3">まだ測っていません</div></section>
<audio id="au"></audio>
</main><script>
const $=id=>document.getElementById(id);
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function plot(svg,series){
  const W=860,H=260,L=56,R=10,T=10,B=30;let lo=Infinity,hi=-Infinity,n=1;
  for(const s of series){n=Math.max(n,s.y.length);for(const v of s.y){if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v)}}}
  if(!isFinite(lo)){svg.innerHTML='';return}
  const pad=Math.max(50,(hi-lo)*.08);lo-=pad;hi+=pad;
  const x=i=>L+(W-L-R)*i/Math.max(1,n-1),y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  const step=(hi-lo)>1600?400:(hi-lo)>700?200:100;let g="";
  for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step){g+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="${css('--line')}"/><text x="${L-6}" y="${y(v)+4}" font-size="12" text-anchor="end" fill="${css('--mute')}">${v}</text>`}
  for(let s=0;s<=n/100;s++){g+=`<text x="${x(s*100)}" y="${H-10}" font-size="12" text-anchor="middle" fill="${css('--mute')}">${s}s</text>`}
  for(const s of series){
    if(s.dots){for(let i=0;i<s.y.length;i++){if(s.y[i]!==null)g+=`<circle cx="${x(i)}" cy="${y(s.y[i])}" r="1.6" fill="${s.c}"/>`}continue}
    let d="",up=false;for(let i=0;i<s.y.length;i++){const v=s.y[i];if(v===null){up=false;continue}d+=(up?"L":"M")+x(i).toFixed(1)+" "+y(v).toFixed(1);up=true}
    g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="${s.w||2}" stroke-linejoin="round"/>`}
  svg.innerHTML=g}
async function poll(){
  let s;try{s=await (await fetch("/api/whistle/status")).json()}catch(e){setTimeout(poll,1500);return}
  $("msg").textContent=s.message||"";$("msg").style.color=s.phase==="error"?css("--warn"):s.phase==="done"?css("--ok"):"";
  if(s.whistle){
    plot($("g1"),[{y:s.whistle,c:css("--c1"),dots:true},{y:s.cleaned,c:css("--c2")}]);
    const r=s.report;$("info1").textContent=`口笛の幅 ${r.whistle_span_cents.toFixed(0)} セント／笛の音域 ${r.range_cents.toFixed(0)} セント`+(r.clipped_fraction>0?`（はみ出し ${(r.clipped_fraction*100).toFixed(0)} %）`:"")}
  if(s.target){const tgt=s.target.map((v,i)=>s.voice[i]?v:null);
    plot($("g2"),[{y:tgt,c:css("--c3"),w:3},...(s.heard?[{y:s.heard.map((v,i)=>s.voice[i]?v:null),c:css("--c4"),dots:true}]:[])]);
    $("info2").textContent=s.relative_mae!=null?`調のずれ ${s.key>=0?"+":""}${s.key.toFixed(0)} セント／それを除いた平均の誤差 ${s.relative_mae.toFixed(0)} セント`:""}
  if(s.steps&&$("step").options.length!==s.steps.length){$("step").innerHTML=s.steps.map(p=>`<option value="${p.name}">${p.title}</option>`).join("")}
  if(s.step&&document.activeElement!==$("step"))$("step").value=s.step;
  const tu=s.tuning;if(!tu){shownTune=null;$("g3").innerHTML="";$("info3").textContent="この版ではまだ測っていません（スタートで先に測ります）"}
  if(tu&&tu.time!==shownTune){shownTune=tu.time;plot($("g3"),[{y:tu.heard,c:css("--c2"),dots:true}]);
    const hz=c=>(440*Math.pow(2,c/1200)).toFixed(0);
    $("info3").textContent=`${tu.time} に測定：縮み切り ${hz(tu.home_cents)} Hz、いちばん高い音 ${tu.sweep_max_cents!=null?hz(tu.sweep_max_cents):"-"} Hz、ツインとのずれ ${tu.shift_cents>=0?"+":""}${tu.shift_cents.toFixed(0)} セント`}
  const busy=!["idle","done","error"].includes(s.phase);$("start").disabled=busy;$("cal").disabled=busy;
  const ha=s.has_audio||[];$("pw").disabled=busy||!ha.includes("whistle");$("pf").disabled=busy||!ha.includes("flute");
  setTimeout(poll,busy?250:1500)}
$("start").onclick=async()=>{$("start").disabled=true;$("g1").innerHTML="";$("g2").innerHTML="";$("info1").textContent="";$("info2").textContent="";
  const r=await fetch("/api/whistle/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({seconds:+$("sec").value,quantize:$("quant").checked,step:$("step").value})});
  if(!r.ok){$("msg").textContent=(await r.json()).error}};
let shownTune=null;
$("cal").onclick=async()=>{$("cal").disabled=true;const r=await fetch("/api/whistle/calibrate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({step:$("step").value})});
  if(!r.ok){$("msg").textContent=(await r.json()).error}};
function listen(name){const a=$("au");a.src=`/api/whistle/${name}.wav?t=${Date.now()}`;a.play()}
$("pw").onclick=()=>listen("whistle");$("pf").onclick=()=>listen("flute");
poll();
</script></body></html>
"""

def make_handler(bench: Bench, whistle: "WhistleJob"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False))

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._run(bench.status)
            elif self.path == "/whistle":
                self._send(200, WHISTLE_PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/whistle/status":
                self._json(200, whistle.status())
            elif self.path.split("?")[0] in ("/api/whistle/whistle.wav", "/api/whistle/flute.wav"):
                name = self.path.split("?")[0].rsplit("/", 1)[1][:-4]
                data = whistle.wavs.get(name)
                if data is None:
                    self._json(404, {"error": "まだ録音がありません"})
                else:
                    self._send(200, data, "audio/wav")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": "bad json"})
            routes = {
                "/api/actuator": lambda: bench.actuator(req.get("pwm", 0.0)),
                "/api/fan": lambda: bench.fan(req.get("on", False), req.get("duty", .5)),
                "/api/valve": lambda: bench.valve(req.get("open", False)),
                "/api/servo": lambda: bench.servo_positions(req["open"], req["closed"], req.get("move_ms", 150)),
                "/api/stop": bench.stop,
                "/api/whistle/start": lambda: whistle.start(req.get("seconds", 5), req.get("quantize", False),
                                                            req.get("step")),
                "/api/whistle/calibrate": lambda: whistle.calibrate(req.get("step")),
            }
            fn = routes.get(self.path)
            if fn is None:
                return self._json(404, {"error": "not found"})
            self._run(fn)

        def _run(self, fn):
            try:
                self._json(200, fn())
            except (OSError, RemoteError, KeyError, ValueError) as exc:
                self._json(503, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--server-port", type=int, default=8765)
    ap.add_argument("--no-calibrate-at-start", action="store_true",
                    help="do not calibrate the flute once the page starts (it sounds the flute for ~10 s)")
    args = ap.parse_args()
    bench = Bench(args.server_port)
    whistle = WhistleJob(bench, args.server_port, load_profiles())
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(bench, whistle))
    if not args.no_calibrate_at_start:
        whistle.calibrate_at_start()
    print(f"yamabiko web on port {args.port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        with bench.lock:
            bench._drop()


if __name__ == "__main__":
    main()
