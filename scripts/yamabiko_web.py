"""UNO Q: the whistle copy page (/) and a page to try the servo, the actuator and the fan by hand (/bench).

    python3 scripts/yamabiko_web.py [--port 8080]      # then open http://<UNO Q address>:8080/

The whistle copy runs any step (models/<step>/profile.json: classic, step1, step2, ...), chosen on the page
and remembered in runs/web_step.txt.  The board's EXEC button drives it: press to record, press again to end
the recording and play, press while calibrating or playing to stop (WhistleJob).  The page's big button does
the same.  Started at boot by the UNO Q's crontab (@reboot, after the command server).

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
<p><a href="/">← 演奏の画面へ</a></p>
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


class Stopped(Exception):
    """The board button or the page stopped the job."""


MAX_RECORD_S = 30.0       # a recording ends at the second press, or here
STEP_FILE = pathlib.Path(__file__).resolve().parents[1] / "runs" / "web_step.txt"   # the step chosen last


class WhistleJob:
    """One job at a time, in a thread: calibrate, or record a whistle until the button (or the page) says
    stop, analyse it with the chosen step and play it back.  The board button drives it (``press``):

        idle -> record      the first press starts recording (after calibrating if this step has not been)
        record -> play      the second press ends the recording; analyse, then play
        otherwise -> stop   a press while calibrating, analysing or playing stops (the server watches the
                            button itself during tune and play, which keep it busy)
    """

    BUSY_SERVER = ("tune", "play")          # phases the command server is busy in (it watches the button)

    def __init__(self, bench, server_port, profiles):
        self.bench, self.server_port = bench, server_port
        self.profiles = profiles                  # models/<step>/profile.json, any of them runnable
        self.step = self._saved_step() or latest(profiles)
        self.lock = threading.Lock()
        self.state = {"phase": "idle", "message": "ボタンを押すと録音が始まります"}
        self.thread = None
        self.end_recording = threading.Event()
        self.stopped = False
        self.tunings = {}         # per step, the last calibration: shift_cents, home / top pitch, the sweep
        self.wavs = {}            # "whistle" / "flute": WAV bytes of the last copy
        self.button_base = None   # the press count after a tune / play (the server consumed those presses)

    # ----- the chosen step ---------------------------------------------------------------------------
    def _saved_step(self):
        try:
            name = STEP_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return name if name in self.profiles else None

    def choose(self, step):
        if step not in self.profiles:
            raise ValueError(f"{step} がありません")
        if self.busy():
            raise ValueError("動いている間は版を変えられません")
        self.step = step
        try:
            STEP_FILE.parent.mkdir(parents=True, exist_ok=True)
            STEP_FILE.write_text(step, encoding="utf-8")
        except OSError:
            pass
        self._set(phase="idle", message=f"{step} にしました。ボタンを押すと録音が始まります")
        return {"step": step}

    # ----- requests (the page, or the board button through press) -----------------------------------
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def _launch(self, target, *args, first="準備中"):
        with self.lock:
            if self.busy():
                raise ValueError("まだ前の作業が動いています")
            self.stopped = False
            self.end_recording.clear()
            self.state = {"phase": "prepare", "message": first}
            self.thread = threading.Thread(target=target, args=args, daemon=True)
            self.thread.start()
        return {"started": True}

    def start(self, quantize=False, step=None):
        if step:
            self.choose(step) if step != self.step else None
        return self._launch(self._run, bool(quantize), self.profiles[self.step])

    def calibrate(self, step=None):
        if step and step != self.step:
            self.choose(step)
        return self._launch(self._calibrate, self.profiles[self.step], first="キャリブレーション中")

    def finish(self):
        """End the recording now (the page's button, or the board button's second press)."""
        self.end_recording.set()
        return {}

    def stop(self):
        """Stop whatever runs.  While the server is busy (tune, play) only the board button reaches it."""
        self.stopped = True
        self.end_recording.set()
        return {"server_busy": self.state.get("phase") in self.BUSY_SERVER}

    def press(self):
        phase = self.state.get("phase")
        if not self.busy():
            self.start(self.state.get("quantize", False))
        elif phase == "record":
            self.finish()
        else:
            self.stop()

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
                self.calibrate()
            except ValueError:
                pass
        threading.Thread(target=go, daemon=True).start()

    def watch_button(self, period_s=.15):
        """Poll the board button through a connection of its own and act on every new press."""
        def go():
            rig, seen = None, None
            while True:
                time.sleep(period_s)
                if self.state.get("phase") in self.BUSY_SERVER:
                    continue                           # the server is busy and watches the button itself
                try:
                    if rig is None:
                        rig = Remote(port=self.server_port, forward=False, timeout=3.0)
                    count = rig.status().get("button_count")
                except Exception:                      # noqa: BLE001  (the server restarts, a timeout)
                    if rig is not None:
                        try:
                            rig.close()
                        except Exception:              # noqa: BLE001
                            pass
                    rig = None                         # keep ``seen``: a press meanwhile still counts
                    time.sleep(.5)
                    continue
                if count is None:
                    continue
                if self.button_base is not None:       # presses up to here went to the server (tune, play)
                    seen, self.button_base = self.button_base, None
                if seen is None:
                    seen = count
                    continue
                if count != seen:
                    seen = count
                    try:
                        self.press()
                    except ValueError:
                        pass
        threading.Thread(target=go, daemon=True).start()

    def status(self):
        with self.lock:
            return dict(self.state, tuning=self.tunings.get(self.step), has_audio=sorted(self.wavs), step=self.step,
                        busy=self.busy(),
                        steps=[{"name": n, "title": p.get("title", n), "ear": p.get("ear", "yin"),
                                "memory": bool(p.get("memory_runtime")), "valve_nn": bool(p.get("valve_runtime"))}
                               for n, p in self.profiles.items()])

    # ----- the jobs -----------------------------------------------------------------------------------
    def _set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def _check(self):
        if self.stopped:
            raise Stopped("止めました")

    def _tune(self, rig, profile):
        self._set(phase="tune", message="キャリブレーション中（縮み切りの音と、上下に 1 往復）")
        try:
            result = rig.tune(runtime=profile["runtime_path"], fan=profile["fan"], stop_on_button=True,
                              **common_ear(profile))
        except RemoteError as exc:
            if "button" in str(exc):
                raise Stopped("ボタンで止めました") from None
            raise
        finally:
            self._note_buttons(rig)
        tuning = {k: result[k] for k in ("shift_cents", "home_cents", "top_cents", "sweep_max_cents", "heard", "phase")}
        tuning["time"] = time.strftime("%H:%M")
        self.tunings[profile["name"]] = tuning

    def _note_buttons(self, rig):
        """After a tune or a play: the press count now, so the watcher does not act on the presses the server
        already acted on (it stopped)."""
        try:
            self.button_base = rig.status().get("button_count")
        except Exception:                              # noqa: BLE001
            pass

    def _connect(self, profile):
        with self.bench.lock:                          # the bench page's connection lets go: one driver at a time
            self.bench._drop()
        rig = Remote(port=self.server_port, forward=False)
        rig.valve_positions(*profile["valve_us"], move_ms=20)
        return rig

    def _calibrate(self, profile):
        rig = None
        try:
            rig = self._connect(profile)
            self._tune(rig, profile)
            self._set(phase="idle", message="キャリブレーションできました。ボタンを押すと録音が始まります")
        except Stopped as exc:
            self._set(phase="idle", message=str(exc))
        except Exception as exc:                       # noqa: BLE001
            self._set(phase="error", message=f"キャリブレーションできませんでした: {exc}")
        finally:
            self._release(rig)

    def _release(self, rig):
        if rig is not None:
            try:
                rig.stop(force=True)
            except Exception:                          # noqa: BLE001
                pass
            finally:
                rig.close()

    def _run(self, quantize, profile):
        import numpy as np
        from flute_rl.yamabiko.whistle import clean_track, to_target, whistle_track
        rig = None
        try:
            rig = self._connect(profile)
            if profile["name"] not in self.tunings:   # not calibrated since the page started
                self._tune(rig, profile)
            self._check()
            rig.stop(force=True)                       # fan off, valve shut: only the whistle in the mic
            time.sleep(.3)                             # the click of the button
            start = rig.audio_mark()
            began = time.monotonic()
            self._set(phase="record", message="口笛をどうぞ。吹き終わったらボタンを押してください",
                      record_started=time.time(), max_seconds=MAX_RECORD_S)
            self.end_recording.wait(MAX_RECORD_S)
            self._check()
            pcm, sr, _ = rig.audio_since(start)
            if len(pcm) > sr:
                pcm = pcm[:-int(.25 * sr)]             # the click of the second press
            self.wavs = {"whistle": wav_bytes(pcm, sr)}
            stamp = time.strftime("%Y%m%d_%H%M%S")
            WHISTLE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(WHISTLE_DIR / f"{stamp}_whistle.npz", pcm=pcm, sr=sr)   # data for the ear (step5)
            self._set(phase="analyse", message=f"聞き取り中（{time.monotonic() - began:.1f} 秒の録音）")
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
            self._check()
            path = WHISTLE_DIR / f"{stamp}_target.npz"
            np.savez(path, cents=target, voice=voice, whistle_cents=raw)

            def listed(a):
                return [None if not np.isfinite(v) else round(float(v), 1) for v in a]
            self._set(phase="play", message="演奏中（ボタンで止められます）", whistle=listed(raw), cleaned=listed(cents),
                      target=[round(float(v), 1) for v in target], voice=[bool(v) for v in voice], report=report,
                      heard=None, relative_mae=None)
            try:
                result = rig.play(profile["runtime_path"], str(path), out=str(WHISTLE_DIR / f"{stamp}_play.npz"),
                                  repeats=1, fan=profile["fan"], shift_cents=self.tunings[profile["name"]]["shift_cents"],
                                  warmup_s=2.0, ear=profile.get("ear", "yin"), stop_on_button=True,
                                  valve_runtime=profile.get("valve_runtime_path"),
                                  ear_runtime=profile.get("ear_runtime_path"))
            finally:
                self._note_buttons(rig)
            played = np.load(WHISTLE_DIR / f"{stamp}_play.npz")
            if "0/pcm" in played:
                self.wavs["flute"] = wav_bytes(played["0/pcm"], int(played["sr"]))
            h, v, tc = played["0/heard"][0], played["0/valid"][0], played["0/target_cents"][0]
            m = v & voice[:len(v)]
            key = float(np.median((h - tc)[m])) if m.any() else 0.0
            rel = float(np.abs(h - tc - key)[m].mean()) if m.any() else None
            shift = float(result.get("shift_cents", 0.0))
            self._set(phase="done", message="ボタンで止めました" if result.get("stopped") else "できました",
                      heard=[round(float(x - shift), 1) if ok else None for x, ok in zip(h, v)],
                      key=key, relative_mae=rel, shift=shift, stamp=stamp)
        except Stopped as exc:
            self._set(phase="idle", message=str(exc))
        except Exception as exc:                       # noqa: BLE001  (shown on the page)
            self._set(phase="error", message=str(exc))
        finally:
            self._release(rig)


MAIN_PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>やまびこ1号</title>
<style>
:root{--bg:#f3f5f4;--card:#fff;--fg:#17201d;--mute:#5f6b66;--line:#dde3e0;--acc:#1f7a6c;--acc2:#e7f2ef;--rec:#c2410c;--warn:#b42318;--ok:#15803d;
--c1:#9aa5a1;--c2:#1f7a6c;--c3:#17201d;--c4:#d9622b;--onacc:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#121816;--card:#1a221f;--fg:#e6ece9;--mute:#9aa8a2;--line:#2b3531;--acc:#4fc3ad;--acc2:#1f2e2a;--rec:#fb923c;--warn:#f87171;--ok:#4ade80;
--c1:#6b7773;--c2:#4fc3ad;--c3:#e6ece9;--c4:#fb923c;--onacc:#0f1513}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,"Hiragino Sans","Yu Gothic UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:14px}
header{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}h1{font-size:24px;margin:0;letter-spacing:.02em}
.mute{color:var(--mute);font-size:14px}a{color:var(--acc)}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
h2{font-size:16px;margin:0 0 10px}
.hero{display:grid;gap:12px}
.phase{display:flex;align-items:center;gap:10px;font-size:15px;color:var(--mute)}
.dot{width:12px;height:12px;border-radius:50%;background:var(--mute)}
.dot.record{background:var(--rec);animation:pulse 1s infinite}.dot.play,.dot.tune,.dot.analyse,.dot.prepare{background:var(--acc);animation:pulse 1.4s infinite}
.dot.done{background:var(--ok)}.dot.error{background:var(--warn)}
@keyframes pulse{50%{opacity:.35}}@media (prefers-reduced-motion:reduce){.dot{animation:none!important}}
#msg{font-size:24px;font-weight:700;line-height:1.35;min-height:1.4em}
.bar{height:8px;background:var(--acc2);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;width:0;background:var(--rec)}
.flow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.flow div{border:1px solid var(--line);border-radius:10px;padding:8px 10px;font-size:14px;color:var(--mute)}
.flow b{display:block;color:var(--fg);font-size:15px}.flow div.on{border-color:var(--acc);background:var(--acc2);color:var(--fg)}
@media (max-width:560px){.flow{grid-template-columns:1fr}}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{font:inherit;border:0;border-radius:12px;cursor:pointer}
button:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
#main{font-size:20px;font-weight:700;padding:16px 28px;background:var(--acc);color:var(--onacc);min-width:240px}
#main.record{background:var(--rec)}#main.stop{background:var(--warn)}
button.sub{background:transparent;color:var(--fg);border:1px solid var(--line);padding:12px 16px}
button:disabled{opacity:.45;cursor:default}
label{color:var(--mute);font-size:14px}
.steps{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.step{text-align:left;background:var(--bg);border:2px solid transparent;padding:10px 12px;display:flex;flex-direction:column;gap:4px;color:var(--fg)}
.step b{font-size:16px}.step span{font-size:13px;color:var(--mute);line-height:1.4}
.step.on{border-color:var(--acc);background:var(--acc2)}
.tags{display:flex;gap:4px;flex-wrap:wrap}.tags em{font-style:normal;font-size:11px;padding:1px 6px;border-radius:6px;border:1px solid var(--line);color:var(--mute)}
svg{width:100%;height:auto;display:block}.legend span{display:inline-block;margin-right:14px;font-size:13px;color:var(--mute)}
.sw{display:inline-block;width:14px;height:4px;vertical-align:middle;margin-right:5px;border-radius:2px}
.num{font-variant-numeric:tabular-nums}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}@media (max-width:760px){.grid2{grid-template-columns:1fr}}
</style></head><body><main>
<header><h1>やまびこ1号</h1><span class="mute">口笛を聞いて、スライドホイッスルでまねて吹きます</span></header>

<section class="hero">
  <div class="phase"><i class="dot" id="dot"></i><span id="phase">接続中</span><span class="mute" id="stepnow"></span></div>
  <div id="msg"></div>
  <div class="bar" id="bar" hidden><i id="barfill"></i></div>
  <div class="flow" id="flow">
    <div data-p="idle"><b>① ボタンを押す</b>録音が始まります</div>
    <div data-p="record"><b>② 吹き終わったらもう一度</b>録音を終えて、聞き取りと演奏へ</div>
    <div data-p="play"><b>③ 演奏中に押すと止まる</b>キャリブレーション中も同じ</div>
  </div>
  <div class="row">
    <button id="main">録音スタート</button>
    <button id="cal" class="sub">キャリブレーション</button>
    <label><input type="checkbox" id="quant"> 半音にそろえる</label>
  </div>
  <div class="mute" id="hint">基板の EXEC ボタンと、この画面のボタンは同じ働きです。版の選択だけはこの画面で行います。</div>
</section>

<section>
  <h2>使う版</h2>
  <div class="steps" id="steps"></div>
  <div class="mute" style="margin-top:8px">選んだ版は覚えておき、次に起動したときも使います。</div>
</section>

<div class="grid2">
<section><h2>聞いた口笛</h2>
<div class="legend"><span><i class="sw" style="background:var(--c1)"></i>耳の聞いた音程</span><span><i class="sw" style="background:var(--c2)"></i>整えた後</span></div>
<svg id="g1" viewBox="0 0 860 260"></svg><div class="mute num" id="info1"></div>
<div class="row" style="margin-top:8px"><button class="sub" id="pw" disabled>▶ 口笛を聞く</button></div></section>
<section><h2>笛の演奏</h2>
<div class="legend"><span><i class="sw" style="background:var(--c3)"></i>目標（笛の音域へ移した口笛）</span><span><i class="sw" style="background:var(--c4)"></i>笛が鳴らした音</span></div>
<svg id="g2" viewBox="0 0 860 260"></svg><div class="mute num" id="info2"></div>
<div class="row" style="margin-top:8px"><button class="sub" id="pf" disabled>▶ 笛の演奏を聞く</button></div></section>
</div>

<section><h2>キャリブレーション（その日の笛）</h2>
<div class="legend"><span><i class="sw" style="background:var(--c2)"></i>縮み切り → 伸び切り → 縮み切り（PWM 0.6）</span></div>
<svg id="g3" viewBox="0 0 860 200"></svg><div class="mute num" id="info3">まだ測っていません</div></section>

<p class="mute">縦軸はセント（100 セント = 半音、0 = ラ 440 Hz）。 <a href="/bench">手動テストのページ</a></p>
<audio id="au"></audio>
</main><script>
const $=id=>document.getElementById(id);
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
const PHASE={idle:"待機中",prepare:"準備中",tune:"キャリブレーション中",record:"録音中",analyse:"聞き取り中",play:"演奏中",done:"完了",error:"エラー"};
function plot(svg,series,H){
  H=H||260;const W=860,L=56,R=10,T=10,B=30;let lo=Infinity,hi=-Infinity,n=1;
  for(const s of series){n=Math.max(n,s.y.length);for(const v of s.y){if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v)}}}
  if(!isFinite(lo)){svg.innerHTML='';return}
  const pad=Math.max(50,(hi-lo)*.08);lo-=pad;hi+=pad;
  const x=i=>L+(W-L-R)*i/Math.max(1,n-1),y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  const step=(hi-lo)>1600?400:(hi-lo)>700?200:100;let g="";
  for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step){g+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="${css('--line')}"/><text x="${L-6}" y="${y(v)+4}" font-size="12" text-anchor="end" fill="${css('--mute')}">${v}</text>`}
  const every=n>1500?5:n>600?2:1;
  for(let s=0;s<=n/100;s+=every){g+=`<text x="${x(s*100)}" y="${H-10}" font-size="12" text-anchor="middle" fill="${css('--mute')}">${s}s</text>`}
  for(const s of series){
    if(s.dots){for(let i=0;i<s.y.length;i++){if(s.y[i]!==null)g+=`<circle cx="${x(i)}" cy="${y(s.y[i])}" r="1.6" fill="${s.c}"/>`}continue}
    let d="",up=false;for(let i=0;i<s.y.length;i++){const v=s.y[i];if(v===null){up=false;continue}d+=(up?"L":"M")+x(i).toFixed(1)+" "+y(v).toFixed(1);up=true}
    g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="${s.w||2}" stroke-linejoin="round"/>`}
  svg.innerHTML=g}
async function api(path,body){const r=await fetch("/api/whistle/"+path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});
  if(!r.ok){$("msg").textContent=(await r.json()).error}return r.ok}
let S={},shownTune=null,shownStamp=null,stepsKey="";
function renderSteps(s){
  const key=JSON.stringify(s.steps)+s.step+s.busy;if(key===stepsKey)return;stepsKey=key;
  $("steps").innerHTML=s.steps.map(p=>{const t=[p.ear==="nn"?"耳 NN":"耳 YIN",p.valve_nn?"弁 NN":"弁 規則",...(p.memory?["記憶"]:[])];
    const title=(p.title||"").replace(/^[^:：]+[:：]\s*/,"");
    return `<button class="step${p.name===s.step?" on":""}" data-s="${p.name}" ${s.busy?"disabled":""}><b>${p.name}</b><span>${title}</span><span class="tags">${t.map(x=>`<em>${x}</em>`).join("")}</span></button>`}).join("");
  for(const b of document.querySelectorAll(".step"))b.onclick=()=>api("step",{step:b.dataset.s})}
async function poll(){
  let s;try{s=await (await fetch("/api/whistle/status")).json()}catch(e){$("phase").textContent="接続できません";setTimeout(poll,1500);return}
  S=s;const ph=s.phase||"idle";
  $("dot").className="dot "+ph;$("phase").textContent=PHASE[ph]||ph;$("stepnow").textContent=s.step?"・"+s.step:"";
  $("msg").textContent=s.message||"";$("msg").style.color=ph==="error"?css("--warn"):"";
  const stage=ph==="record"?"record":["tune","analyse","play","prepare"].includes(ph)?"play":"idle";
  for(const d of $("flow").children)d.classList.toggle("on",d.dataset.p===stage);
  if(ph==="record"&&s.record_started){$("bar").hidden=false;$("barfill").style.width=Math.min(100,(Date.now()/1000-s.record_started)/s.max_seconds*100)+"%"}else{$("bar").hidden=true}
  const m=$("main");
  if(!s.busy){m.textContent="録音スタート";m.className=""}else if(ph==="record"){m.textContent="録音終了して演奏";m.className="record"}else{m.textContent="止める";m.className="stop"}
  m.disabled=s.busy&&["tune","play"].includes(ph);
  $("hint").textContent=m.disabled?"いまは基板の EXEC ボタンで止められます（画面からは届きません）。":"基板の EXEC ボタンと、この画面のボタンは同じ働きです。版の選択だけはこの画面で行います。";
  $("cal").disabled=s.busy;
  renderSteps(s);
  if(s.whistle){plot($("g1"),[{y:s.whistle,c:css("--c1"),dots:true},{y:s.cleaned,c:css("--c2")}]);
    const r=s.report||{};$("info1").textContent=r.whistle_span_cents!=null?`口笛の幅 ${r.whistle_span_cents.toFixed(0)} セント／笛の音域 ${r.range_cents.toFixed(0)} セント`+(r.clipped_fraction>0?`（はみ出し ${(r.clipped_fraction*100).toFixed(0)} %）`:""):r.memory?`記憶に ${r.listened_steps} ステップ書き込み、そこから思い出した楽譜`:""}
  else{$("g1").innerHTML="";$("info1").textContent=""}
  if(s.target){const tgt=s.target.map((v,i)=>s.voice[i]?v:null);
    plot($("g2"),[{y:tgt,c:css("--c3"),w:3},...(s.heard?[{y:s.heard.map((v,i)=>s.voice[i]?v:null),c:css("--c4"),dots:true}]:[])]);
    $("info2").textContent=s.relative_mae!=null?`調のずれ ${s.key>=0?"+":""}${s.key.toFixed(0)} セント／それを除いた平均の誤差 ${s.relative_mae.toFixed(0)} セント`:""}
  else{$("g2").innerHTML="";$("info2").textContent=""}
  const tu=s.tuning;if(!tu){shownTune=null;$("g3").innerHTML="";$("info3").textContent="この版ではまだ測っていません（最初の録音の前に測ります）"}
  if(tu&&tu.time!==shownTune){shownTune=tu.time;plot($("g3"),[{y:tu.heard,c:css("--c2"),dots:true}],200);
    const hz=c=>(440*Math.pow(2,c/1200)).toFixed(0);
    $("info3").textContent=`${tu.time} に測定：縮み切り ${hz(tu.home_cents)} Hz、いちばん高い音 ${tu.sweep_max_cents!=null?hz(tu.sweep_max_cents):"-"} Hz、ツインとのずれ ${tu.shift_cents>=0?"+":""}${tu.shift_cents.toFixed(0)} セント`}
  const ha=s.has_audio||[];$("pw").disabled=s.busy||!ha.includes("whistle");$("pf").disabled=s.busy||!ha.includes("flute");
  setTimeout(poll,s.busy?250:1000)}
$("main").onclick=()=>{if(!S.busy)api("start",{quantize:$("quant").checked});else if(S.phase==="record")api("finish");else api("stop")};
$("cal").onclick=()=>api("calibrate");
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
            if self.path in ("/", "/index.html", "/whistle"):
                self._send(200, MAIN_PAGE, "text/html; charset=utf-8")
            elif self.path == "/bench":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._run(bench.status)
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
                "/api/whistle/start": lambda: whistle.start(req.get("quantize", False), req.get("step")),
                "/api/whistle/finish": whistle.finish,
                "/api/whistle/stop": whistle.stop,
                "/api/whistle/step": lambda: whistle.choose(req["step"]),
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
    ap.add_argument("--no-button", action="store_true", help="do not start and stop with the board's EXEC button")
    args = ap.parse_args()
    bench = Bench(args.server_port)
    whistle = WhistleJob(bench, args.server_port, load_profiles())
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(bench, whistle))
    if not args.no_calibrate_at_start:
        whistle.calibrate_at_start()
    if not args.no_button:
        whistle.watch_button()
    print(f"yamabiko web on port {args.port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        with bench.lock:
            bench._drop()


if __name__ == "__main__":
    main()
