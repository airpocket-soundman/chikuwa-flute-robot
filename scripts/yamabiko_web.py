"""UNO Q: the whistle copy page (/) and a page to try the servo, the actuator and the fan by hand (/bench).

    python3 scripts/yamabiko_web.py [--port 8080]      # then open http://<UNO Q address>:8080/

The whistle copy runs any step (models/<step>/profile.json: classic, step1, step2, ...), chosen on the page
and remembered in runs/web_step.txt.  The board's EXEC button drives it: press to record, press again to end
the recording and play, press while calibrating or playing to stop (WhistleJob).  The page's big button does
the same.  Started at boot by the UNO Q's crontab (@reboot, after the command server).

The page also plays archive clips (a whistled language, Silbo Gomero, and Japanese tit calls): the example is
heard from the page (the clip, and the clip at the flute's speed) and the flute plays its score with the chosen
step; pressing twice plays it again.  The clips are made on the PC by scripts/archive_pack.py into
runs/archive_pack/ and copied by scripts/uno_q_push.sh.

The touch panel (M5Stack Tab5 on the B board's I2C, tab5/yamabiko_panel/) shows the same: PanelBridge sends
it a page (flute_rl/yamabiko/panel.py) whenever the status changes and acts on what it asks for (choose a
step, calibrate, play an archive clip).  Its record / again / stop buttons are presses of the EXEC button
in the MCU, so they work as the board's button does, also while the flute plays.

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
ARCHIVE_PACK = pathlib.Path(__file__).resolve().parents[1] / "runs" / "archive_pack"   # scripts/archive_pack.py


def common_ear(profile):
    """Server arguments for the profile's ear when it hears everything (step5: "common_ear": true), else {}."""
    if profile.get("common_ear") and profile.get("ear") == "nn":
        return {"ear": "nn", "ear_runtime": profile["ear_runtime_path"]}
    return {}
def np_load(path):
    import numpy as np
    return np.load(path)


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
COLLECT_DIR = pathlib.Path(__file__).resolve().parents[1] / "runs" / "real" / "whistle_data"   # recordings only
COLLECT_KINDS = {"whistle": "口笛", "quiet": "無音・部屋の音", "other": "話し声など"}
STEP_FILE = pathlib.Path(__file__).resolve().parents[1] / "runs" / "web_step.txt"   # the step chosen last


class WhistleJob:
    """One job at a time, in a thread: calibrate, record a whistle until the button (or the page) says stop,
    analyse it with the chosen step and play it, or play the last song again.  The board button drives it:

        idle, one press      record (after calibrating if this step has not been): a new song, practice reset
        record, a press      end the recording; analyse, then play (practice play 1)
        idle, two presses    play the last song again without recording (practice): the server keeps the
                             performer's song memory between these plays, so a step with one gets better
        otherwise, a press   stop (calibrating, analysing, playing: the server watches the button itself
                             during tune and play, which keep it busy)

    Every press is also put in the status (``press``: its number and what it did), so the page shows the
    same button pressed.  The page's buttons call the same actions.
    """

    BUSY_SERVER = ("tune", "play")          # phases the command server is busy in (it watches the button)
    DOUBLE_S = .5                           # a second press within this is a double press
    QUIET_S = 1.5                           # presses this soon after a stop belong to the stop

    def __init__(self, bench, server_port, profiles):
        self.bench, self.server_port = bench, server_port
        self.profiles = profiles                  # models/<step>/profile.json, any of them runnable
        self.step = self._saved_step() or latest(profiles)
        self.lock = threading.Lock()
        self.state = {"phase": "idle", "message": "ボタンを押すと録音が始まります"}
        self.thread = None
        self.end_recording = threading.Event()
        self.stopped = False
        self.tuning = None        # the last calibration of the rig (any step): home / top pitch, the sweep
        self.twin_home = {}       # per step, the home pitch its twin expects (shift = measured home - this)
        self.wavs = {}            # "whistle" / "flute": WAV bytes of the last copy
        self.button_base = None   # the press count after a tune / play (the server consumed those presses)
        self.song = None          # the last recorded song: its samples, and the step, target path it was analysed for
        self.practice = []        # relative error of every play of that song (practice play 1, 2, ...)
        self.plays = []           # every play of that song for the page: n, heard (as the target), error
        self.presses = 0
        self.collect = False      # the data collection mode: record only, no calibration, no play
        self.collect_kind, self.collect_seconds = "whistle", MAX_RECORD_S
        self.takes = self._load_takes()
        self.panel = None         # PanelBridge, when the touch panel runs
        self._restore_song()

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
            keep = ("press", "whistle", "cleaned", "target", "voice", "report", "heard", "key", "relative_mae", "shift",
                    "stamp", "plays_n", "plays_id", "archive")       # the last song's graphs stay while it is played again
            self.state = {"phase": "prepare", "message": first, **{k: self.state[k] for k in keep if k in self.state}}
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

    def _pressed(self, action, presses):
        """Put a board-button press in the status for the page: its number, single or double, the action."""
        self.presses += 1
        if action == "stop":                   # a press or two more (the stop seemed slow) must not start a play
            self.quiet_until = time.monotonic() + self.QUIET_S
        self._set(press={"n": self.presses, "action": action, "double": presses >= 2})

    def press(self, presses=1):
        """The board button: ``presses`` within DOUBLE_S of each other (1 or 2)."""
        phase = self.state.get("phase")
        if self.collect:                               # data collection: every press starts or ends a take
            if not self.busy():
                self._pressed("record", 1)
                try:
                    self.start_take()
                except ValueError as exc:
                    self._set(phase="idle", message=str(exc))
            elif phase == "record":
                self._pressed("finish", 1)
                self.finish()
            return
        if not self.busy():
            if time.monotonic() < getattr(self, "quiet_until", 0.0):
                self._set(message="止めました（止めた直後の押下は、続きの「止める」として扱います）")
                return
            action = "replay" if presses >= 2 else "record"
            self._pressed(action, presses)
            try:
                self.replay() if presses >= 2 else self.start(self.state.get("quantize", False))
            except ValueError as exc:
                self._set(phase="idle", message=str(exc))
        elif phase == "record":
            self._pressed("finish", presses)
            self.finish()
        else:
            self._pressed("stop", presses)
            self.stop()

    def _restore_song(self):
        """After a restart, the last whistle recorded is the song again: any step can play it (it is heard again
        with that step on the first play)."""
        import numpy as np
        try:
            last = max(WHISTLE_DIR.glob("*_whistle.npz"))
            with np.load(last) as d:
                pcm, sr = d["pcm"].copy(), int(d["sr"])
        except (ValueError, OSError, KeyError):
            return
        stamp = last.stem[:-len("_whistle")]
        self.song = {"step": None, "path": None, "voice": None, "stamp": stamp, "pcm": pcm, "sr": sr, "quantize": False}
        self.wavs["whistle"] = wav_bytes(pcm, sr)
        self._set(message=f"最後の録音（{stamp}）を読み込みました。2 回押すと、選んだ版で吹きます")

    # ----- archive clips (scripts/archive_pack.py) ---------------------------------------------------
    def archive_list(self):
        """The clips in runs/archive_pack/: their meta.json (title, what it says, credit, licence, ...)."""
        items = []
        for meta in sorted(ARCHIVE_PACK.glob("*/meta.json")) if ARCHIVE_PACK.exists() else []:
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (meta.parent / "target.npz").exists():
                m["id"] = meta.parent.name
                m["has_slow"] = (meta.parent / "source_slow.wav").exists()
                items.append(m)
        return items

    def archive_wav(self, item_id, name):
        """Bytes of an archive clip's source.wav or source_slow.wav (only ids of the pack, only those names)."""
        if name not in ("source", "source_slow") or item_id not in {m["id"] for m in self.archive_list()}:
            return None
        try:
            return (ARCHIVE_PACK / item_id / f"{name}.wav").read_bytes()
        except OSError:
            return None

    def archive_play(self, item_id, step=None):
        """Play an archive clip's score with the chosen step; it becomes the song (two presses: again)."""
        item = next((m for m in self.archive_list() if m["id"] == item_id), None)
        if item is None:
            raise ValueError(f"{item_id} がありません（PC で scripts/archive_pack.py を作って送ってください）")
        if step and step != self.step:
            self.choose(step)
        return self._launch(self._archive, self.profiles[self.step], item, first=f"{item['title']} を吹く準備")

    def _archive(self, profile, item):
        import numpy as np
        rig = None
        try:
            rig = self._connect(profile)
            if self.tuning is None:
                self._tune(rig, profile)
            self._check()
            folder = ARCHIVE_PACK / item["id"]
            with np.load(folder / "target.npz") as d:
                target, voice = d["cents"].astype(float), d["voice"].astype(bool)
                heard = d["source_cents"].astype(float) if "source_cents" in d.files else np.where(voice, target, np.nan)
            self.wavs = {"whistle": (folder / "source.wav").read_bytes()}
            self.song = {"step": profile["name"], "path": str(folder / "target.npz"), "voice": voice,
                         "stamp": f"archive_{item['id']}", "pcm": None, "sr": None, "quantize": False,
                         "fixed": True, "archive": item}
            self.practice, self.plays = [], []

            def listed(a):
                return [None if not np.isfinite(v) else round(float(v), 1) for v in a]
            span = float(np.nanpercentile(heard, 98) - np.nanpercentile(heard, 2)) if np.isfinite(heard).any() else 0.0
            report = {"whistle_span_cents": span, "range_cents": profile["high_cents"] - profile["low_cents"],
                      "clipped_fraction": 0.0, "archive": item["title"], "interval_scale": item.get("interval_scale"),
                      "slow": item.get("slow")}
            self._set(whistle=listed(heard), cleaned=listed(heard), target=[round(float(v), 1) for v in target],
                      voice=[bool(v) for v in voice], report=report, plays_n=0, plays_id=None, archive=item["id"])
            self._play(rig, profile)
        except Stopped as exc:
            self._set(phase="idle", message=str(exc))
        except Exception as exc:                       # noqa: BLE001
            self._set(phase="error", message=str(exc))
        finally:
            self._release(rig)

    # ----- data collection --------------------------------------------------------------------------
    def _load_takes(self):
        takes = []
        for f in sorted(COLLECT_DIR.glob("*.npz")) if COLLECT_DIR.exists() else []:
            try:
                with np_load(f) as d:
                    takes.append({"name": f.stem, "kind": str(d["kind"]) if "kind" in d.files else "whistle",
                                  "seconds": round(len(d["pcm"]) / int(d["sr"]), 1)})
            except Exception:                          # noqa: BLE001
                continue
        return takes

    def set_mode(self, collect=None, kind=None, seconds=None):
        if self.busy():
            raise ValueError("動いている間は切り替えられません")
        if collect is not None:
            self.collect = bool(collect)
        if kind is not None:
            if kind not in COLLECT_KINDS:
                raise ValueError(f"{kind} は選べません")
            self.collect_kind = kind
        if seconds is not None:
            self.collect_seconds = float(min(max(float(seconds), 5.0), MAX_RECORD_S))
        self._set(phase="idle", message="ボタンを押すと録音が始まります（録音だけ。笛は鳴りません）" if self.collect
                  else "ボタンを押すと録音が始まります")
        return {"collect": self.collect, "kind": self.collect_kind, "seconds": self.collect_seconds}

    def start_take(self):
        return self._launch(self._take, self.collect_kind, self.collect_seconds, first="録音の準備")

    def delete_last(self):
        if self.busy():
            raise ValueError("動いている間は消せません")
        if not self.takes:
            raise ValueError("消す録音がありません")
        take = self.takes.pop()
        try:
            (COLLECT_DIR / f"{take['name']}.npz").unlink()
        except OSError:
            pass
        self.wavs.pop("take", None)
        self._set(phase="idle", message=f"{take['name']} を消しました", take=None)
        return {"deleted": take["name"]}

    def _take(self, kind, seconds):
        import numpy as np
        from flute_rl.yamabiko.whistle import whistle_track
        rig = None
        try:
            with self.bench.lock:
                self.bench._drop()
            rig = Remote(port=self.server_port, forward=False)
            rig.stop(force=True)                       # fan off, valve shut: only the room in the mic
            time.sleep(.3)                             # the click of the button
            start = rig.audio_mark()
            began = time.monotonic()
            self._set(phase="record", message=f"{COLLECT_KINDS[kind]}を録音中。終わったらボタンを押してください",
                      record_began=began, max_seconds=seconds)
            self.end_recording.wait(seconds)
            self._check()
            pcm, sr, _ = rig.audio_since(start)
            if len(pcm) > sr:
                pcm = pcm[:-int(.25 * sr)]
            stamp = time.strftime("%Y%m%d_%H%M%S")
            COLLECT_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(COLLECT_DIR / f"{stamp}_{kind}.npz", pcm=pcm, sr=sr, kind=kind)
            take = {"name": f"{stamp}_{kind}", "kind": kind, "seconds": round(len(pcm) / sr, 1)}
            self.takes.append(take)
            self.wavs["take"] = wav_bytes(pcm, sr)
            x = pcm / 32768.0
            level = 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-9)) if len(x) else -120.0
            track = whistle_track(pcm, sr)
            heard = float(np.mean(np.isfinite(track))) if len(track) else 0.0
            self._set(phase="saved", message=f"保存しました（{take['seconds']:.1f} 秒）。ボタンで次を録音します",
                      take=dict(take, level_db=round(level, 1), voiced=round(heard, 2),
                                track=[None if not np.isfinite(v) else round(float(v), 1) for v in track[::2]]))
        except Stopped as exc:
            self._set(phase="idle", message=str(exc))
        except Exception as exc:                       # noqa: BLE001
            self._set(phase="error", message=str(exc))
        finally:
            self._release(rig)

    def replay(self, step=None):
        """Play the last recorded song again, without recording (practice)."""
        if self.song is None:
            raise ValueError("まだ録音がありません。1 回押して録音してください")
        return self._launch(self._replay, self.profiles[self.step], first="もう一度（練習）")

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
                    presses = (count - seen) % 256
                    seen = count
                    if not self.busy():                # idle: wait for a second press
                        until = time.monotonic() + self.DOUBLE_S
                        while time.monotonic() < until and presses < 2:
                            time.sleep(.05)
                            try:
                                now = rig.status().get("button_count")
                            except Exception:          # noqa: BLE001
                                break
                            if now is not None and now != seen:
                                presses += (now - seen) % 256
                                seen = now
                    try:
                        self.press(presses)
                    except ValueError:
                        pass
        threading.Thread(target=go, daemon=True).start()

    def status(self):
        with self.lock:
            tuning = dict(self.tuning, shift_cents=self.shift_for(self.profiles[self.step])) if self.tuning else None
            return dict(self.state, tuning=tuning, has_audio=sorted(self.wavs), step=self.step,
                        busy=self.busy(), practice=list(self.practice),
                        collect=self.collect, collect_kind=self.collect_kind, collect_seconds=self.collect_seconds,
                        takes=self.takes[-12:], take_count=len(self.takes),
                        take_seconds=round(sum(t["seconds"] for t in self.takes if t["kind"] == "whistle"), 1),
                        record_elapsed=(time.monotonic() - self.state["record_began"]
                                        if self.state.get("phase") == "record" and self.state.get("record_began")
                                        else None),
                        song_step=self.song["step"] if self.song else None, has_song=self.song is not None,
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
        if self.panel:
            self.panel.push_now()
        try:
            result = rig.tune(runtime=profile["runtime_path"], fan=profile["fan"], stop_on_button=True,
                              **common_ear(profile))
        except RemoteError as exc:
            if "button" in str(exc):
                self._pressed("stop", 1)
                raise Stopped("ボタンで止めました") from None
            raise
        finally:
            self._note_buttons(rig)
        tuning = {k: result[k] for k in ("home_cents", "top_cents", "sweep_max_cents", "heard", "phase")}
        tuning["time"] = time.strftime("%H:%M")
        self.tuning = tuning                           # the rig's, for every step

    def shift_for(self, profile):
        """Today's offset for a step: the measured home pitch minus the home its twin expects (as cmd_tune)."""
        import numpy as np
        from flute_rl.yamabiko.residual_numpy import runtime_for
        name = profile["name"]
        if name not in self.twin_home:
            self.twin_home[name] = float(runtime_for(dict(np.load(profile["runtime_path"]))).cents_at(0.0))
        return float(np.clip(self.tuning["home_cents"] - self.twin_home[name], -150, 150))

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

    def _play(self, rig, profile):
        """Play the last song (practice: the server keeps the song memory from its previous play)."""
        import numpy as np
        song = self.song
        n = len(self.practice) + 1
        self._set(phase="play", message=f"演奏中（練習 {n} 回目、ボタンで止められます）", heard=None, relative_mae=None)
        if self.panel:
            self.panel.push_now()
        out = WHISTLE_DIR / f"{song['stamp']}_{profile['name']}_play{n}.npz"
        try:
            result = rig.play(profile["runtime_path"], song["path"], out=str(out), repeats=1, fan=profile["fan"],
                              shift_cents=self.shift_for(profile), warmup_s=2.0,
                              ear=profile.get("ear", "yin"), stop_on_button=True, practice=True,
                              valve_runtime=profile.get("valve_runtime_path"),
                              ear_runtime=profile.get("ear_runtime_path"))
        finally:
            self._note_buttons(rig)
        played = np.load(out)
        if "0/pcm" in played:
            self.wavs["flute"] = wav_bytes(played["0/pcm"], int(played["sr"]))
        h, v, tc = played["0/heard"][0], played["0/valid"][0], played["0/target_cents"][0]
        m = v & song["voice"][:len(v)]
        key = float(np.median((h - tc)[m])) if m.any() else 0.0
        rel = float(np.abs(h - tc - key)[m].mean()) if m.any() else None
        shift = float(result.get("shift_cents", 0.0))
        if result.get("stopped"):
            self._pressed("stop", 1)
            message = "ボタンで止めました"
        else:
            self.practice.append(None if rel is None else round(rel, 1))
            memory = "（この版は練習の記憶を持っていないので、毎回同じように吹きます）" if not result.get("song_memory") else ""
            message = f"できました（練習 {n} 回目）。2 回押すともう一度、1 回押すと新しく録音します{memory}"
        heard = [round(float(x - shift - key), 1) if ok else None for x, ok in zip(h, v)]   # in the target's key
        if not result.get("stopped"):
            self.plays = (self.plays + [{"n": n, "heard": heard, "error": None if rel is None else round(rel, 1)}])[-8:]
        self._set(phase="done", message=message, heard=[round(float(x - shift), 1) if ok else None for x, ok in zip(h, v)],
                  key=key, relative_mae=rel, shift=shift, stamp=song["stamp"], plays_n=len(self.plays),
                  plays_id=f"{song['stamp']}:{profile['name']}:{n}", plays_step=profile["name"])

    def _replay(self, profile):
        rig = None
        try:
            rig = self._connect(profile)
            if self.tuning is None:
                self._tune(rig, profile)
            self._check()
            if self.song["step"] != profile["name"] and not self.song.get("fixed"):   # another step: hear it again
                self._set(phase="analyse", message=f"同じ録音を {profile['name']} で聞き取り中")
                self._analyse(self.song["pcm"], self.song["sr"], profile, self.song["quantize"], self.song["stamp"])
            self._play(rig, profile)
        except Stopped as exc:
            self._set(phase="idle", message=str(exc))
        except Exception as exc:                       # noqa: BLE001
            self._set(phase="error", message=str(exc))
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
            if self.tuning is None:                    # not calibrated since the page started
                self._tune(rig, profile)
            self._check()
            rig.stop(force=True)                       # fan off, valve shut: only the whistle in the mic
            time.sleep(.3)                             # the click of the button
            start = rig.audio_mark()
            began = time.monotonic()
            self._set(whistle=None, cleaned=None, target=None, voice=None, report=None, heard=None, relative_mae=None,
                      plays_n=0, plays_id=None, archive=None)   # a new song: the last one's graphs go
            self._set(phase="record", message="口笛をどうぞ。吹き終わったらボタンを押してください",
                      record_began=time.monotonic(), max_seconds=MAX_RECORD_S)
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
            self._analyse(pcm, sr, profile, quantize, stamp)
            self._play(rig, profile)
        except Stopped as exc:
            self._set(phase="idle", message=str(exc))
        except Exception as exc:                       # noqa: BLE001  (shown on the page)
            self._set(phase="error", message=str(exc))
        finally:
            self._release(rig)

    def _analyse(self, pcm, sr, profile, quantize, stamp):
        """The recording heard and made a target with this step; it becomes the song (practice starts over)."""
        import numpy as np
        from flute_rl.yamabiko.whistle import clean_track, to_target, whistle_track
        if True:
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
            path = WHISTLE_DIR / f"{stamp}_{profile['name']}_target.npz"
            np.savez(path, cents=target, voice=voice, whistle_cents=raw)

            def listed(a):
                return [None if not np.isfinite(v) else round(float(v), 1) for v in a]
            self.song = {"step": profile["name"], "path": str(path), "voice": voice, "stamp": stamp,
                         "pcm": pcm, "sr": sr, "quantize": quantize}
            self.practice = []
            self.plays = []
            self._set(whistle=listed(raw), cleaned=listed(cents), target=[round(float(v), 1) for v in target],
                      voice=[bool(v) for v in voice], report=report, plays_n=0, plays_id=None)


class PanelBridge:
    """The touch panel: a page whenever the web page's status changes, and its requests (events) acted on.
    Talks to the command server on its own connection; while the server is busy (tune, play) it waits, so
    WhistleJob calls push_now() just before those (the panel then shows the phase, with its own clock)."""

    PERIOD_S = .25
    RESEND_S = 5.0          # the page again even unchanged (the MCU loses it when it restarts or is flashed)

    def __init__(self, whistle, server_port):
        self.whistle, self.server_port = whistle, server_port
        self.lock = threading.Lock()
        self.remote = None
        self.version, self.sent, self.sent_at = 0, None, 0.0
        self.archive, self.archive_at = [], 0.0
        whistle.panel = self

    def _page(self):
        from flute_rl.yamabiko.panel import encode
        if time.monotonic() - self.archive_at > 30:
            self.archive, self.archive_at = self.whistle.archive_list(), time.monotonic()
        status = self.whistle.status()
        plays = self.whistle.plays
        heard = plays[-1]["heard"] if plays else None
        return encode(status, self.archive, heard)

    def _call(self, page=None):
        if self.remote is None:
            self.remote = Remote(port=self.server_port, forward=False, timeout=3.0)
        if page is not None:
            self.version = (self.version + 1) & 0xFFFF
            self.sent_at = time.monotonic()
        return self.remote.panel(page, self.version)

    def push_now(self):
        """Send the page now if it changed (before the server becomes busy)."""
        with self.lock:
            try:
                page = self._page()
                if page != self.sent:
                    events = self._call(page)
                    self.sent = page
                    self._act(events)
            except Exception:                          # noqa: BLE001  (the panel is optional)
                self._drop()

    def _drop(self):
        if self.remote is not None:
            try:
                self.remote.close()
            except Exception:                          # noqa: BLE001
                pass
        self.remote, self.sent = None, None

    def _act(self, events):
        from flute_rl.yamabiko.panel import EV_ARCHIVE, EV_CALIBRATE, EV_STEP
        for ev in events:
            if not ev:
                continue
            try:
                if ev[0] == EV_STEP and len(ev) > 1:
                    names = list(self.whistle.profiles)
                    if ev[1] < len(names):
                        self.whistle.choose(names[ev[1]])
                elif ev[0] == EV_CALIBRATE:
                    self.whistle.calibrate()
                elif ev[0] == EV_ARCHIVE and len(ev) > 1:
                    items = self.whistle.archive_list()
                    if ev[1] < len(items):
                        self.whistle.archive_play(items[ev[1]]["id"])
            except ValueError as exc:
                self.whistle._set(message=str(exc))

    def run(self):
        def go():
            while True:
                time.sleep(self.PERIOD_S)
                st = self.whistle.state
                if self.whistle.busy() and st.get("phase") in WhistleJob.BUSY_SERVER:
                    continue                           # the server plays: it would not answer
                with self.lock:
                    try:
                        page = self._page()
                        stale = time.monotonic() - self.sent_at > self.RESEND_S
                        events = self._call(page if page != self.sent or stale else None)
                        self.sent = page
                        self._act(events)
                    except Exception:                  # noqa: BLE001
                        self._drop()
                        time.sleep(2)
        threading.Thread(target=go, daemon=True).start()


MAIN_PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>やまびこ1号</title>
<style>
:root{--bg:#f3f5f4;--card:#fff;--fg:#17201d;--mute:#5f6b66;--line:#dde3e0;--acc:#1f7a6c;--acc2:#e7f2ef;--rec:#c2410c;--warn:#b42318;--ok:#15803d;
--c1:#9aa5a1;--c2:#1f7a6c;--c3:#17201d;--c4:#d9622b;--onacc:#fff;
--p1:#f2b38a;--p2:#e8875a;--p3:#d9622b;--p4:#b5481c;--p5:#8f3613;--p6:#6b270c}
@media (prefers-color-scheme:dark){:root{--bg:#121816;--card:#1a221f;--fg:#e6ece9;--mute:#9aa8a2;--line:#2b3531;--acc:#4fc3ad;--acc2:#1f2e2a;--rec:#fb923c;--warn:#f87171;--ok:#4ade80;
--c1:#6b7773;--c2:#4fc3ad;--c3:#e6ece9;--c4:#fb923c;--onacc:#0f1513;
--p1:#7a4a2c;--p2:#a35f33;--p3:#cc7738;--p4:#f0913f;--p5:#fbb071;--p6:#fdd2a8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,"Hiragino Sans","Yu Gothic UI",sans-serif}
[hidden]{display:none!important}
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
.act{font-size:19px;font-weight:700;padding:14px 22px;color:var(--onacc);display:flex;flex-direction:column;align-items:flex-start;gap:2px;min-width:200px;transition:transform .08s}
.act small{font-size:12px;font-weight:500;opacity:.85}
#rec,#take{background:var(--acc)}#rec.record,#take.record{background:var(--rec)}#again{background:var(--c2);filter:saturate(.7)}#stop{background:var(--warn)}
.act:active,.act.flash{transform:scale(.95);box-shadow:0 0 0 6px var(--acc2),0 0 0 8px var(--acc)}
.act.flash{animation:flash .7s}@keyframes flash{0%{filter:brightness(1.6)}100%{filter:none}}
@media (prefers-reduced-motion:reduce){.act.flash{animation:none}}
.modes{display:inline-flex;border:1px solid var(--line);border-radius:12px;padding:3px;gap:3px;background:var(--bg)}
.modes button{padding:8px 16px;background:transparent;color:var(--mute);font-weight:700}
.modes button[aria-pressed="true"]{background:var(--card);color:var(--fg);box-shadow:0 1px 3px rgba(0,0,0,.15)}
.remain{display:flex;flex-direction:column;gap:6px}
.remain .track{position:relative;height:34px;border-radius:17px;background:var(--acc2);overflow:hidden}
.remain .fill{position:absolute;inset:0 auto 0 0;background:var(--rec);border-radius:17px;transition:width .25s linear}
.remain .knob{position:absolute;top:3px;width:28px;height:28px;border-radius:50%;background:var(--card);box-shadow:0 1px 4px rgba(0,0,0,.35);transform:translateX(-50%);transition:left .25s linear}
.remain .nums{display:flex;justify-content:space-between;font-size:14px;color:var(--mute);font-variant-numeric:tabular-nums}
.remain .nums b{font-size:26px;color:var(--fg)}
.goal{height:10px;border-radius:5px;background:var(--acc2);overflow:hidden}.goal i{display:block;height:100%;background:var(--acc)}
.takes{display:flex;flex-direction:column;gap:4px;font-size:14px;font-variant-numeric:tabular-nums}
.takes div{display:flex;gap:12px;color:var(--mute)}.takes div b{color:var(--fg);font-weight:500}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chips button{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--fg);font-size:13px;font-variant-numeric:tabular-nums}
.chips button i{width:12px;height:12px;border-radius:50%}
.chips button[aria-pressed="false"]{opacity:.4}
.multi{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
.multi div{border:1px solid var(--line);border-radius:10px;padding:6px 8px;display:flex;flex-direction:column;gap:2px;font-size:13px}
.multi b{font-variant-numeric:tabular-nums}
.practice{display:flex;gap:10px;align-items:flex-end;min-height:84px;flex-wrap:wrap}
.practice div{display:flex;flex-direction:column;align-items:center;gap:4px;font-size:12px;color:var(--mute)}
.practice i{display:block;width:34px;background:var(--c4);border-radius:4px 4px 0 0}
.practice b{font-size:13px;color:var(--fg);font-variant-numeric:tabular-nums}
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
.arc{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:10px}
.arc>div{border:2px solid var(--line);border-radius:12px;padding:10px 12px;display:flex;flex-direction:column;gap:6px;background:var(--bg)}
.arc>div.on{border-color:var(--acc);background:var(--acc2)}
.arc b{font-size:15px;line-height:1.4}.arc q{font-size:13px;color:var(--fg);font-style:italic}
.arc small{font-size:11px;color:var(--mute);line-height:1.35;overflow-wrap:anywhere}.arc small a{color:var(--mute)}
.arc .row{gap:6px}.arc button.sub{padding:6px 10px;font-size:13px}.arc button.go{background:var(--acc);color:var(--onacc);padding:6px 12px;font-size:13px;font-weight:700}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}@media (max-width:760px){.grid2{grid-template-columns:1fr}}
</style></head><body><main>
<header><h1>やまびこ1号</h1><span class="mute">口笛を聞いて、スライドホイッスルでまねて吹きます</span>
<span class="modes" role="group" aria-label="モード" style="margin-left:auto"><button id="mPlay" aria-pressed="true">演奏</button><button id="mCollect" aria-pressed="false">録音だけ（データ集め）</button></span></header>

<section class="hero">
  <div class="phase"><i class="dot" id="dot"></i><span id="phase">接続中</span><span class="mute" id="stepnow"></span></div>
  <div id="msg"></div>
  <div class="remain" id="remain" hidden>
    <div class="nums"><span>残り <b id="remainS">30</b> 秒</span><span id="remainOf"></span></div>
    <div class="track"><div class="fill" id="remainFill"></div><div class="knob" id="remainKnob"></div></div>
  </div>
  <div class="flow" id="flow">
    <div data-p="idle"><b>① 1 回押す</b>録音が始まります（新しい曲）</div>
    <div data-p="record"><b>② 吹き終わったら 1 回押す</b>録音を終えて、聞き取りと演奏へ</div>
    <div data-p="again"><b>③ 2 回押す</b>録音せずにもう一度。練習を重ねるほどうまくなります</div>
  </div>
  <div class="row">
    <button id="rec" class="act">録音<small>基板のボタン 1 回</small></button>
    <button id="again" class="act">練習：もう一度<small>基板のボタン 2 回</small></button>
    <button id="stop" class="act">止める<small>動いている間に 1 回</small></button>
  </div>
  <div id="collectBox" hidden style="display:flex;flex-direction:column;gap:10px">
    <div class="row">
      <button id="take" class="act">録音<small>基板のボタン 1 回</small></button>
      <button id="del" class="sub">最後の録音を消す</button>
      <label>種類 <select id="kind"><option value="whistle">口笛</option><option value="quiet">無音・部屋の音</option><option value="other">話し声など</option></select></label>
      <label>1 本の長さ <select id="secs"><option>10</option><option>20</option><option selected>30</option></select> 秒まで</label>
    </div>
    <div><div class="nums mute" style="display:flex;justify-content:space-between"><span id="goalText"></span><span>目標 20 分</span></div><div class="goal"><i id="goalFill" style="width:0"></i></div></div>
  </div>
  <div class="row" id="playControls2">
    <button id="cal" class="sub">キャリブレーション</button>
    <label><input type="checkbox" id="quant"> 半音にそろえる</label>
  </div>
  <div class="mute" id="hint"></div>
</section>

<section id="takeBox" hidden>
  <h2>直前の録音</h2>
  <svg id="g4" viewBox="0 0 860 200"></svg><div class="mute num" id="info4">まだ録音がありません</div>
  <div class="row" style="margin-top:8px"><button class="sub" id="pt" disabled>▶ 聞く</button></div>
  <h2 style="margin-top:14px">最近の録音</h2><div class="takes" id="takes"></div>
  <p class="mute">録音は UNO Q の runs/real/whistle_data/ に保存されます（耳と記憶の学び直しに使います）。</p>
</section>

<section id="stepBox">
  <h2>使う版</h2>
  <div class="steps" id="steps"></div>
  <div class="mute" style="margin-top:8px">選んだ版は覚えておき、次に起動したときも使います。</div>
</section>

<section id="arcBox">
  <h2>アーカイブの声を吹く <span class="mute">口笛言語（シルボ・ゴメロ）とシジュウカラ</span></h2>
  <div class="mute" style="margin-bottom:8px">「原音」でアーカイブの声を、「笛の速さ」で笛が吹く速さに伸ばした声を、この画面から聞けます。「笛で吹く」で選んだ版が吹きます（その後、基板のボタン 2 回でもう一度）。</div>
  <div class="arc" id="arc"><span class="mute">読み込み中</span></div>
</section>

<section><h2>練習の成績 <span class="mute" id="practiceinfo"></span></h2>
<div class="practice" id="practice"><span class="mute">録音して演奏すると、1 回目から順に並びます（調を合わせた平均の誤差、セント。低いほど良い）</span></div></section>

<div class="grid2">
<section><h2 id="h1">聞いた口笛</h2>
<div class="legend"><span><i class="sw" style="background:var(--c1)"></i>耳の聞いた音程</span><span><i class="sw" style="background:var(--c2)"></i>整えた後</span></div>
<svg id="g1" viewBox="0 0 860 260"></svg><div class="mute num" id="info1"></div>
<div class="row" style="margin-top:8px"><button class="sub" id="pw" disabled>▶ 口笛を聞く</button></div></section>
<section><h2>笛の演奏</h2>
<div class="legend"><span><i class="sw" style="background:var(--c3)"></i>目標（笛の音域へ移した口笛）</span><span><i class="sw" style="background:var(--c4)"></i>笛が鳴らした音（新しい回ほど濃い）</span></div>
<svg id="g2" viewBox="0 0 860 260"></svg><div class="mute num" id="info2"></div>
<div class="chips" id="playChips"></div>
<div class="row" style="margin-top:8px"><button class="sub" id="pf" disabled>▶ 笛の演奏を聞く（最後の回）</button></div></section>
</div>

<section id="multiBox" hidden><h2>回ごとの演奏 <span class="mute">目標（黒）と、その回に鳴った音（調を合わせて）</span></h2>
<div class="multi" id="multi"></div></section>

<section><h2>キャリブレーション（その日の笛）</h2>
<div class="legend"><span><i class="sw" style="background:var(--c2)"></i>縮み切り → 伸び切り → 縮み切り（PWM 0.6）</span></div>
<svg id="g3" viewBox="0 0 860 200"></svg><div class="mute num" id="info3">まだ測っていません</div></section>

<p class="mute">縦軸はセント（100 セント = 半音、0 = ラ 440 Hz）。 <a href="/bench">手動テストのページ</a></p>
<audio id="au"></audio>
</main><script>
const $=id=>document.getElementById(id);
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
const PHASE={idle:"待機中",prepare:"準備中",tune:"キャリブレーション中",record:"録音中",analyse:"聞き取り中",play:"演奏中",done:"完了",saved:"保存しました",error:"エラー"};
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
let P={plays:[]},playsId=null,hiddenPlays=new Set();
function playColor(i,n){const k=n<=1?6:Math.round(1+i*5/(n-1));return css("--p"+Math.min(6,Math.max(1,k)))}
async function loadPlays(){try{P=await (await fetch("/api/whistle/plays")).json()}catch(e){return}hiddenPlays.clear();drawPlays()}
function drawPlays(){
  const pl=P.plays||[];if(!pl.length||!P.target){$("multiBox").hidden=true;return}
  const tgt=P.target.map((v,i)=>P.voice[i]?v:null),n=pl.length,mask=a=>a.map((v,i)=>P.voice[i]?v:null);
  plot($("g2"),[{y:tgt,c:css("--c3"),w:3},...pl.map((p,i)=>hiddenPlays.has(p.n)?null:{y:mask(p.heard),c:playColor(i,n),dots:true}).filter(Boolean)]);
  $("playChips").innerHTML=pl.map((p,i)=>`<button data-n="${p.n}" aria-pressed="${!hiddenPlays.has(p.n)}"><i style="background:${playColor(i,n)}"></i>${p.n} 回目 ${p.error==null?"-":p.error.toFixed(0)+" セント"}</button>`).join("");
  for(const b of $("playChips").children)b.onclick=()=>{const k=+b.dataset.n;hiddenPlays.has(k)?hiddenPlays.delete(k):hiddenPlays.add(k);drawPlays()};
  $("multiBox").hidden=false;
  $("multi").innerHTML=pl.map((p,i)=>`<div><span><b>${p.n} 回目</b>　誤差 ${p.error==null?"-":p.error.toFixed(0)+" セント"}</span><svg viewBox="0 0 860 200" id="m${i}"></svg></div>`).join("");
  pl.forEach((p,i)=>plot($("m"+i),[{y:tgt,c:css("--c3"),w:3},{y:mask(p.heard),c:playColor(i,n),dots:true}],200))}
let S={},shownTune=null,shownStamp=null,stepsKey="",lastPress=null,shownPractice="",shownTake="",polledAt=0;
const KIND={whistle:"口笛",quiet:"無音・部屋の音",other:"話し声など"};
function remainNow(){if(S.phase!=="record"||S.record_elapsed==null)return null;const el=S.record_elapsed+(Date.now()-polledAt)/1000;return Math.max(0,S.max_seconds-el)}
function drawRemain(){const r=remainNow();const box=$("remain");if(r===null){box.hidden=true;return}
  box.hidden=false;const f=r/S.max_seconds*100;$("remainFill").style.width=f+"%";$("remainKnob").style.left=`calc(14px + (100% - 28px) * ${f/100})`;
  $("remainS").textContent=Math.ceil(r);$("remainOf").textContent=`最長 ${S.max_seconds.toFixed(0)} 秒`}
setInterval(drawRemain,200);
function renderSteps(s){
  const key=JSON.stringify(s.steps)+s.step+s.busy;if(key===stepsKey)return;stepsKey=key;
  $("steps").innerHTML=s.steps.map(p=>{const t=[p.ear==="nn"?"耳 NN":"耳 YIN",p.valve_nn?"弁 NN":"弁 規則",...(p.memory?["記憶"]:[])];
    const title=(p.title||"").replace(/^[^:：]+[:：]\s*/,"");
    return `<button class="step${p.name===s.step?" on":""}" data-s="${p.name}" ${s.busy?"disabled":""}><b>${p.name}</b><span>${title}</span><span class="tags">${t.map(x=>`<em>${x}</em>`).join("")}</span></button>`}).join("");
  for(const b of document.querySelectorAll(".step"))b.onclick=()=>api("step",{step:b.dataset.s})}
async function poll(){
  let s;try{s=await (await fetch("/api/whistle/status")).json()}catch(e){$("phase").textContent="接続できません";setTimeout(poll,1500);return}
  S=s;polledAt=Date.now();const ph=s.phase||"idle";
  const col=!!s.collect;$("mPlay").setAttribute("aria-pressed",!col);$("mCollect").setAttribute("aria-pressed",col);
  for(const id of ["mPlay","mCollect"])$(id).disabled=s.busy;
  $("collectBox").hidden=!col;$("takeBox").hidden=!col;$("stepBox").hidden=col;$("flow").hidden=col;
  for(const id of ["rec","again","stop"])$(id).hidden=col;$("playControls2").hidden=col;
  for(const sec of document.querySelectorAll(".grid2"))sec.hidden=col;
  if(col){const t=$("take");t.firstChild.textContent=ph==="record"?"録音終了して保存":"録音";t.classList.toggle("record",ph==="record");
    t.disabled=s.busy&&ph!=="record";$("del").disabled=s.busy||!s.take_count;
    if(document.activeElement!==$("kind"))$("kind").value=s.collect_kind;if(document.activeElement!==$("secs"))$("secs").value=String(Math.round(s.collect_seconds));
    $("kind").disabled=$("secs").disabled=s.busy;
    const mins=s.take_seconds/60;$("goalText").textContent=`口笛 ${mins.toFixed(1)} 分（全部で ${s.take_count} 本）`;$("goalFill").style.width=Math.min(100,mins/20*100)+"%";
    $("takes").innerHTML=(s.takes||[]).slice().reverse().map(t=>`<div><b>${t.name}</b><span>${KIND[t.kind]||t.kind}</span><span>${t.seconds.toFixed(1)} 秒</span></div>`).join("")||'<span class="mute">まだありません</span>';
    const tk=s.take;const key=tk?tk.name:"";if(key!==shownTake){shownTake=key;
      if(tk){plot($("g4"),[{y:tk.track,c:css("--c2"),dots:true}],200);$("info4").textContent=`${tk.name}：${tk.seconds.toFixed(1)} 秒、音量 ${tk.level_db.toFixed(0)} dB、音程が取れた割合 ${(tk.voiced*100).toFixed(0)} %（横軸は 0.02 秒刻み）`}
      else{$("g4").innerHTML="";$("info4").textContent="まだ録音がありません"}}
    $("pt").disabled=s.busy||!(s.has_audio||[]).includes("take")}
  $("dot").className="dot "+ph;$("phase").textContent=PHASE[ph]||ph;$("stepnow").textContent=s.step?"・"+s.step:"";
  $("msg").textContent=s.message||"";$("msg").style.color=ph==="error"?css("--warn"):"";
  const stage=ph==="record"?"record":(s.practice||[]).length&&!s.busy?"again":"idle";
  for(const d of $("flow").children)d.classList.toggle("on",d.dataset.p===stage);
  drawRemain();
  const rec=$("rec"),serverBusy=s.busy&&["tune","play"].includes(ph);
  rec.firstChild.textContent=ph==="record"?"録音終了して演奏":"録音";rec.classList.toggle("record",ph==="record");
  rec.disabled=s.busy&&ph!=="record";
  $("again").disabled=s.busy||!s.has_song;
  $("again").firstChild.textContent=s.has_song&&s.song_step!==s.step?`同じ録音を ${s.step} で`:"練習：もう一度";
  $("stop").disabled=!s.busy||ph==="record"||serverBusy;
  $("hint").textContent=serverBusy?"いまは基板の EXEC ボタンで止められます（画面からは届きません）。":"基板の EXEC ボタンと、この画面のボタンは同じ働きです。版の選択だけはこの画面で行います。";
  if(s.press&&s.press.n!==lastPress){if(lastPress!==null){const b=$(s.collect?"take":{record:"rec",finish:"rec",replay:"again",stop:"stop"}[s.press.action]);
      if(b){b.classList.remove("flash");void b.offsetWidth;b.classList.add("flash");setTimeout(()=>b.classList.remove("flash"),700)}}lastPress=s.press.n}
  const pr=s.practice||[];const key2=JSON.stringify(pr);
  if(key2!==shownPractice){shownPractice=key2;
    if(pr.length){const mx=Math.max(...pr.filter(v=>v!==null),1);
      $("practice").innerHTML=pr.map((v,i)=>`<div><b>${v===null?"-":v.toFixed(0)}</b><i style="height:${v===null?2:Math.max(4,v/mx*60)}px"></i>${i+1} 回目</div>`).join("");
      $("practiceinfo").textContent=pr.length>1&&pr[0]!==null&&pr[pr.length-1]!==null?`1 回目 ${pr[0].toFixed(0)} → ${pr.length} 回目 ${pr[pr.length-1].toFixed(0)} セント`:""}
    else{$("practice").innerHTML='<span class="mute">録音して演奏すると、1 回目から順に並びます（調を合わせた平均の誤差、セント。低いほど良い）</span>';$("practiceinfo").textContent=""}}
  $("cal").disabled=s.busy;
  renderSteps(s);
  if(s.whistle){plot($("g1"),[{y:s.whistle,c:css("--c1"),dots:true},{y:s.cleaned,c:css("--c2")}]);
    const r=s.report||{};$("info1").textContent=r.whistle_span_cents!=null?`口笛の幅 ${r.whistle_span_cents.toFixed(0)} セント／笛の音域 ${r.range_cents.toFixed(0)} セント`+(r.clipped_fraction>0?`（はみ出し ${(r.clipped_fraction*100).toFixed(0)} %）`:""):r.memory?`記憶に ${r.listened_steps} ステップ書き込み、そこから思い出した楽譜`:""}
  else{$("g1").innerHTML="";$("info1").textContent=""}
  if(s.target){
    if(s.plays_id&&s.plays_id!==playsId&&!s.busy){playsId=s.plays_id;loadPlays()}
    if(s.busy||!P.plays.length){const tgt=s.target.map((v,i)=>s.voice[i]?v:null);
      plot($("g2"),[{y:tgt,c:css("--c3"),w:3},...(s.heard?[{y:s.heard.map((v,i)=>s.voice[i]?v:null),c:css("--c4"),dots:true}]:[])])}
    $("info2").textContent=s.relative_mae!=null?`最後の回：調のずれ ${s.key>=0?"+":""}${s.key.toFixed(0)} セント／それを除いた平均の誤差 ${s.relative_mae.toFixed(0)} セント`:""}
  else{$("g2").innerHTML="";$("info2").textContent="";$("playChips").innerHTML="";$("multiBox").hidden=true;P={plays:[]};playsId=null}
  const tu=s.tuning;if(!tu){shownTune=null;$("g3").innerHTML="";$("info3").textContent="まだ測っていません（最初の録音の前に一度だけ測ります）"}
  if(tu&&tu.time+":"+s.step!==shownTune){shownTune=tu.time+":"+s.step;plot($("g3"),[{y:tu.heard,c:css("--c2"),dots:true}],200);
    const hz=c=>(440*Math.pow(2,c/1200)).toFixed(0);
    $("info3").textContent=`${tu.time} に測定：縮み切り ${hz(tu.home_cents)} Hz、いちばん高い音 ${tu.sweep_max_cents!=null?hz(tu.sweep_max_cents):"-"} Hz、ツインとのずれ ${tu.shift_cents>=0?"+":""}${tu.shift_cents.toFixed(0)} セント`}
  const ha=s.has_audio||[];$("pw").disabled=s.busy||!ha.includes("whistle");$("pf").disabled=s.busy||!ha.includes("flute");
  setTimeout(poll,s.busy?250:1000)}
$("rec").onclick=()=>{if(S.phase==="record")api("finish");else api("start",{quantize:$("quant").checked})};
$("again").onclick=()=>api("replay");$("stop").onclick=()=>api("stop");
$("mPlay").onclick=()=>api("mode",{collect:false});$("mCollect").onclick=()=>api("mode",{collect:true});
$("take").onclick=()=>{if(S.phase==="record")api("finish");else api("take")};$("del").onclick=()=>api("delete_last");
$("kind").onchange=()=>api("mode",{kind:$("kind").value});$("secs").onchange=()=>api("mode",{seconds:+$("secs").value});
$("pt").onclick=()=>listen("take");
if(location.pathname==="/collect")api("mode",{collect:true});         // the page that opens in the record-only mode
$("cal").onclick=()=>api("calibrate");
function listen(name){const a=$("au");a.src=`/api/whistle/${name}.wav?t=${Date.now()}`;a.play()}
$("pw").onclick=()=>listen("whistle");$("pf").onclick=()=>listen("flute");
let ARC=[],arcKey="";
const esc=t=>String(t==null?"":t).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function loadArchive(){try{ARC=(await (await fetch("/api/whistle/archive")).json()).items||[]}catch(e){ARC=[]}arcKey="";renderArchive(S)}
function hear(id,name){const a=$("au");a.src=`/api/whistle/archive/${encodeURIComponent(id)}/${name}.wav`;a.play()}
function renderArchive(s){
  const key=JSON.stringify([ARC.map(m=>m.id),s.busy,s.archive,s.step]);if(key===arcKey)return;arcKey=key;
  if(!ARC.length){$("arc").innerHTML='<span class="mute">まだありません。PC で <code>python scripts/archive_fetch.py silbo</code>・<code>tit</code>、<code>python scripts/archive_pack.py</code> を実行し、<code>scripts/uno_q_push.sh</code> で送ってください。</span>';return}
  $("arc").innerHTML=ARC.map(m=>{const lic=(m.licence||"").split("/licenses/")[1]||"";
    return `<div class="${m.id===s.archive?"on":""}" data-id="${esc(m.id)}"><b>${esc(m.title)}</b>${m.says?`<q>${esc(m.says)}</q>`:""}
    <span class="mute num" style="font-size:12px">原音 ${(+m.seconds).toFixed(1)} 秒 → 笛 ${(+m.play_seconds).toFixed(1)} 秒（${m.slow} 倍ゆっくり、音程の間隔 ×${(+m.interval_scale).toFixed(2)}）</span>
    <div class="row"><button class="sub" data-a="source">▶ 原音</button>${m.has_slow?'<button class="sub" data-a="source_slow">▶ 笛の速さ</button>':""}<button class="go" data-a="play" ${s.busy?"disabled":""}>笛で吹く（${esc(s.step||"")}）</button></div>
    <small>${esc(m.credit)}${lic?` · CC ${esc(lic.replace(/\/$/,"").toUpperCase())}`:""}${m.page?` · <a href="${esc(m.page)}" target="_blank" rel="noopener">出典</a>`:""}</small></div>`}).join("");
  for(const b of $("arc").querySelectorAll("button")){const id=b.closest("[data-id]").dataset.id;
    b.onclick=()=>b.dataset.a==="play"?api("archive_play",{id}):hear(id,b.dataset.a)}}
setInterval(()=>{if(S&&S.phase!==undefined){renderArchive(S);$("arcBox").hidden=!!S.collect;
  $("h1").textContent=S.archive?"お手本（アーカイブの声）":"聞いた口笛";$("pw").textContent=S.archive?"▶ お手本を聞く":"▶ 口笛を聞く"}},500);
loadArchive();
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
            if self.path.split("?")[0] in ("/", "/index.html", "/whistle", "/collect"):
                self._send(200, MAIN_PAGE, "text/html; charset=utf-8")
            elif self.path == "/bench":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._run(bench.status)
            elif self.path == "/api/whistle/status":
                self._json(200, whistle.status())
            elif self.path == "/api/whistle/plays":          # every play of the song (fetched when it changes)
                self._json(200, {"plays": whistle.plays, "target": whistle.state.get("target"),
                                 "voice": whistle.state.get("voice")})
            elif self.path == "/api/whistle/archive":
                self._json(200, {"items": whistle.archive_list()})
            elif self.path.startswith("/api/whistle/archive/"):     # /api/whistle/archive/<id>/<source|source_slow>.wav
                parts = self.path.split("?")[0].split("/")
                data = (whistle.archive_wav(parts[4], parts[5][:-4])
                        if len(parts) == 6 and parts[5].endswith(".wav") else None)
                if data is None:
                    self._json(404, {"error": "ありません"})
                else:
                    self._send(200, data, "audio/wav")
            elif self.path.split("?")[0] in ("/api/whistle/whistle.wav", "/api/whistle/flute.wav", "/api/whistle/take.wav"):
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
                "/api/whistle/replay": lambda: whistle.replay(),
                "/api/whistle/mode": lambda: whistle.set_mode(req.get("collect"), req.get("kind"), req.get("seconds")),
                "/api/whistle/take": whistle.start_take,
                "/api/whistle/delete_last": whistle.delete_last,
                "/api/whistle/stop": whistle.stop,
                "/api/whistle/step": lambda: whistle.choose(req["step"]),
                "/api/whistle/calibrate": lambda: whistle.calibrate(req.get("step")),
                "/api/whistle/archive_play": lambda: whistle.archive_play(str(req["id"]), req.get("step")),
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
    ap.add_argument("--no-panel", action="store_true", help="do not drive the touch panel (Tab5 on I2C)")
    args = ap.parse_args()
    bench = Bench(args.server_port)
    whistle = WhistleJob(bench, args.server_port, load_profiles())
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(bench, whistle))
    if not args.no_calibrate_at_start:
        whistle.calibrate_at_start()
    if not args.no_button:
        whistle.watch_button()
    if not args.no_panel:
        PanelBridge(whistle, args.server_port).run()
    print(f"yamabiko web on port {args.port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        with bench.lock:
            bench._drop()


if __name__ == "__main__":
    main()
