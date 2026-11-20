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


def make_handler(bench: Bench):
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
    args = ap.parse_args()
    bench = Bench(args.server_port)
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(bench))
    print(f"yamabiko web on port {args.port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        with bench.lock:
            bench._drop()


if __name__ == "__main__":
    main()
