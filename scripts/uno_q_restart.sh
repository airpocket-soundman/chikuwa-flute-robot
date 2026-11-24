#!/bin/bash
# Restart the command server and the web page on the UNO Q (as at boot).  Extra arguments go to the page,
# e.g.  bash restart.sh --no-calibrate-at-start --no-button   (silent: no calibration, the button does nothing)
cd /home/arduino/yamabiko
for p in $(pgrep -f "scripts/yamabiko_(server|web).py"); do [ "$p" != "$$" ] && kill $p; done
sleep 2
setsid nohup python3 scripts/yamabiko_server.py --wait 120 --host 0.0.0.0 >> /home/arduino/yamabiko_server.log 2>&1 < /dev/null &
sleep 6
setsid nohup python3 scripts/yamabiko_web.py "$@" >> /home/arduino/yamabiko_web.log 2>&1 < /dev/null &
sleep 4
pgrep -af "scripts/yamabiko_(server|web).py"
