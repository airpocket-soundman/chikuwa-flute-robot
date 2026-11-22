#!/usr/bin/env bash
# Copy what the UNO Q runs to /home/arduino/yamabiko over adb, or over Wi-Fi with UNOQ=ssh (Git Bash on the PC):
#   scripts/uno_q_push.sh [policy.npz ...]
#   UNOQ=ssh scripts/uno_q_push.sh             # ssh host "unoq", set up by scripts/uno_q_setup_wifi.sh
# flute_rl/ (without the torch modules), the scripts of the real rig, the firmware source and the given
# files (policies, runtimes, targets). E2E deployment uses a NumPy .npz exported on the PC.
# Then on the UNO Q: cd /home/arduino/yamabiko && python3 scripts/...
set -e
cd "$(dirname "$0")/.."
ADB="${ADB:-/c/Users/yamas/platform-tools/adb.exe}"
[ -x "$ADB" ] || ADB=adb                      # fall back to adb on the PATH
D=/home/arduino/yamabiko
export MSYS_NO_PATHCONV=1
if [ "${UNOQ:-adb}" = ssh ]; then
  sh_() { ssh unoq "$1"; }
  put() { scp -q "$1" "unoq:$2"; }
else
  sh_() { "$ADB" shell "$1"; }
  put() { "$ADB" push "$1" "$2" >/dev/null; }
fi
sh_ "mkdir -p $D/flute_rl/yamabiko $D/scripts $D/runs/real $D/yamabiko_fw"
for f in flute_rl/*.py flute_rl/yamabiko/*.py; do
  case "$f" in *torch*|*pitchnet*) continue ;; esac
  put "$f" "$D/$f"
done
for f in scripts/yamabiko_collect.py scripts/yamabiko_play.py scripts/yamabiko_e2e_play.py \
         scripts/bench_yamabiko_e2e.py scripts/yamabiko_link.py scripts/yamabiko_bench_watch.py \
         scripts/yamabiko_mic_check.py scripts/yamabiko_twin_calibrate.py scripts/yamabiko_twin_play.py \
         scripts/yamabiko_server.py scripts/yamabiko_web.py scripts/yamabiko_button_jog.py scripts/yamabiko_pitch_sweep.py; do
  put "$f" "$D/$f"
done
put mcu/yamabiko_fw/yamabiko_fw.ino "$D/yamabiko_fw/yamabiko_fw.ino"
for p in "$@"; do
  put "$p" "$D/$p"
done
echo "pushed to $D"
