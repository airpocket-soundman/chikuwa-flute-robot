#!/usr/bin/env bash
# Copy what the UNO Q runs to /home/arduino/yamabiko over adb (Git Bash on the PC):
#   scripts/uno_q_push.sh [policy.npz ...]
# flute_rl/ (without the torch modules), the scripts of the real rig, the firmware source and the given
# policies (default runs/yamabiko_gru.npz). E2E deployment uses a NumPy .npz exported on the PC.
# Then on the UNO Q: cd /home/arduino/yamabiko && python3 scripts/...
set -e
cd "$(dirname "$0")/.."
ADB="${ADB:-/c/Users/yamas/platform-tools/adb.exe}"
D=/home/arduino/yamabiko
export MSYS_NO_PATHCONV=1
"$ADB" shell "mkdir -p $D/flute_rl/yamabiko $D/scripts $D/runs/real $D/yamabiko_fw"
for f in flute_rl/*.py flute_rl/yamabiko/*.py; do
  case "$f" in *torch*|*pitchnet*) continue ;; esac
  "$ADB" push "$f" "$D/$f" >/dev/null
done
for f in scripts/yamabiko_collect.py scripts/yamabiko_play.py scripts/yamabiko_e2e_play.py \
         scripts/bench_yamabiko_e2e.py scripts/yamabiko_link.py; do
  "$ADB" push "$f" "$D/$f" >/dev/null
done
"$ADB" push mcu/yamabiko_fw/yamabiko_fw.ino "$D/yamabiko_fw/" >/dev/null
for p in "${@:-runs/yamabiko_gru.npz}"; do
  "$ADB" push "$p" "$D/$p" >/dev/null
done
echo "pushed to $D"
