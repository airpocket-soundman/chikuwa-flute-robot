#!/usr/bin/env bash
# One-time setup over adb (Git Bash on the PC) so the UNO Q no longer needs the USB cable:
#   scripts/uno_q_setup_wifi.sh [192.168.50.160]
# - an SSH key for the PC (~/.ssh/unoq_ed25519) authorised on the UNO Q, and "Host unoq" in ~/.ssh/config
#   (then: ssh unoq, UNOQ=ssh scripts/uno_q_push.sh)
# - crontab @reboot: the command server listening on Wi-Fi too (--host 0.0.0.0, no authentication: trusted
#   LAN only) and the bench web page (port 8080)
set -e
cd "$(dirname "$0")/.."
IP="${1:-192.168.50.160}"
ADB="${ADB:-/c/Users/yamas/platform-tools/adb.exe}"
[ -x "$ADB" ] || ADB=adb
export MSYS_NO_PATHCONV=1
KEY="$HOME/.ssh/unoq_ed25519"
[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C "pc-to-unoq" -f "$KEY"
if ! grep -q "^Host unoq$" "$HOME/.ssh/config" 2>/dev/null; then
  printf '\nHost unoq\n  HostName %s\n  User arduino\n  IdentityFile ~/.ssh/unoq_ed25519\n  IdentitiesOnly yes\n' "$IP" >> "$HOME/.ssh/config"
fi
"$ADB" push "$(cygpath -w "$KEY.pub" 2>/dev/null || echo "$KEY.pub")" /tmp/pc_unoq.pub >/dev/null
"$ADB" shell 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys &&
  (grep -qxF "$(cat /tmp/pc_unoq.pub)" ~/.ssh/authorized_keys || cat /tmp/pc_unoq.pub >> ~/.ssh/authorized_keys) &&
  chmod 600 ~/.ssh/authorized_keys && rm /tmp/pc_unoq.pub'
"$ADB" shell 'crontab -l > /tmp/cron.old 2>/dev/null || true
  grep -v "yamabiko_server.py\|yamabiko_web.py" /tmp/cron.old > /tmp/cron.new || true
  echo "@reboot sleep 15 && cd /home/arduino/yamabiko && python3 scripts/yamabiko_server.py --wait 120 --host 0.0.0.0 >> /home/arduino/yamabiko_server.log 2>&1" >> /tmp/cron.new
  echo "@reboot sleep 45 && cd /home/arduino/yamabiko && python3 scripts/yamabiko_web.py >> /home/arduino/yamabiko_web.log 2>&1" >> /tmp/cron.new
  crontab /tmp/cron.new && crontab -l'
echo "set up: try  ssh unoq echo ok"
