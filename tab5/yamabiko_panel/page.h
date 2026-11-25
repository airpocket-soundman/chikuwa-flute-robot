// The page the UNO Q sends (flute_rl/yamabiko/panel.py: encode / decode; keep the two the same).
#pragma once
#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>

namespace page {

constexpr int16_t NO_VALUE = -32768;
constexpr uint8_t FLAG_BUSY = 1, FLAG_HAS_SONG = 2, FLAG_COLLECT = 4, FLAG_SERVER_BUSY = 8, FLAG_ARCHIVE = 16;
constexpr uint8_t EV_STEP = 0x10, EV_CALIBRATE = 0x11, EV_ARCHIVE = 0x12;
constexpr uint8_t EV_RECORD = 0x13, EV_FINISH = 0x14, EV_AGAIN = 0x15;   // the panel's record / end / again
static const char *PHASES[] = {"idle", "prepare", "tune", "record", "analyse", "play", "done", "saved", "error"};
static const char *PHASE_JA[] = {"待機中", "準備中", "キャリブレーション中", "録音中", "聞き取り中",
                                 "演奏中", "完了", "保存しました", "エラー"};
enum Phase { IDLE, PREPARE, TUNE, RECORD, ANALYSE, PLAY, DONE, SAVED, ERROR_ };

struct Page {
  uint8_t format = 0, phase = 0, flags = 0, step = 0;
  int8_t archive_playing = -1;
  float record_s = 30, play_s = 0;
  int16_t home_cents = NO_VALUE, shift_cents = NO_VALUE;
  std::vector<float> practice;          // < 0: none
  std::string message;
  std::vector<std::string> steps, archive;
  std::vector<int16_t> target, heard;
  uint8_t stride = 1;
  bool busy() const { return flags & FLAG_BUSY; }
};

struct Reader {
  const uint8_t *p;
  size_t n, k = 0;
  bool ok = true;
  Reader(const uint8_t *p_, size_t n_) : p(p_), n(n_) {}
  bool need(size_t m) { if (k + m > n) ok = false; return ok; }
  uint8_t u8() { return need(1) ? p[k++] : 0; }
  uint16_t u16() { if (!need(2)) return 0; uint16_t v = p[k] | (p[k + 1] << 8); k += 2; return v; }
  int16_t i16() { return (int16_t)u16(); }
  std::string str() {
    uint8_t m = u8();
    if (!need(m)) return "";
    std::string s((const char *)p + k, m);
    k += m;
    return s;
  }
};

inline bool parse(const uint8_t *data, size_t n, Page &pg) {
  Reader r(data, n);
  Page q;
  q.format = r.u8();
  if (q.format != 1) return false;
  q.phase = r.u8();
  if (q.phase > 8) q.phase = 0;
  q.flags = r.u8();
  q.step = r.u8();
  uint8_t n_steps = r.u8();
  q.archive_playing = (int8_t)r.u8();
  uint8_t n_arc = r.u8();
  q.record_s = r.u16() / 10.0f;
  q.play_s = r.u16() / 10.0f;
  q.home_cents = r.i16();
  q.shift_cents = r.i16();
  uint8_t n_pr = r.u8();
  for (int i = 0; i < n_pr; i++) {
    uint16_t v = r.u16();
    q.practice.push_back(v == 0xFFFF ? -1.0f : v / 10.0f);
  }
  q.message = r.str();
  for (int i = 0; i < n_steps; i++) q.steps.push_back(r.str());
  for (int i = 0; i < n_arc; i++) q.archive.push_back(r.str());
  uint16_t pts = r.u16();
  for (int i = 0; i < pts; i++) q.target.push_back(r.i16());
  for (int i = 0; i < pts; i++) q.heard.push_back(r.i16());
  q.stride = r.u8();
  if (!r.ok) return false;
  pg = q;
  return true;
}

}  // namespace page
