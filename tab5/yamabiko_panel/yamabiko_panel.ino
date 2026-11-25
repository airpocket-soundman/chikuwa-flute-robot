// M5Stack Tab5: the touch panel of Yamabiko No.1, on I2C to the UNO Q's MCU (B board J3).
//
// Wiring (the Grove cable of PORT.A):  black GND -> J3-2 GND,  yellow G53 SDA -> J3-3 SDA,
//                                      white G54 SCL -> J3-4 SCL,  red 5 V: NOT connected (J3-1 is 3.3 V)
// The B board pulls SDA / SCL up to 3.3 V (4.7 k).  The Tab5 runs on its own power.
//
// The MCU (mcu/yamabiko_fw/yamabiko_fw.ino, "Panel") is the target at 0x42:
//   write 0x03 lo hi      where the next read starts;   read: 'Y' '1' version(u16) length(u16), the page
//   write 0x01 n          n presses of the EXEC button (1: record / end / stop, 2: play again)
//   write 0x02 bytes      an event for the UNO Q's web page process (choose a step, calibrate, archive clip)
// The page is made by flute_rl/yamabiko/panel.py (read by page.h).  The buttons do what the board button
// and the web page (http://<UNO Q>:8080/) do; the web page and this panel show the same state.
//
// Build: Arduino IDE / arduino-cli, board "M5Tab5" (m5stack:esp32:m5stack_tab5), libraries M5Unified, M5GFX.
#include <M5Unified.h>
#include "page.h"

constexpr uint8_t ADDR = 0x42;
constexpr uint32_t FREQ = 100000;      // 100 kHz: a long cable and the MCU's clock stretching are fine
constexpr uint32_t POLL_MS = 200;
constexpr size_t PIECE = 64;

static M5Canvas cv(&M5.Display);
static page::Page pg;
static uint16_t shown_ver = 0xFFFF;
static bool have_page = false;
static uint32_t last_ok_ms = 0, last_poll_ms = 0, phase_since_ms = 0, flash_until_ms = 0;
static uint8_t last_phase = 255;
static int flash_btn = -1;
static bool tab_archive = false;
static bool dirty = true;
static uint8_t buf[3200];

// ---- colours and layout ------------------------------------------------------------------------
static uint16_t C(uint32_t rgb) { return lgfx::color565(rgb >> 16, (rgb >> 8) & 0xFF, rgb & 0xFF); }
static uint16_t BG, CARD, FG, MUTE, LINE, ACC, ACC2, REC, WARN, C_ORANGE, C_ORANGE2, C_WHITE;
static int W, H;
struct Rect { int x, y, w, h; bool in(int px, int py) const { return px >= x && px < x + w && py >= y && py < y + h; } };
enum Btn { B_REC, B_AGAIN, B_CAL, B_STOP, B_N };
static Rect btn[B_N], tabs[2], listArea, graph, bars;

static void layout() {
  W = cv.width(); H = cv.height();
  int top = 96, left = 300, right = 350, gap = 12;
  for (int i = 0; i < B_N; i++) btn[i] = {gap, top + gap + i * 124, left - 2 * gap, 112};
  int rx = W - right;
  tabs[0] = {rx, top + gap, right / 2 - gap, 54};
  tabs[1] = {rx + right / 2 - gap / 2, top + gap, right / 2 - gap, 54};
  listArea = {rx, top + gap + 62, right - gap, H - top - gap - 62 - gap};
  graph = {left + gap, top + gap, rx - left - 3 * gap, (H - top) * 62 / 100};
  bars = {left + gap, graph.y + graph.h + gap, graph.w, H - (graph.y + graph.h + gap) - gap};
}

// ---- I2C ------------------------------------------------------------------------------------------
static bool set_ptr(uint16_t at) {
  uint8_t b[2] = {(uint8_t)(at & 0xFF), (uint8_t)(at >> 8)};
  return M5.Ex_I2C.writeRegister(ADDR, 0x03, b, 2, FREQ);
}
static bool read_at(uint16_t at, uint8_t *out, size_t n) {
  if (!set_ptr(at)) return false;
  if (!M5.Ex_I2C.start(ADDR, true, FREQ)) { M5.Ex_I2C.stop(); return false; }
  bool ok = M5.Ex_I2C.read(out, n, true);
  M5.Ex_I2C.stop();
  return ok;
}
static bool header(uint16_t &ver, uint16_t &len) {
  uint8_t h[6];
  if (!read_at(0, h, 6) || h[0] != 'Y' || h[1] != '1') return false;
  ver = h[2] | (h[3] << 8);
  len = h[4] | (h[5] << 8);
  return len <= sizeof(buf);
}
static void poll_page() {
  uint16_t ver, len;
  if (!header(ver, len)) return;
  last_ok_ms = millis();
  if (ver == shown_ver && have_page) return;
  for (size_t off = 0; off < len; off += PIECE) {
    size_t n = len - off < PIECE ? len - off : PIECE;
    if (!read_at(6 + off, buf + off, n)) return;
  }
  uint16_t ver2, len2;
  if (!header(ver2, len2) || ver2 != ver) return;   // changed while read: next time
  page::Page q;
  if (!page::parse(buf, len, q)) return;
  pg = q;
  shown_ver = ver;
  have_page = true;
  if (pg.phase != last_phase) { last_phase = pg.phase; phase_since_ms = millis(); }
  dirty = true;
}
static void press(uint8_t n) { M5.Ex_I2C.writeRegister(ADDR, 0x01, &n, 1, FREQ); }
static void event(uint8_t a, int b = -1) {
  uint8_t e[2] = {a, (uint8_t)b};
  M5.Ex_I2C.writeRegister(ADDR, 0x02, e, b < 0 ? 1 : 2, FREQ);
}

// ---- drawing ----------------------------------------------------------------------------------------
static void text_fit(const std::string &s, int x, int y, int w, const lgfx::IFont *font) {
  cv.setFont(font);
  if (cv.textWidth(s.c_str()) <= w) { cv.drawString(s.c_str(), x, y); return; }
  std::string t = s;
  while (!t.empty() && cv.textWidth((t + "…").c_str()) > w) {
    size_t k = t.size() - 1;
    while (k > 0 && (t[k] & 0xC0) == 0x80) k--;      // drop one UTF-8 character
    t.resize(k);
  }
  cv.drawString((t + "…").c_str(), x, y);
}

static bool btn_enabled(int i) {
  bool busy = pg.busy(), rec = pg.phase == page::RECORD;
  if (!have_page) return false;
  switch (i) {
    case B_REC: return !busy || rec;
    case B_AGAIN: return !busy && (pg.flags & page::FLAG_HAS_SONG);
    case B_CAL: return !busy;
    case B_STOP: return busy && !rec;
  }
  return false;
}

static void draw_button(int i) {
  const Rect &r = btn[i];
  bool en = btn_enabled(i), fl = (flash_btn == i && millis() < flash_until_ms);
  uint16_t col = i == B_REC ? (pg.phase == page::RECORD ? REC : ACC) : i == B_AGAIN ? ACC : i == B_CAL ? CARD : WARN;
  if (!en) col = LINE;
  cv.fillRoundRect(r.x, r.y, r.w, r.h, 16, col);
  if (fl) cv.drawRoundRect(r.x - 4, r.y - 4, r.w + 8, r.h + 8, 18, ACC);
  uint16_t tc = (i == B_CAL && en) ? FG : (en ? C_WHITE : MUTE);
  if (i == B_CAL) cv.drawRoundRect(r.x, r.y, r.w, r.h, 16, LINE);
  cv.setTextColor(tc);
  const char *label[] = {pg.phase == page::RECORD ? "録音終了して演奏" : "録音", "練習：もう一度", "キャリブレーション", "止める"};
  const char *sub[] = {"基板のボタン 1 回", "基板のボタン 2 回", "その日の笛を測る（約 10 秒鳴る）", "動いている間に"};
  cv.setTextDatum(textdatum_t::top_left);
  text_fit(label[i], r.x + 18, r.y + 20, r.w - 30, &fonts::lgfxJapanGothic_32);
  cv.setFont(&fonts::lgfxJapanGothic_16);
  text_fit(sub[i], r.x + 18, r.y + 70, r.w - 30, &fonts::lgfxJapanGothic_16);
}

static void draw_header() {
  cv.fillRect(0, 0, W, 96, CARD);
  cv.drawFastHLine(0, 96, W, LINE);
  uint16_t dot = !have_page ? MUTE : pg.phase == page::RECORD ? REC : pg.phase == page::ERROR_ ? WARN
                 : pg.busy() ? C_ORANGE : ACC;
  cv.fillCircle(30, 32, 11, dot);
  cv.setTextColor(FG);
  cv.setTextDatum(textdatum_t::top_left);
  cv.setFont(&fonts::lgfxJapanGothic_32);
  std::string head = have_page ? page::PHASE_JA[pg.phase] : "接続を待っています";
  if (have_page && pg.step < pg.steps.size()) head += "・" + pg.steps[pg.step];
  cv.drawString(head.c_str(), 54, 14);
  cv.setTextColor(pg.phase == page::ERROR_ ? WARN : MUTE);
  std::string msg = have_page ? pg.message : "B 基板 J3 と Tab5 の PORT.A をつないでください（赤の 5 V はつながない）";
  text_fit(msg, 54, 60, W - 80, &fonts::lgfxJapanGothic_20);
  if (have_page && pg.home_cents != page::NO_VALUE) {
    char s[64];
    snprintf(s, sizeof(s), "縮み切り %.0f Hz", 440.0 * pow(2.0, pg.home_cents / 1200.0));
    cv.setFont(&fonts::lgfxJapanGothic_20);
    cv.setTextColor(MUTE);
    cv.setTextDatum(textdatum_t::top_right);
    cv.drawString(s, W - 20, 20);
    cv.setTextDatum(textdatum_t::top_left);
  }
}

static void draw_progress() {
  const Rect &last = btn[B_N - 1];
  int y = last.y + last.h + 18, x = last.x, w = last.w;
  float total = pg.phase == page::RECORD ? pg.record_s : pg.phase == page::PLAY ? pg.play_s + 2.0f : 0;
  if (!have_page || total <= 0) return;
  float el = (millis() - phase_since_ms) / 1000.0f;
  float f = el / total; if (f > 1) f = 1;
  cv.fillRoundRect(x, y, w, 26, 13, ACC2);
  cv.fillRoundRect(x, y, (int)(w * f), 26, 13, pg.phase == page::RECORD ? REC : C_ORANGE);
  char s[48];
  snprintf(s, sizeof(s), pg.phase == page::RECORD ? "残り %.0f 秒" : "%.0f / %.0f 秒",
           pg.phase == page::RECORD ? (total - el > 0 ? total - el : 0) : el, total);
  cv.setTextColor(MUTE);
  cv.setFont(&fonts::lgfxJapanGothic_20);
  cv.drawString(s, x, y + 34);
}

static void draw_graph() {
  const Rect &g = graph;
  cv.fillRoundRect(g.x, g.y, g.w, g.h, 14, CARD);
  cv.setTextColor(FG);
  cv.setFont(&fonts::lgfxJapanGothic_24);
  cv.drawString((pg.flags & page::FLAG_ARCHIVE) ? "お手本（アーカイブ）と笛の演奏" : "楽譜と笛の演奏", g.x + 16, g.y + 10);
  cv.setFont(&fonts::lgfxJapanGothic_16);
  cv.setTextColor(MUTE);
  cv.drawString("黒：楽譜（笛の音域へ移したもの）  橙：最後の回に鳴った音", g.x + 16, g.y + 42);
  int px = g.x + 60, py = g.y + 70, pw = g.w - 80, ph = g.h - 100;
  int lo = 32767, hi = -32768;
  for (auto v : pg.target) if (v != page::NO_VALUE) { lo = std::min<int>(lo, v); hi = std::max<int>(hi, v); }
  for (auto v : pg.heard) if (v != page::NO_VALUE) { lo = std::min<int>(lo, v); hi = std::max<int>(hi, v); }
  if (lo > hi) {
    cv.setFont(&fonts::lgfxJapanGothic_20);
    cv.drawString("まだ楽譜がありません。録音するか、お手本を選んでください", px, py + ph / 2 - 10);
    return;
  }
  int pad = std::max(50, (hi - lo) / 12); lo -= pad; hi += pad;
  int step = (hi - lo) > 1600 ? 400 : (hi - lo) > 700 ? 200 : 100;
  auto Y = [&](int v) { return py + ph - (int)((long)(v - lo) * ph / (hi - lo)); };
  size_t n = pg.target.size();
  auto X = [&](size_t i) { return px + (int)(n > 1 ? (long)i * pw / (n - 1) : 0); };
  cv.setFont(&fonts::lgfxJapanGothic_12);
  for (int v = (lo / step + 1) * step; v <= hi; v += step) {
    cv.drawFastHLine(px, Y(v), pw, LINE);
    char s[12]; snprintf(s, sizeof(s), "%d", v);
    cv.setTextDatum(textdatum_t::middle_right);
    cv.drawString(s, px - 6, Y(v));
  }
  cv.setTextDatum(textdatum_t::top_left);
  for (size_t i = 1; i < n; i++) {
    if (pg.target[i] == page::NO_VALUE || pg.target[i - 1] == page::NO_VALUE) continue;
    cv.drawWideLine(X(i - 1), Y(pg.target[i - 1]), X(i), Y(pg.target[i]), 2.5f, FG);
  }
  for (size_t i = 0; i < pg.heard.size() && i < n; i++)
    if (pg.heard[i] != page::NO_VALUE) cv.fillCircle(X(i), Y(pg.heard[i]), 3, C_ORANGE);
  char s[40];
  snprintf(s, sizeof(s), "%.0f 秒", n * pg.stride / 100.0f);
  cv.setFont(&fonts::lgfxJapanGothic_12);
  cv.setTextColor(MUTE);
  cv.setTextDatum(textdatum_t::top_right);
  cv.drawString(s, px + pw, py + ph + 6);
  cv.setTextDatum(textdatum_t::top_left);
}

static void draw_bars() {
  const Rect &b = bars;
  cv.fillRoundRect(b.x, b.y, b.w, b.h, 14, CARD);
  cv.setTextColor(FG);
  cv.setFont(&fonts::lgfxJapanGothic_24);
  cv.drawString("練習の成績", b.x + 16, b.y + 10);
  cv.setFont(&fonts::lgfxJapanGothic_16);
  cv.setTextColor(MUTE);
  cv.drawString("調を合わせた平均の誤差（セント、低いほど良い）", b.x + 160, b.y + 16);
  if (pg.practice.empty()) {
    cv.drawString("演奏すると、1 回目から順に並びます", b.x + 16, b.y + 60);
    return;
  }
  float mx = 1;
  for (float v : pg.practice) mx = std::max(mx, v);
  int n = pg.practice.size(), bw = std::min(70, (b.w - 40) / std::max(n, 1) - 10);
  int base = b.y + b.h - 34, top = b.y + 70;
  for (int i = 0; i < n; i++) {
    int x = b.x + 24 + i * (bw + 10);
    float v = pg.practice[i];
    int h = v < 0 ? 2 : std::max(4, (int)((base - top - 20) * v / mx));
    cv.fillRoundRect(x, base - h, bw, h, 4, i == n - 1 ? C_ORANGE : C_ORANGE2);
    char s[16];
    snprintf(s, sizeof(s), v < 0 ? "-" : "%.0f", v);
    cv.setTextColor(FG);
    cv.setTextDatum(textdatum_t::bottom_center);
    cv.setFont(&fonts::lgfxJapanGothic_16);
    cv.drawString(s, x + bw / 2, base - h - 2);
    snprintf(s, sizeof(s), "%d 回目", i + 1);
    cv.setTextColor(MUTE);
    cv.setTextDatum(textdatum_t::top_center);
    cv.setFont(&fonts::lgfxJapanGothic_12);
    cv.drawString(s, x + bw / 2, base + 6);
  }
  cv.setTextDatum(textdatum_t::top_left);
}

static int row_h() { return tab_archive ? 74 : 42; }

static void draw_list() {
  for (int t = 0; t < 2; t++) {
    const Rect &r = tabs[t];
    bool on = tab_archive == (t == 1);
    cv.fillRoundRect(r.x, r.y, r.w, r.h, 12, on ? CARD : BG);
    cv.drawRoundRect(r.x, r.y, r.w, r.h, 12, on ? ACC : LINE);
    cv.setTextColor(on ? FG : MUTE);
    cv.setTextDatum(textdatum_t::middle_center);
    cv.setFont(&fonts::lgfxJapanGothic_24);
    cv.drawString(t ? "お手本" : "使う版", r.x + r.w / 2, r.y + r.h / 2);
  }
  cv.setTextDatum(textdatum_t::top_left);
  const Rect &a = listArea;
  cv.fillRoundRect(a.x, a.y, a.w, a.h, 14, CARD);
  const auto &items = tab_archive ? pg.archive : pg.steps;
  int rh = row_h();
  for (size_t i = 0; i < items.size(); i++) {
    int y = a.y + 6 + i * rh;
    if (y + rh > a.y + a.h) break;
    bool on = tab_archive ? (int)i == pg.archive_playing : i == pg.step;
    if (on) cv.fillRoundRect(a.x + 4, y, a.w - 8, rh - 4, 10, ACC2);
    cv.setTextColor(pg.busy() ? MUTE : FG);
    if (tab_archive) {
      std::string t = items[i];
      size_t cut = t.find("「");                   // "シルボ・ゴメロ「…」": the language, then what it says
      std::string a1 = cut == std::string::npos ? t : t.substr(0, cut);
      std::string a2 = cut == std::string::npos ? "" : t.substr(cut);
      size_t colon = a1.find("：");
      if (cut == std::string::npos && colon != std::string::npos) { a2 = a1.substr(colon + 3); a1 = a1.substr(0, colon); }
      text_fit(a1, a.x + 14, y + 6, a.w - 28, &fonts::lgfxJapanGothic_16);
      text_fit(a2, a.x + 14, y + 30, a.w - 28, &fonts::lgfxJapanGothic_24);
    } else {
      text_fit(items[i], a.x + 14, y + 8, a.w - 28, &fonts::lgfxJapanGothic_24);
    }
  }
  if (items.empty()) {
    cv.setTextColor(MUTE);
    text_fit(tab_archive ? "お手本がありません（PC で archive_pack.py）" : "版がありません", a.x + 14, a.y + 14, a.w - 28,
             &fonts::lgfxJapanGothic_16);
  }
}

static void draw() {
  cv.fillScreen(BG);
  draw_header();
  for (int i = 0; i < B_N; i++) draw_button(i);
  draw_progress();
  draw_graph();
  draw_bars();
  draw_list();
  if (have_page && millis() - last_ok_ms > 3000) {
    cv.fillRoundRect(W / 2 - 330, H / 2 - 50, 660, 100, 16, WARN);
    cv.setTextColor(C_WHITE);
    cv.setTextDatum(textdatum_t::middle_center);
    cv.setFont(&fonts::lgfxJapanGothic_28);
    cv.drawString("UNO Q と通信できません", W / 2, H / 2);
    cv.setTextDatum(textdatum_t::top_left);
  }
  cv.pushSprite(0, 0);
  dirty = false;
}

// ---- touch ----------------------------------------------------------------------------------------
static void tapped(int x, int y) {
  for (int i = 0; i < B_N; i++) {
    if (!btn[i].in(x, y) || !btn_enabled(i)) continue;
    if (i == B_REC || i == B_STOP) press(1);
    else if (i == B_AGAIN) press(2);
    else event(page::EV_CALIBRATE);
    flash_btn = i;
    flash_until_ms = millis() + 600;
    dirty = true;
    return;
  }
  for (int t = 0; t < 2; t++)
    if (tabs[t].in(x, y)) { tab_archive = t == 1; dirty = true; return; }
  if (listArea.in(x, y) && have_page && !pg.busy()) {
    int i = (y - listArea.y - 6) / row_h();
    if (tab_archive && i >= 0 && i < (int)pg.archive.size()) event(page::EV_ARCHIVE, i);
    if (!tab_archive && i >= 0 && i < (int)pg.steps.size()) event(page::EV_STEP, i);
    dirty = true;
  }
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  M5.Display.setRotation(M5.Display.width() > M5.Display.height() ? 0 : 1);
  M5.Display.setBrightness(160);
  BG = C(0xF3F5F4); CARD = C(0xFFFFFF); FG = C(0x17201D); MUTE = C(0x5F6B66); LINE = C(0xDDE3E0);
  ACC = C(0x1F7A6C); ACC2 = C(0xE7F2EF); REC = C(0xC2410C); WARN = C(0xB42318); C_ORANGE = C(0xD9622B);
  C_ORANGE2 = C(0xF2B38A); C_WHITE = C(0xFFFFFF);
  cv.setColorDepth(16);
  cv.setPsram(true);
  cv.createSprite(M5.Display.width(), M5.Display.height());
  layout();
  M5.Ex_I2C.begin();
  draw();
}

void loop() {
  M5.update();
  auto t = M5.Touch.getDetail();
  if (t.wasClicked()) tapped(t.x, t.y);
  uint32_t now = millis();
  if (now - last_poll_ms >= POLL_MS) {
    last_poll_ms = now;
    poll_page();
    if (have_page && (pg.phase == page::RECORD || pg.phase == page::PLAY)) dirty = true;   // the progress bar
    if (have_page && now - last_ok_ms > 3000) dirty = true;
    if (flash_btn >= 0 && now > flash_until_ms) { flash_btn = -1; dirty = true; }
  }
  if (dirty) draw();
  delay(5);
}
