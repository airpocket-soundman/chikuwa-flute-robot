#include "hil_protocol.h"

#include <math.h>
#include <string.h>

static float get_f32(const uint8_t *p) {
  float v;
  memcpy(&v, p, 4);
  return v;
}

static int frame(uint8_t type, const uint8_t *payload, int len, uint8_t *out) {
  uint8_t sum = 0;
  out[0] = 0xA5;
  out[1] = type;
  out[2] = (uint8_t)(len & 0xFF);
  out[3] = (uint8_t)(len >> 8);
  for (int i = 0; i < len; ++i) {
    out[4 + i] = payload[i];
    sum += payload[i];
  }
  out[4 + len] = sum;
  return 5 + len;
}

void hil_init(hil_t *h, const mlp_int8_t *net, int history, int takes) {
  memset(h, 0, sizeof(*h));
  ctrl_init(&h->ctrl, net, history, takes);
}

static int handle(hil_t *h, uint8_t type, const uint8_t *p, int len, uint8_t *out) {
  if (type == 'T' && len >= 29) {
    int take = p[0], fit_ok = p[1], has_z = p[2];
    ctrl_real z[4];
    for (int k = 0; k < 4; ++k) z[k] = get_f32(p + 3 + 4 * k);
    h->angle_comp = get_f32(p + 19);
    h->angle_range = get_f32(p + 23);
    int n = p[27] | (p[28] << 8);
    if (n > HIL_MAX_T || len != 29 + 8 * n) return 0;
    h->n = n;
    for (int i = 0; i < HIL_MAX_T + CTRL_LOOKAHEAD; ++i) {
      h->target[i] = i < n ? get_f32(p + 29 + 4 * i) : NAN;
      h->prev_err[i] = i < n ? get_f32(p + 29 + 4 * (n + i)) : NAN;
    }
    ctrl_begin_take(&h->ctrl, take, has_z ? z : 0, fit_ok);
    h->t = 0;
    return frame('K', 0, 0, out);
  }
  if (type == 'S' && len == 8) {
    ctrl_obs_t o;
    int t = h->t < HIL_MAX_T ? h->t : HIL_MAX_T - 1;
    o.target = h->target + t;
    o.prev_err = h->prev_err + t;
    o.fb_err = get_f32(p);
    o.readback_deg = get_f32(p + 4);
    o.angle_comp_deg = h->angle_comp;
    o.angle_range_deg = h->angle_range;
    float a[2];
    ctrl_step(&h->ctrl, &o, &a[0], &a[1]);
    h->t += 1;
    return frame('A', (const uint8_t *)a, 8, out);
  }
  return 0;
}

int hil_feed(hil_t *h, uint8_t byte, uint8_t *out) {
  if (h->got == 0 && byte != 0xA5) return 0;  // wait for a frame start
  h->buf[h->got++] = byte;
  if (h->got == 4) {
    int len = h->buf[2] | (h->buf[3] << 8);
    if (len + 5 > HIL_MAX_FRAME) {
      h->got = 0;
      return 0;
    }
    h->need = len + 5;
  }
  if (h->got < 4 || h->got < h->need) return 0;
  int len = h->need - 5;
  uint8_t sum = 0;
  for (int i = 0; i < len; ++i) sum += h->buf[4 + i];
  h->got = 0;
  if (sum != h->buf[4 + len]) return 0;  // corrupted frame: drop it
  return handle(h, h->buf[1], h->buf + 4, len, out);
}
