#include "controller.h"

#include <math.h>
#include <string.h>

#define DT 0.01

static ctrl_real clampr(ctrl_real v, ctrl_real lo, ctrl_real hi) { return v < lo ? lo : (v > hi ? hi : v); }
static ctrl_real sound_speed(ctrl_real temp) { return 331.3 + 0.606 * temp; }
static ctrl_real tube(const ctrl_t *c) { return c->tube_len + c->z[2]; }
static ctrl_real v_in(const ctrl_t *c) { return c->v_in_nom * exp(c->z[0]); }
static ctrl_real v_out(const ctrl_t *c) { return c->v_out_nom * exp(c->z[1]); }
static ctrl_real length_at(const ctrl_t *c, ctrl_real x) {
  ctrl_real l = tube(c) - x + c->end_corr;
  return l > 0.02 ? l : 0.02;
}

// pitch the model expects at plunger position x (optimum angle), cents from A4
static ctrl_real model_cents(const ctrl_t *c, ctrl_real x) {
  return 1200.0 * log2(sound_speed(c->temp_c) / (4.0 * length_at(c, x)) / 440.0) + 100.0 * c->z[3];
}

static ctrl_real x_for_cents(const ctrl_t *c, ctrl_real cents) {
  ctrl_real f = 440.0 * pow(2.0, cents / 1200.0);
  return tube(c) + c->end_corr - sound_speed(c->temp_c) / (4.0 * f);
}

static ctrl_real pwm_for(const ctrl_t *c, ctrl_real x_des) {
  ctrl_real vi = v_in(c), vo = v_out(c);
  ctrl_real vd = clampr((x_des - c->x_hat) / c->track_time, -vo, vi);
  ctrl_real drive = vd / (vd > 0 ? vi : vo);
  if (fabs(drive) < 0.02) return 0.0;
  ctrl_real p = c->deadband + (1.0 - c->deadband) * fabs(drive);
  if (p > 1.0) p = 1.0;
  return drive > 0 ? p : -p;
}

static void dead_reckon(ctrl_t *c, ctrl_real pwm) {
  ctrl_real u = c->q_prev;  // nominal command delay: one step
  c->q_prev = pwm;
  ctrl_real mag = fabs(u), drive = 0.0;
  if (mag >= c->deadband) drive = (u > 0 ? 1.0 : -1.0) * (mag - c->deadband) / (1.0 - c->deadband);
  ctrl_real v_cmd = drive * (drive > 0 ? v_in(c) : v_out(c));
  ctrl_real k = DT / c->tau_v;
  if (k > 1.0) k = 1.0;
  c->v_hat += (v_cmd - c->v_hat) * k;
  c->x_hat = clampr(c->x_hat + c->v_hat * DT, 0.0, c->stroke);
}

void ctrl_init(ctrl_t *c, const mlp_int8_t *net, int history, int takes) {
  memset(c, 0, sizeof(*c));
  c->fb_gain = 0.05; c->ilc_gain = 0.5; c->scale = 0.3; c->track_time = 0.04; c->rest_angle = -1.0; c->max_err = 300.0;
  c->angle_lead = 3; c->history = history; c->takes = takes; c->net = net;
  c->tube_len = 0.150; c->end_corr = 0.005; c->temp_c = 25.0; c->stroke = 0.120;
  c->v_in_nom = 0.150; c->v_out_nom = 0.150; c->deadband = 0.20; c->tau_v = 0.030;
  // lead = cmd_delay + round(tau_v / DT) + 1 (nominal), capped by the look-ahead
  c->lead = 1 + (int)lround(c->tau_v / DT) + 1;
  if (c->lead > CTRL_LOOKAHEAD - 1) c->lead = CTRL_LOOKAHEAD - 1;
}

void ctrl_begin_take(ctrl_t *c, int take, const ctrl_real *z, int fit_ok) {
  c->t = 0;
  c->take = take;
  if (z) {
    memcpy(c->z_pending, z, sizeof(c->z_pending));
    c->pending = 1;
  }
  c->mask_ilc = (take == 1 && fit_ok);
}

void ctrl_step(ctrl_t *c, const ctrl_obs_t *o, float *pwm_out, float *angle_out) {
  const int L = CTRL_LOOKAHEAD, H = CTRL_HORIZON, t0 = (c->t == 0);
  if (t0) c->x_hat = c->v_hat = 0.0;  // every take starts from home

  const int fb_valid = isfinite(o->fb_err);
  const ctrl_real fb = fb_valid ? clampr(o->fb_err / 100.0, -3.0, 3.0) : 0.0;
  ctrl_real out[3] = {0.0, 0.0, 0.0};

  // learned residual: features exactly as flute_rl.feedback.FeedbackResidualPolicy
  if (c->net) {
    float feats[CTRL_FEATURES(CTRL_MAX_HIST)], hid[64], y[3];
    ctrl_real mc = model_cents(c, c->x_hat);
    for (int j = 0; j < H; ++j) {
      int m = isfinite(o->target[j]);
      feats[j] = m ? (float)clampr((o->target[j] - mc) / 100.0, -3.0, 3.0) : 0.0f;
      feats[H + j] = (float)m;
    }
    if (c->history) {
      for (int k = 0; k < c->history - 1; ++k) memcpy(c->hist[k], c->hist[k + 1], sizeof(c->hist[k]));
      c->hist[c->history - 1][0] = fb;
      c->hist[c->history - 1][1] = c->last[0];
      c->hist[c->history - 1][2] = c->last[1];
    }
    int n = 2 * H;
    feats[n++] = (float)(c->v_hat / 0.15);
    feats[n++] = (float)((o->readback_deg - o->angle_comp_deg) / o->angle_range_deg);
    feats[n++] = (float)fb;
    feats[n++] = (float)fb_valid;
    feats[n++] = (float)(c->takes > 1 ? (ctrl_real)c->take / (c->takes - 1) : 0.0);
    feats[n++] = (float)c->last[0];
    feats[n++] = (float)c->last[1];
    feats[n++] = 1.0f;
    for (int k = 0; k < c->history; ++k)
      for (int j = 0; j < 3; ++j) feats[n++] = (float)c->hist[k][j];
    mlp_int8_forward(c->net, feats, hid, y);
    out[0] = y[0]; out[1] = y[1]; out[2] = y[2];
  }
  const ctrl_real gain_scale = c->net ? 1.0 + out[2] : 1.0;

  // live feedback: move the plunger belief by what the delayed pitch error says
  if (!t0 && fb_valid) {
    ctrl_real e = clampr(o->fb_err, -c->max_err, c->max_err);
    if (fabs(e) < c->max_err) {
      ctrl_real dx = e * length_at(c, c->x_hat) * log(2.0) / 1200.0;
      c->x_hat = clampr(c->x_hat + c->fb_gain * gain_scale * dx, 0.0, c->stroke);
    }
  }

  // a new take: adopt the rig identified from the previous one
  if (t0) {
    if (c->pending) {
      memcpy(c->z, c->z_pending, sizeof(c->z));
      c->pending = 0;
    }
    c->q_prev = 0.0;
  }

  // ILC on the previous take's error
  int i = c->lead < L - 1 ? c->lead : L - 1;
  if (isfinite(o->prev_err[i]) && !c->mask_ilc && c->t + i < CTRL_MAX_T + L)
    c->offset[c->t + i] += c->ilc_gain * clampr(o->prev_err[i], -1800.0, 1800.0);

  // physics prior: aim at the first note at/after the lead time (else the last note in view)
  int idx = -1;
  for (int j = c->lead; j < L; ++j)
    if (isfinite(o->target[j])) { idx = j; break; }
  if (idx < 0)
    for (int j = L - 1; j >= 0; --j)
      if (isfinite(o->target[j])) { idx = j; break; }
  ctrl_real x_des = c->x_hat;
  if (idx >= 0) {
    ctrl_real aim = o->target[idx] - 100.0 * c->z[3] - c->offset[c->t + idx];
    x_des = x_for_cents(c, aim);
  }
  x_des = clampr(x_des, 0.0, c->stroke);
  ctrl_real pwm = pwm_for(c, x_des);
  int lead_a = c->angle_lead < L - 1 ? c->angle_lead : L - 1;
  int soon = isfinite(o->target[lead_a]) || isfinite(o->target[0]);
  ctrl_real angle = soon ? o->angle_comp_deg / o->angle_range_deg : c->rest_angle;

  if (c->net) {
    pwm = clampr(pwm + c->scale * out[0], -1.0, 1.0);
    angle = clampr(angle + c->scale * out[1], -1.0, 1.0);
  }
  dead_reckon(c, pwm);  // the PWM actually sent
  c->last[0] = pwm;
  c->last[1] = angle;
  c->t += 1;
  *pwm_out = (float)pwm;
  *angle_out = (float)angle;
}
