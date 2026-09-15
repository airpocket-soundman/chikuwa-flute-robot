// One control step of the live-feedback controller, in C for the UNO Q's MCU.
//
// This is flute_rl's FeedbackResidualPolicy(FeedbackPolicy(...), net) for a
// single rig: physics prior (dead reckoning, look-ahead aim, PWM inversion),
// live correction of the plunger belief from the delayed pitch error, ILC
// across takes, and the learned residual network (int8, policy_mlp.h).
//
// Split of work on the UNO Q:
// * Linux side, between takes: record the take, identify the rig (numpy
//   flute_rl.adapt.fit_rig) and send the fitted z = (log s_in, log s_out,
//   tube length offset [m], cents offset / 100) with ctrl_begin_take().
// * MCU, every 10 ms: ctrl_step() with the target look-ahead, the previous
//   take's error at the same place, the delayed pitch error heard now and the
//   servo read-back; it returns the PWM and the angle command.
//
// ctrl_real is double by default (bit-for-bit checks against numpy on the PC);
// build with -DCTRL_FLOAT for single precision on the MCU.
#pragma once
#include "policy_mlp.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef CTRL_FLOAT
typedef float ctrl_real;
#else
typedef double ctrl_real;
#endif

#define CTRL_LOOKAHEAD 50
#define CTRL_HORIZON 30
#define CTRL_MAX_T 4096
#define CTRL_MAX_HIST 16
#define CTRL_FEATURES(history) (2 * CTRL_HORIZON + 8 + 3 * (history))

typedef struct {
  // settings (ctrl_init sets the defaults used on the PC)
  ctrl_real fb_gain, ilc_gain, scale, track_time, rest_angle, max_err;
  int angle_lead, lead, history, takes;
  const mlp_int8_t *net;  // NULL: hand-made feedback only
  // nominal rig model (flute_rl.sim.FluteParams defaults)
  ctrl_real tube_len, end_corr, temp_c, stroke, v_in_nom, v_out_nom, deadband, tau_v;
  // identified model and take bookkeeping
  ctrl_real z[4], z_pending[4];
  int pending, mask_ilc, t, take;
  // state
  ctrl_real x_hat, v_hat, q_prev, last[2], hist[CTRL_MAX_HIST][3];
  ctrl_real offset[CTRL_MAX_T + CTRL_LOOKAHEAD];
} ctrl_t;

typedef struct {
  const float *target;    // CTRL_LOOKAHEAD target cents from now on (NaN = rest)
  const float *prev_err;  // previous take's measured error at the same places (NaN = none)
  float fb_err;           // delayed pitch error heard now [cents] (NaN = nothing heard)
  float readback_deg;     // servo read-back angle
  float angle_comp_deg;   // sounding-compensation angle
  float angle_range_deg;  // angle command +-1 maps to +- this
} ctrl_obs_t;

void ctrl_init(ctrl_t *c, const mlp_int8_t *net, int history, int takes);
// Call before the first step of each take (take = 0, 1, ...). z: newly identified rig, or NULL to keep.
// fit_ok: the identification succeeded (then take 2 does not feed take 1's errors to the ILC).
void ctrl_begin_take(ctrl_t *c, int take, const ctrl_real *z, int fit_ok);
void ctrl_step(ctrl_t *c, const ctrl_obs_t *o, float *pwm, float *angle);

#ifdef __cplusplus
}
#endif
