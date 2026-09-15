// Hardware-in-the-loop link between the simulator and the MCU's controller (roadmap stage 3).
//
// The simulator side (a PC, or the UNO Q's Linux side) owns the rig; the MCU
// owns the controller (controller.h). Frames on a byte stream:
//
//   0xA5, type, length (uint16 LE), payload[length], checksum (sum of payload bytes & 0xFF)
//
// Simulator -> MCU
//   'T' take setup:  take u8, fit_ok u8, has_z u8, z[4] f32, angle_comp_deg f32, angle_range_deg f32,
//                    n u16, target[n] f32 (NaN = rest), prev_err[n] f32 (NaN = none)
//   'S' step:        fb_err f32 (NaN = nothing heard), readback_deg f32
// MCU -> simulator
//   'A' action:      pwm f32, angle f32     (reply to every 'S')
//   'K' ack          (reply to 'T')
//
// All numbers little-endian. The whole take (targets, the previous take's
// errors, the identified rig) travels between takes, so each 10 ms step is
// only 13 bytes out and 13 bytes back (about 2.3 ms at 115200 baud).
#pragma once
#include <stdint.h>

#include "controller.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HIL_MAX_T 2048
#define HIL_MAX_FRAME (8 + 23 + 8 * HIL_MAX_T)

typedef struct {
  ctrl_t ctrl;
  float target[HIL_MAX_T + CTRL_LOOKAHEAD];
  float prev_err[HIL_MAX_T + CTRL_LOOKAHEAD];
  float angle_comp, angle_range;
  int n, t;
  // frame parser
  uint8_t buf[HIL_MAX_FRAME];
  int got, need;
} hil_t;

void hil_init(hil_t *h, const mlp_int8_t *net, int history, int takes);
// Feed received bytes one at a time. When a reply is ready it is written to `out`
// and its length returned (0 otherwise). `out` needs 16 bytes.
int hil_feed(hil_t *h, uint8_t byte, uint8_t *out);

#ifdef __cplusplus
}
#endif
