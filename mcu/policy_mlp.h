// int8 MLP inference for the UNO Q's MCU (STM32U585, Cortex-M33).
//
// The network is the policy trained on the PC (flute_rl.policy.MLP: one tanh
// hidden layer, tanh output) exported by scripts/quantize_policy.py: int8
// weights with one float scale per matrix, stored output-major, and float
// biases. Activations are float, so no requantisation is needed.
//
// Usage with a generated header (prefix HARSH_HIST_INT8 for example):
//
//   #include "harsh_hist_int8.h"
//   static const mlp_int8_t policy = MLP_INT8_FROM(HARSH_HIST_INT8);
//   float hidden[HARSH_HIST_INT8_HIDDEN], out[HARSH_HIST_INT8_OUT];
//   mlp_int8_forward(&policy, features, hidden, out);
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int in, hidden, out;
  const int8_t *w1;  // hidden x in
  float w1_scale;
  const float *b1;   // hidden
  const int8_t *w2;  // out x hidden
  float w2_scale;
  const float *b2;   // out
} mlp_int8_t;

#define MLP_INT8_FROM(P) \
  { P##_IN, P##_HIDDEN, P##_OUT, P##_W1, P##_W1_SCALE, P##_B1, P##_W2, P##_W2_SCALE, P##_B2 }

// x: in values; hidden: scratch of `hidden` floats; y: out values in [-1, 1].
void mlp_int8_forward(const mlp_int8_t *m, const float *x, float *hidden, float *y);

#ifdef __cplusplus
}
#endif
