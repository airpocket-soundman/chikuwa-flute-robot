#include "policy_mlp.h"

#include <math.h>

void mlp_int8_forward(const mlp_int8_t *m, const float *x, float *hidden, float *y) {
  for (int h = 0; h < m->hidden; ++h) {
    const int8_t *row = m->w1 + h * m->in;
    float acc = 0.0f;
    for (int i = 0; i < m->in; ++i) acc += (float)row[i] * x[i];
    hidden[h] = tanhf(acc * m->w1_scale + m->b1[h]);
  }
  for (int o = 0; o < m->out; ++o) {
    const int8_t *row = m->w2 + o * m->hidden;
    float acc = 0.0f;
    for (int h = 0; h < m->hidden; ++h) acc += (float)row[h] * hidden[h];
    y[o] = tanhf(acc * m->w2_scale + m->b2[o]);
  }
}
