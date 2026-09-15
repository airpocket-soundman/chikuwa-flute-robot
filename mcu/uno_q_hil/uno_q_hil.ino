// UNO Q MCU sketch for hardware-in-the-loop (roadmap stage 3). NOT yet tried on hardware.
//
// The simulator (scripts/hil.py, on a PC or on the UNO Q's Linux side) sends frames;
// this sketch runs the controller for every 'S' step and answers with the action.
//
// Setup:
//   1. Copy into this folder: ../policy_mlp.{h,c}, ../controller.{h,c}, ../hil_protocol.{h,c}
//      and the network header written by scripts/quantize_policy.py (e.g. harsh_hist_int8.h).
//   2. Set NET_HEADER / NET_PREFIX / NET_HISTORY below to that network.
//   3. Choose the port. HIL_PORT = Serial1 uses D0 (RX) / D1 (TX) at 3.3 V: connect a USB-UART
//      adapter to the PC. Which object reaches the PC over USB or the Linux side on the UNO Q
//      has to be checked on the board (the Linux <-> MCU link normally goes through the Bridge).
//   4. On the PC: python scripts/hil.py --link serial:COMx --baud 115200 --history 10
//
// The controller uses double by default; add #define CTRL_FLOAT before the includes (and in
// controller.c) to run it in single precision on the Cortex-M33's FPU.
#include <Arduino.h>

extern "C" {
#include "hil_protocol.h"
}

#define NET_HEADER "harsh_hist_int8.h"
#define NET_PREFIX HARSH_HIST_INT8
#define NET_HISTORY 10
#define TAKES 2
#define HIL_PORT Serial1
#define HIL_BAUD 115200

#include NET_HEADER
static const mlp_int8_t NET = MLP_INT8_FROM(NET_PREFIX);
static hil_t H;

void setup() {
  HIL_PORT.begin(HIL_BAUD);
  hil_init(&H, &NET, NET_HISTORY, TAKES);
}

void loop() {
  uint8_t out[16];
  while (HIL_PORT.available()) {
    int n = hil_feed(&H, (uint8_t)HIL_PORT.read(), out);
    if (n) HIL_PORT.write(out, n);
  }
}
