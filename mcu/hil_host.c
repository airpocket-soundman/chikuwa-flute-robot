// Stand-in for the MCU on the PC: speaks the HIL protocol over stdin/stdout, so
// scripts/hil.py can be tested without hardware ("software in the loop").
// Build: python -m ziglang cc -O2 -Imcu [-I<dir of net header> -DHIL_NET_HEADER='"net.h"' -DHIL_NET=PREFIX]
//        mcu/hil_host.c mcu/hil_protocol.c mcu/controller.c mcu/policy_mlp.c -o hil_host
#include <stdio.h>
#include <stdlib.h>

#include "hil_protocol.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#ifdef HIL_NET_HEADER
#include HIL_NET_HEADER
static const mlp_int8_t NET = MLP_INT8_FROM(HIL_NET);
#define NET_PTR (&NET)
#else
#define NET_PTR NULL
#endif

static hil_t H;

int main(int argc, char **argv) {
#ifdef _WIN32
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  int history = argc > 1 ? atoi(argv[1]) : 0;
  int takes = argc > 2 ? atoi(argv[2]) : 2;
  hil_init(&H, NET_PTR, history, takes);
  uint8_t out[16];
  int c;
  while ((c = getchar()) != EOF) {
    int n = hil_feed(&H, (uint8_t)c, out);
    if (n) {
      fwrite(out, 1, n, stdout);
      fflush(stdout);
    }
  }
  return 0;
}
