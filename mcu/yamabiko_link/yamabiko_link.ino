// UNO Q MCU: stream the ICS-43434 on SAI1 to the Linux side without gaps, for Yamabiko No.1.
//
// SAI1 block A is set up as in mcu/sai1_mic_test (48 kHz, left slot, registers driven directly),
// but a GPDMA channel now copies the FIFO into a ring buffer on its own, with no interrupt:
// the channel's linked list points back at itself, so it restarts the block forever. The loop
// reads the channel's remaining byte count to know how far the DMA has written.
//
// The left slot is low-pass filtered and decimated by 3 to 16 kHz, 16-bit (the top 16 of the 24
// bits), and sent over Serial1 (/dev/ttyHS1 on the Linux side; stop arduino-router first) in
// packets of PKT_SAMPLES (5 ms). 16 kHz x 2 bytes = 32 kB/s, about a third of what 921600 baud carries.
//
// Link protocol, little endian. MCU -> Linux, every packet:
//   A5 5A type len(u16) payload[len] sum(u8)       sum = 8-bit sum of type, len and payload
//   type 'A' audio:  seq(u16) first(u32, 16 kHz sample index) samples(i16 x n)
//   type 'P' pong:   token(u32) frame48(u32, SAI frames captured when the ping was read)
//                    overruns(u32) max_loop_us(u32)
//   type 'T' text:   one line (info)
// Linux -> MCU, single bytes: 'S' start streaming, 'X' stop, 'i' info,
//   'P' + token(u32): ping, answered at once (between two audio packets).
#include <Arduino.h>
#include <stm32u5xx.h>

#define LINK Serial1
#define LINK_BAUD 921600

#define RING_FRAMES 2048          // 42.7 ms of stereo 32-bit words at 48 kHz = 16 KB
#define DECIM 3                   // 48 kHz -> 16 kHz
#define PKT_SAMPLES 80            // 5 ms at 16 kHz
#define NTAPS 31                  // anti-alias FIR at 48 kHz, cut-off 7 kHz
#define DMA_CH GPDMA1_Channel7
#define DMA_REQ_SAI1_A 36u        // GPDMA1_REQUEST_SAI1_A (stm32u5xx_hal_dma.h)

static volatile uint32_t ring[RING_FRAMES * 2] __attribute__((aligned(32)));
// Linked-list item: the words the channel reloads after each block, in register order
// CBR1, CDAR, CLLR (UB1 | UDA | ULL). It must share the upper 16 address bits with CLBAR.
static uint32_t lli[3] __attribute__((aligned(16)));

static float taps[NTAPS];
static float hist[NTAPS];
static uint32_t hpos;
static uint32_t phase;            // 0..DECIM-1

static uint32_t cyc_per_us;
static uint32_t read_word;        // next word of the ring to read
static uint32_t wraps;            // completed passes of the DMA over the ring
static uint32_t frames_seen;      // SAI frames consumed since streaming started
static uint32_t left_parity;      // 0 or 1: which word of a frame is the left slot
static bool streaming;
static uint32_t overruns, max_loop_us;

static int16_t pkt[PKT_SAMPLES];
static uint32_t pkt_n, pkt_first, out_index;
static uint16_t seq;
static uint32_t pong_token, pong_frame;

// ---- clocks, pins, SAI (as in mcu/sai1_mic_test) ------------------------------------

static bool pll3_from_hse() {
  if (!(RCC->CR & RCC_CR_HSERDY)) return false;
  RCC->CR &= ~RCC_CR_PLL3ON;
  while (RCC->CR & RCC_CR_PLL3RDY) {}
  RCC->PLL3CFGR = (3u << RCC_PLL3CFGR_PLL3SRC_Pos) | (0u << RCC_PLL3CFGR_PLL3RGE_Pos)
                | ((4u - 1u) << RCC_PLL3CFGR_PLL3M_Pos);
  RCC->PLL3DIVR = ((49u - 1u) << RCC_PLL3DIVR_PLL3N_Pos) | ((4u - 1u) << RCC_PLL3DIVR_PLL3P_Pos)
                | ((4u - 1u) << RCC_PLL3DIVR_PLL3Q_Pos) | ((4u - 1u) << RCC_PLL3DIVR_PLL3R_Pos);
  RCC->PLL3FRACR = 1245u << RCC_PLL3FRACR_PLL3FRACN_Pos;
  RCC->PLL3CFGR |= RCC_PLL3CFGR_PLL3FRACEN | RCC_PLL3CFGR_PLL3PEN;
  RCC->CR |= RCC_CR_PLL3ON;
  for (uint32_t t = 0; !(RCC->CR & RCC_CR_PLL3RDY); t++) {
    if (t > 1000000u) return false;
  }
  return true;
}

static const char *clk_name = "none";

static void sai_setup() {
  uint32_t sel = 1u, mckdiv = 16u;
  clk_name = "PLL3P(HSE)";
  if (!pll3_from_hse()) {   // HSI16 / 6 / 64 = 41.67 kHz; the stream then runs at 13.9 kHz
    RCC->CR |= RCC_CR_HSION;
    while (!(RCC->CR & RCC_CR_HSIRDY)) {}
    sel = 4u; mckdiv = 6u; clk_name = "HSI16";
  }
  RCC->CCIPR2 = (RCC->CCIPR2 & ~RCC_CCIPR2_SAI1SEL) | (sel << RCC_CCIPR2_SAI1SEL_Pos);

  RCC->AHB2ENR1 |= RCC_AHB2ENR1_GPIOEEN;
  (void)RCC->AHB2ENR1;
  for (uint32_t pin = 4; pin <= 6; pin++) {
    GPIOE->MODER = (GPIOE->MODER & ~(3u << (2 * pin))) | (2u << (2 * pin));
    GPIOE->OSPEEDR = (GPIOE->OSPEEDR & ~(3u << (2 * pin))) | (2u << (2 * pin));
    GPIOE->AFR[0] = (GPIOE->AFR[0] & ~(0xFu << (4 * pin))) | (13u << (4 * pin));
  }
  GPIOE->PUPDR = (GPIOE->PUPDR & ~(3u << (2 * 6))) | (2u << (2 * 6));

  RCC->APB2ENR |= RCC_APB2ENR_SAI1EN;
  (void)RCC->APB2ENR;
  SAI_Block_TypeDef *a = SAI1_Block_A;
  a->CR1 = 0;
  while (a->CR1 & SAI_xCR1_SAIEN) {}
  a->CR1 = SAI_xCR1_MODE_0 | (7u << SAI_xCR1_DS_Pos) | SAI_xCR1_CKSTR | SAI_xCR1_NODIV
         | (mckdiv << SAI_xCR1_MCKDIV_Pos) | SAI_xCR1_DMAEN;
  a->CR2 = SAI_xCR2_FFLUSH;                                  // FTH = 0: a request per word
  a->FRCR = ((64u - 1u) << SAI_xFRCR_FRL_Pos) | ((32u - 1u) << SAI_xFRCR_FSALL_Pos)
          | SAI_xFRCR_FSDEF | SAI_xFRCR_FSOFF;
  a->SLOTR = (2u << SAI_xSLOTR_SLOTSZ_Pos) | ((2u - 1u) << SAI_xSLOTR_NBSLOT_Pos)
           | (3u << SAI_xSLOTR_SLOTEN_Pos);
  a->IMR = 0;
  a->CLRFR = 0xFFFFFFFFu;
}

// ---- DMA ring ---------------------------------------------------------------------

static const uint32_t RING_BYTES = sizeof(ring);

static void dma_start() {
  RCC->AHB1ENR |= RCC_AHB1ENR_GPDMA1EN;
  (void)RCC->AHB1ENR;
  DMA_Channel_TypeDef *c = DMA_CH;
  if (c->CCR & DMA_CCR_EN) {                                 // a running channel ignores EN = 0:
    c->CCR |= DMA_CCR_SUSP;                                  // suspend it first, then reset
    for (uint32_t t = 0; !(c->CSR & DMA_CSR_SUSPF) && t < 1000000u; t++) {}
  }
  c->CCR = DMA_CCR_RESET;
  for (uint32_t t = 0; (c->CCR & DMA_CCR_EN) && t < 1000000u; t++) {}
  c->CFCR = 0x7F00u;                                         // clear all flags

  uint32_t link = DMA_CLLR_UB1 | DMA_CLLR_UDA | DMA_CLLR_ULL | ((uint32_t)lli & 0xFFFCu);
  lli[0] = RING_BYTES;                                       // CBR1: BNDT
  lli[1] = (uint32_t)ring;                                   // CDAR
  lli[2] = link;                                             // CLLR: back to itself

  c->CLBAR = (uint32_t)lli & 0xFFFF0000u;
  c->CTR1 = (2u << DMA_CTR1_SDW_LOG2_Pos)                    // word from SAI DR, no increment
          | (2u << DMA_CTR1_DDW_LOG2_Pos) | DMA_CTR1_DINC;   // word to the ring, incrementing
  c->CTR2 = (DMA_REQ_SAI1_A << DMA_CTR2_REQSEL_Pos);         // hardware request, block-level TC
  c->CBR1 = RING_BYTES;
  c->CSAR = (uint32_t)&SAI1_Block_A->DR;
  c->CDAR = (uint32_t)ring;
  c->CLLR = link;
  c->CCR = (3u << DMA_CCR_PRIO_Pos) | DMA_CCR_EN;            // high priority, no interrupts
}

// Words the DMA has written into the current pass (0 .. RING_FRAMES * 2).
static inline uint32_t dma_words() {
  return (RING_BYTES - (DMA_CH->CBR1 & DMA_CBR1_BNDT)) / 4u;
}

// Words written since the stream started. The count is read before the pass-complete flag: if
// the channel restarts in between, the flag is seen and the count read again after the restart.
static uint32_t dma_total_words() {
  uint32_t w = dma_words();
  if (DMA_CH->CSR & DMA_CSR_TCF) {
    DMA_CH->CFCR = DMA_CFCR_TCF;
    wraps++;
    w = dma_words();
  }
  return wraps * RING_FRAMES * 2 + w;
}

static void stream_restart() {
  SAI_Block_TypeDef *a = SAI1_Block_A;
  // Keep SCK running across restarts would be better (docs/i2s_mic_trial.md), but a restart only
  // happens when the Linux side (re)starts the stream; the first 20 ms after it are unsettled.
  a->CR1 &= ~SAI_xCR1_SAIEN;
  while (a->CR1 & SAI_xCR1_SAIEN) {}
  a->CR2 = SAI_xCR2_FFLUSH;
  a->CLRFR = 0xFFFFFFFFu;
  dma_start();
  a->CR1 |= SAI_xCR1_SAIEN;
  read_word = 0; wraps = 0; frames_seen = 0; left_parity = 0;
  for (uint32_t i = 0; i < NTAPS; i++) hist[i] = 0.0f;
  hpos = 0; phase = 0;
  pkt_n = 0; out_index = 0; seq = 0;
  overruns = 0; max_loop_us = 0;
}

// ---- link -------------------------------------------------------------------------

static void send_packet(uint8_t type, const uint8_t *a, uint16_t na, const uint8_t *b, uint16_t nb) {
  uint8_t head[5] = {0xA5, 0x5A, type, (uint8_t)((na + nb) & 0xFF), (uint8_t)((na + nb) >> 8)};
  uint8_t sum = head[2] + head[3] + head[4];
  for (uint16_t i = 0; i < na; i++) sum += a[i];
  for (uint16_t i = 0; i < nb; i++) sum += b[i];
  LINK.write(head, 5);
  if (na) LINK.write(a, na);
  if (nb) LINK.write(b, nb);
  LINK.write(&sum, 1);
}

static void send_text(const char *s) {
  send_packet('T', (const uint8_t *)s, strlen(s), nullptr, 0);
}

static void send_pong() {
  uint32_t p[4] = {pong_token, pong_frame, overruns, max_loop_us};
  send_packet('P', (const uint8_t *)p, sizeof(p), nullptr, 0);
}

static void send_audio() {
  uint8_t h[6];
  memcpy(h, &seq, 2);
  memcpy(h + 2, &pkt_first, 4);
  send_packet('A', h, 6, (const uint8_t *)pkt, pkt_n * 2);
  seq++;
  pkt_n = 0;
}

static uint32_t frames_now() {  // SAI frames captured so far in this stream
  return dma_total_words() / 2;
}

static void info() {
  char s[160];
  snprintf(s, sizeof(s), "INFO clock %s cyc_per_us %lu streaming %d frames %lu overruns %lu max_loop_us %lu",
           clk_name, (unsigned long)cyc_per_us, streaming, (unsigned long)frames_now(),
           (unsigned long)overruns, (unsigned long)max_loop_us);
  send_text(s);
  snprintf(s, sizeof(s), "DMA CCR %08lX CSR %08lX CBR1 %08lX CLLR %08lX SAI SR %08lX",
           DMA_CH->CCR, DMA_CH->CSR, DMA_CH->CBR1, DMA_CH->CLLR, SAI1_Block_A->SR);
  send_text(s);
}

static void commands() {
  while (LINK.available()) {
    int c = LINK.peek();
    if (c == 'P') {
      if (LINK.available() < 5) return;          // wait for the whole token
      LINK.read();
      uint8_t t[4];
      for (int i = 0; i < 4; i++) t[i] = LINK.read();
      pong_frame = frames_now();
      memcpy(&pong_token, t, 4);
      send_pong();                               // between packets, never inside one
      continue;
    }
    LINK.read();
    if (c == 'S') { stream_restart(); streaming = true; }
    else if (c == 'X') streaming = false;
    else if (c == 'i') info();
  }
}

// ---- audio ------------------------------------------------------------------------

static inline void push_sample(float x) {
  hist[hpos] = x;
  hpos = (hpos + 1) % NTAPS;
  if (++phase < DECIM) return;
  phase = 0;
  float y = 0.0f;
  uint32_t k = hpos;                        // oldest sample
  for (uint32_t i = 0; i < NTAPS; i++) {
    y += taps[i] * hist[k];
    k = (k + 1 == NTAPS) ? 0 : k + 1;
  }
  float q = y / 256.0f;                     // 24-bit -> top 16 bits, rounded
  int32_t v = (int32_t)(q < 0.0f ? q - 0.5f : q + 0.5f);
  if (v > 32767) v = 32767;
  if (v < -32768) v = -32768;
  if (pkt_n == 0) pkt_first = out_index;
  pkt[pkt_n++] = (int16_t)v;
  out_index++;
}

static void pump() {
  uint32_t total = dma_total_words();                     // words written so far
  uint32_t done = frames_seen * 2;
  if (total - done > RING_FRAMES * 2) {                   // fell a whole ring behind
    overruns++;
    frames_seen = total / 2 - RING_FRAMES / 2;
    done = frames_seen * 2;
  }
  while (done + 2 <= total) {
    uint32_t i = done % (RING_FRAMES * 2);
    int32_t l = (int32_t)ring[i + left_parity];
    if (frames_seen == 64) {                              // pick the slot that carries sound
      uint32_t e0 = 0, e1 = 0;
      for (uint32_t j = 0; j < 128; j += 2) { e0 |= ring[j]; e1 |= ring[j + 1]; }
      left_parity = (e0 == 0 && e1 != 0) ? 1 : 0;
    }
    push_sample((float)(l >> 8));
    frames_seen++;
    done += 2;
    if (pkt_n == PKT_SAMPLES) send_audio();
  }
}

// ---- main ---------------------------------------------------------------------------

void setup() {
  LINK.begin(LINK_BAUD);
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  uint32_t c0 = DWT->CYCCNT;
  delay(100);
  cyc_per_us = (DWT->CYCCNT - c0) / 100000u;

  // Windowed-sinc low-pass, cut-off 7 kHz at 48 kHz, Hamming window, unity DC gain.
  const float fc = 7000.0f / 48000.0f;
  float s = 0.0f;
  for (int i = 0; i < NTAPS; i++) {
    int m = i - (NTAPS - 1) / 2;
    float h = (m == 0) ? 2.0f * fc : sinf(2.0f * PI * fc * m) / (PI * m);
    h *= 0.54f - 0.46f * cosf(2.0f * PI * i / (NTAPS - 1));
    taps[i] = h;
    s += h;
  }
  for (int i = 0; i < NTAPS; i++) taps[i] /= s;

  sai_setup();
  stream_restart();
  streaming = false;
  send_text("READY yamabiko_link");
}

void loop() {
  uint32_t t0 = DWT->CYCCNT;
  commands();
  if (streaming) {
    pump();
  } else {
    dma_total_words();                        // keep counting passes while idle
  }
  uint32_t us = (DWT->CYCCNT - t0) / cyc_per_us;
  if (us > max_loop_us) max_loop_us = us;
}
