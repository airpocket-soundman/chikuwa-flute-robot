// UNO Q MCU: read the ICS-43434 on SAI1 block A by driving the registers directly.
//
// The Arduino Zephyr core (0.53.1) ships its loader without CONFIG_I2S, so there is no
// Zephyr i2s driver to call. The sketch runs privileged with the MPU off, so it programs
// RCC, GPIOE and SAI1 itself through the CMSIS device header.
//
// Wiring (docs/i2s_mic_trial.md): PE5 = SCK (J15-8), PE4 = WS (J15-9), PE6 = SD (J15-10),
// all AF13; VDD 3.3 V, LR to GND (left slot).
//
// Link: Serial1 (the UART the Bridge normally uses, /dev/ttyHS1 on the Linux side) at
// LINK_BAUD. Stop arduino-router first. Commands, one character each:
//   i  clocks and SAI registers
//   c  capture: restart the SAI, drop SETTLE_FRAMES, keep CAP_FRAMES of the left slot,
//      print stats, then send the left slot as S32_LE ("DATA <bytes>\n" + raw + "\nEND\n")
//   h  same capture, stats only
#include <Arduino.h>
#include <stm32u5xx.h>

#define LINK Serial1
#define LINK_BAUD 921600

#define CAP_FRAMES 8192          // 32 KB; the sketch lives in a 128 KB llext heap
#define SETTLE_MS 25             // ICS-43434: 10.7 ms silent, sensitivity settled by 20 ms

static int32_t cap[CAP_FRAMES];
static float rate_hz;            // nominal SAI frame rate for the clock path in use
static const char *clk_name;

// ---- clocks -------------------------------------------------------------------------

// PLL3 from HSE 16 MHz: /4 = 4 MHz ref, x49.152 (N = 49, FRACN = 1245) = 196.608 MHz VCO,
// /P 4 = 49.152 MHz kernel clock, /MCKDIV 16 = 3.072 MHz SCK = 48 kHz x 64.
static bool pll3_from_hse() {
  if (!(RCC->CR & RCC_CR_HSERDY)) return false;
  RCC->CR &= ~RCC_CR_PLL3ON;
  while (RCC->CR & RCC_CR_PLL3RDY) {}
  RCC->PLL3CFGR = (3u << RCC_PLL3CFGR_PLL3SRC_Pos)      // HSE
                | (0u << RCC_PLL3CFGR_PLL3RGE_Pos)      // 4-8 MHz input
                | ((4u - 1u) << RCC_PLL3CFGR_PLL3M_Pos);
  RCC->PLL3DIVR = ((49u - 1u) << RCC_PLL3DIVR_PLL3N_Pos)
                | ((4u - 1u) << RCC_PLL3DIVR_PLL3P_Pos)
                | ((4u - 1u) << RCC_PLL3DIVR_PLL3Q_Pos)
                | ((4u - 1u) << RCC_PLL3DIVR_PLL3R_Pos);
  RCC->PLL3FRACR = 1245u << RCC_PLL3FRACR_PLL3FRACN_Pos;
  RCC->PLL3CFGR |= RCC_PLL3CFGR_PLL3FRACEN | RCC_PLL3CFGR_PLL3PEN;
  RCC->CR |= RCC_CR_PLL3ON;
  for (uint32_t t = 0; !(RCC->CR & RCC_CR_PLL3RDY); t++) {
    if (t > 1000000u) return false;
  }
  return true;
}

// Returns MCKDIV for the chosen kernel clock and sets rate_hz / clk_name.
static uint32_t select_sai_clock() {
  uint32_t sel, mckdiv;
  if (pll3_from_hse()) {
    sel = 1u; mckdiv = 16u; rate_hz = 49152000.0f / 16.0f / 64.0f; clk_name = "PLL3P(HSE)";
  } else {
    // Fallback: HSI16 / 6 = 2.667 MHz SCK = 41.67 kHz, still in the high-performance range.
    RCC->CR |= RCC_CR_HSION;
    while (!(RCC->CR & RCC_CR_HSIRDY)) {}
    sel = 4u; mckdiv = 6u; rate_hz = 16000000.0f / 6.0f / 64.0f; clk_name = "HSI16";
  }
  RCC->CCIPR2 = (RCC->CCIPR2 & ~RCC_CCIPR2_SAI1SEL) | (sel << RCC_CCIPR2_SAI1SEL_Pos);
  return mckdiv;
}

// ---- pins and SAI -------------------------------------------------------------------

static void pins_af13() {
  RCC->AHB2ENR1 |= RCC_AHB2ENR1_GPIOEEN;
  (void)RCC->AHB2ENR1;
  for (uint32_t pin = 4; pin <= 6; pin++) {
    GPIOE->MODER = (GPIOE->MODER & ~(3u << (2 * pin))) | (2u << (2 * pin));
    GPIOE->OSPEEDR = (GPIOE->OSPEEDR & ~(3u << (2 * pin))) | (2u << (2 * pin));
    GPIOE->AFR[0] = (GPIOE->AFR[0] & ~(0xFu << (4 * pin))) | (13u << (4 * pin));
  }
  // SD floats between slots; the datasheet wants 100k to GND. Add the internal pull-down too.
  GPIOE->PUPDR = (GPIOE->PUPDR & ~(3u << (2 * 6))) | (2u << (2 * 6));
}

static uint32_t sai_cr1;

static void sai_setup() {
  uint32_t mckdiv = select_sai_clock();
  RCC->APB2ENR |= RCC_APB2ENR_SAI1EN;
  (void)RCC->APB2ENR;

  SAI_Block_TypeDef *a = SAI1_Block_A;
  a->CR1 = 0;
  while (a->CR1 & SAI_xCR1_SAIEN) {}
  // Master receiver, free protocol, 32-bit data (24 bits MSB-aligned, low byte zero),
  // sample on the rising SCK edge (CKSTR = 1, I2S), no MCLK: SCK = ker_ck / MCKDIV.
  sai_cr1 = SAI_xCR1_MODE_0
          | (7u << SAI_xCR1_DS_Pos)
          | SAI_xCR1_CKSTR
          | SAI_xCR1_NODIV
          | (mckdiv << SAI_xCR1_MCKDIV_Pos);
  a->CR1 = sai_cr1;
  a->CR2 = SAI_xCR2_FFLUSH;
  // I2S frame: 64 bits, WS low for the first 32 (left), WS changes one bit before the MSB.
  a->FRCR = ((64u - 1u) << SAI_xFRCR_FRL_Pos)
          | ((32u - 1u) << SAI_xFRCR_FSALL_Pos)
          | SAI_xFRCR_FSDEF
          | SAI_xFRCR_FSOFF;
  a->SLOTR = (2u << SAI_xSLOTR_SLOTSZ_Pos)               // 32-bit slots
           | ((2u - 1u) << SAI_xSLOTR_NBSLOT_Pos)
           | (3u << SAI_xSLOTR_SLOTEN_Pos);
  a->IMR = 0;
  a->CLRFR = 0xFFFFFFFFu;
}

static void sai_stop() {
  SAI1_Block_A->CR1 = sai_cr1;
  while (SAI1_Block_A->CR1 & SAI_xCR1_SAIEN) {}
  SAI1_Block_A->CR2 = SAI_xCR2_FFLUSH;
  SAI1_Block_A->CLRFR = 0xFFFFFFFFu;
}

static void sai_start() {
  SAI1_Block_A->CR1 = sai_cr1 | SAI_xCR1_SAIEN;
}

static inline uint32_t sai_read() {
  while (((SAI1_Block_A->SR & SAI_xSR_FLVL) >> SAI_xSR_FLVL_Pos) == 0) {}
  return SAI1_Block_A->DR;
}

// ---- capture ------------------------------------------------------------------------

struct slot_stats {
  int64_t sum;
  double sumsq;
  int32_t minv, maxv;
  uint32_t nonzero;
};

// Pointers, not references: the .ino prototype generator puts its declarations above the struct.
static void stats_add(struct slot_stats *sp, int32_t v) {
  struct slot_stats &s = *sp;
  s.sum += v;
  s.sumsq += (double)v * (double)v;
  if (v < s.minv) s.minv = v;
  if (v > s.maxv) s.maxv = v;
  if (v) s.nonzero++;
}

static void print_stats(const char *name, const struct slot_stats *sp, uint32_t n) {
  const struct slot_stats &s = *sp;
  const double fs = 8388608.0;   // 2^23: 24-bit full scale
  double mean = (double)s.sum / n;
  double rms = sqrt(s.sumsq / n - mean * mean) / fs;
  LINK.print(name);
  LINK.print(" nonzero ");  LINK.print(s.nonzero);
  LINK.print("/");          LINK.print(n);
  LINK.print(" min ");      LINK.print(s.minv);
  LINK.print(" max ");      LINK.print(s.maxv);
  LINK.print(" dc ");       LINK.print(mean / fs, 6);
  LINK.print(" ac_dbfs ");  LINK.println(rms > 0 ? 20.0 * log10(rms) : -999.0, 1);
}

static uint32_t cyc_per_s;

static void capture(bool send) {
  slot_stats l = {0, 0, INT32_MAX, INT32_MIN, 0}, r = {0, 0, INT32_MAX, INT32_MIN, 0};
  const uint32_t settle = (uint32_t)(rate_hz * SETTLE_MS / 1000.0f);

  sai_stop();
  uint32_t key = __get_PRIMASK();
  __disable_irq();
  sai_start();
  for (uint32_t i = 0; i < settle; i++) { sai_read(); sai_read(); }
  uint32_t t0 = DWT->CYCCNT;
  for (uint32_t i = 0; i < CAP_FRAMES; i++) {
    int32_t lv = (int32_t)sai_read() >> 8;
    int32_t rv = (int32_t)sai_read() >> 8;
    cap[i] = lv << 8;
    stats_add(&l, lv);
    stats_add(&r, rv);
  }
  uint32_t t1 = DWT->CYCCNT;
  uint32_t sr = SAI1_Block_A->SR;
  __set_PRIMASK(key);
  // Keep the clocks running after the capture (docs/i2s_mic_trial.md: do not stop SCK).
  // The FIFO overruns while nobody reads it; the next capture restarts the block anyway.

  double secs = (double)(t1 - t0) / cyc_per_s;
  LINK.print("CAP clock ");    LINK.print(clk_name);
  LINK.print(" nominal_hz ");  LINK.print(rate_hz, 2);
  LINK.print(" measured_hz "); LINK.print(CAP_FRAMES / secs, 1);
  LINK.print(" frames ");      LINK.print(CAP_FRAMES);
  LINK.print(" overrun ");     LINK.println((sr & SAI_xSR_OVRUDR) ? 1 : 0);
  print_stats("L", &l, CAP_FRAMES);
  print_stats("R", &r, CAP_FRAMES);
  if (send) {
    LINK.print("DATA ");
    LINK.println((unsigned long)sizeof(cap));
    LINK.write((const uint8_t *)cap, sizeof(cap));
    LINK.println();
  }
  LINK.println("END");
}

static void info() {
  LINK.print("INFO cyc_per_s ");  LINK.print(cyc_per_s);
  LINK.print(" clock ");          LINK.print(clk_name);
  LINK.print(" nominal_hz ");     LINK.println(rate_hz, 2);
  LINK.print("RCC CR ");       LINK.print(RCC->CR, HEX);
  LINK.print(" CFGR1 ");       LINK.print(RCC->CFGR1, HEX);
  LINK.print(" PLL1CFGR ");    LINK.print(RCC->PLL1CFGR, HEX);
  LINK.print(" PLL3CFGR ");    LINK.print(RCC->PLL3CFGR, HEX);
  LINK.print(" PLL3DIVR ");    LINK.print(RCC->PLL3DIVR, HEX);
  LINK.print(" CCIPR2 ");      LINK.println(RCC->CCIPR2, HEX);
  LINK.print("SAI CR1 ");      LINK.print(SAI1_Block_A->CR1, HEX);
  LINK.print(" FRCR ");        LINK.print(SAI1_Block_A->FRCR, HEX);
  LINK.print(" SLOTR ");       LINK.print(SAI1_Block_A->SLOTR, HEX);
  LINK.print(" SR ");          LINK.println(SAI1_Block_A->SR, HEX);
  LINK.print("GPIOE MODER ");  LINK.print(GPIOE->MODER, HEX);
  LINK.print(" AFRL ");        LINK.print(GPIOE->AFR[0], HEX);
  LINK.print(" IDR ");         LINK.println(GPIOE->IDR, HEX);
  LINK.println("END");
}

void setup() {
  LINK.begin(LINK_BAUD);
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  uint32_t c0 = DWT->CYCCNT;
  delay(200);
  cyc_per_s = (DWT->CYCCNT - c0) * 5u;

  pins_af13();
  sai_setup();
  sai_start();
  LINK.println("READY sai1_mic_test");
}

void loop() {
  if (!LINK.available()) { delay(1); return; }
  switch (LINK.read()) {
    case 'i': info(); break;
    case 'c': capture(true); break;
    case 'h': capture(false); break;
  }
}
