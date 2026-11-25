// UNO Q MCU firmware of Yamabiko No.1: the microphone stream of mcu/yamabiko_link plus the IO of the
// A board (hardware/yamabiko_a), driven from the Linux side over the same link.
//
// Audio: unchanged from yamabiko_link (SAI1 -> GPDMA ring -> FIR, decimate to 16 kHz -> 5 ms packets).
//
// IO (pins as in the IO spec; the A board is an UNO shield):
//   D9 / D10  PB8 / PB9  TIM4 CH3 / CH4  TB6643KQ IN1 / IN2, 20 kHz. pwm > 0: IN1 high, IN2 high for
//                                        1 - duty (drive / brake); pwm < 0 the other way round;
//                                        0 = both low (coast); brake = both high
//   A0        PA4        ADC            current sense: 0.1 ohm, RC filtered, reported in mV
//   D5        PA11       TIM1 CH4       fan PWM, 25 kHz. An N-MOSFET pulls the fan's PWM line low, so the
//                                        output is inverted (CC4P): the duty set is the fan's duty
//   D6        PB1        GPIO           fan 12 V high-side switch
//   D2        PB3        EXTI           fan tach, 2 pulses per turn
//   D1        PB6        TIM16 CH1N     SG90 valve servo, 50 Hz, pulse width in us, through the 74HC125
//   D7        PB2        GPIO           the buffer's enable, active low: held low, the buffer always passes
//                                        D1 to J4 (pulled up while booting, so the servo line idles then)
//   D3        PB0        GPIO           backup solenoid valve (not fitted): follows the valve command
//   D8        PB4        GPIO           EXEC button, active low
//   PF14/PF15 J15-16/18  I2C4           the touch panel (M5Stack Tab5) on the B board's J3, as an I2C target
//                                        at PANEL_ADDR (see "Panel" below)
//
// The valve is the rotary disc on an SG90 (hobby PWM servo, no read-back): the Linux side sends open /
// closed, the pulse widths are set with 'K' and the pulse moves from one to the other over move_ms.
// A command that does not come again within CMD_TIMEOUT_MS stops the actuator (coast), shuts the valve
// and stops the fan: a crashed controller must not leave the plunger driving or the air blowing.
//
// Link protocol, little endian. MCU -> Linux, every packet:
//   A5 5A type len(u16) payload[len] sum(u8)       sum = 8-bit sum of type, len and payload
//   type 'A' audio:  seq(u16) first(u32, 16 kHz sample index) samples(i16 x n)
//   type 'P' pong:   token(u32) frame48(u32) overruns(u32) max_loop_us(u32)
//   type 'Y' status, every 10 ms:
//                    ms(u32) frame48(u32) isense_mv(u16, mean of the last 10 ms) isense_peak_mv(u16)
//                    fan_rpm(u16) buttons(u8, presses so far, wraps) flags(u8) pwm(i16, applied, -1000..1000)
//                    servo_pos(i16, the servo pulse now [us]) cmd_age_ms(u16)
//                    flags: 1 button held, 2 valve open, 4 fan on, 8 command timed out (16 unused: was the
//                    SCS0009 not answering)
//   type 'T' text:   one line (info)
// Linux -> MCU, every command framed as  A5 cmd payload sum(u8)   sum = 8-bit sum of cmd and payload.
//   Bytes outside a frame and frames with a bad sum are dropped (counted as cmd_bad in INFO): unframed, one
//   lost byte made a data byte a command (pwm 600 is 58 02, and 0x58 is 'X', which stopped the audio).
//   'S' start streaming, 'X' stop, 'i' info, 'P' + token(u32): ping, answered at once
//   'C' + pwm(i16, -1000..1000) + flags(u8: 1 valve open, 2 brake)   actuator and valve, every control step
//   'F' + on(u8) + duty(u16, 0..1000)                                 fan
//   'K' + open(u16) + closed(u16) + move_ms(u16)                      servo pulses of the valve [us, 500..2500]
//   'L' + max_pwm(u16, 0..1000)                                       limit of |pwm|
//   'U' + offset(u16) + data(32 bytes)                                 a piece of the next panel page
//   'V' + length(u16) + version(u16)                                   the panel page is complete: show it
// MCU -> Linux also:
//   type 'U' panel:  what the panel asked for (the bytes it wrote after register 0x02)
//
// Panel: the Tab5 is the I2C controller, this MCU the target at PANEL_ADDR on I2C4, remapped to PF14 / PF15
// (the core's I2C4 pins are PD12 / PD13, and its Zephyr build has no I2C target support, so the registers
// are served from loop(), a step per pass, with clock stretching holding the bus between the steps).  The Tab5 writes
//   0x01 n           n presses of the EXEC button (1 or 2, 150 ms apart): record / end / stop, 2 = again
//   0x02 bytes...    an event for the Linux side (choose a step, calibrate, play an archive clip), sent on
//   0x03 lo hi       where the next read starts
// and reads the page: 'Y' '1' version(u16) length(u16), then `length` bytes the Linux side made
// (flute_rl/yamabiko/panel.py).  Reads past the end give 0xFF.  Two buffers: Linux fills the one not shown,
// 'V' shows it, so a page is never half new (the Tab5 checks the version before and after its reads).
#include <Arduino.h>
#include <stm32u5xx.h>

#define LINK Serial1
#define LINK_BAUD 921600
#define CMD_SYNC 0xA5

#define RING_FRAMES 2048          // 42.7 ms of stereo 32-bit words at 48 kHz = 16 KB
#define DECIM 3                   // 48 kHz -> 16 kHz
#define PKT_SAMPLES 80            // 5 ms at 16 kHz
#define NTAPS 31                  // anti-alias FIR at 48 kHz, cut-off 7 kHz
#define DMA_CH GPDMA1_Channel7
#define DMA_REQ_SAI1_A 36u        // GPDMA1_REQUEST_SAI1_A (stm32u5xx_hal_dma.h)

#define PIN_TACH 2
#define PIN_VALVE 3
#define PIN_FAN_EN 6
#define PIN_SERVO_TXEN_N 7
#define PIN_BTN 8
#define ACT_HZ 20000u
#define FAN_HZ 25000u
#define CMD_TIMEOUT_MS 100u
#define SERVO_US_MIN 500u
#define SERVO_US_MAX 2500u
#define STATUS_MS 10u
#define SENSE_MV_FULL 3300u       // ADC full scale
#define PANEL_ADDR 0x42           // I2C target address of the panel link
#define PANEL_MAX 3072u           // bytes of one panel page
#define PANEL_CHUNK 32u           // bytes in a 'U' command
#define PANEL_PRESS_GAP_MS 150u   // between the two presses of a double press from the panel

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
static uint32_t cmd_bad;          // command bytes dropped: outside a frame, or in a frame with a bad sum
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

// ---- IO of the A board -------------------------------------------------------------

static uint32_t act_top, fan_top;            // timer periods [ticks]
static int16_t act_pwm;                      // applied, -1000..1000
static bool act_brake, valve_open, fan_on, cmd_timed_out = true;
static uint16_t max_pwm = 1000, fan_duty;
static uint32_t last_cmd_ms;
static uint16_t valve_pos_open = 1800, valve_pos_closed = 1200, valve_move_ms = 150;   // [us], [ms]

static volatile uint32_t tach_pulses;
static uint32_t tach_last, rpm_ms;
static uint16_t fan_rpm;

static uint32_t sense_sum, sense_n, sense_peak;
static uint16_t sense_mv, sense_peak_mv;

static uint8_t btn_presses, btn_stable = 1, btn_run;
static uint8_t panel_page[2][PANEL_MAX];
static uint16_t panel_len[2], panel_ver[2];
static uint8_t panel_cur;                     // the page the Tab5 reads
static uint16_t panel_ptr;                    // where its next read starts
static uint8_t panel_presses_left;
static uint32_t panel_press_at;
static uint32_t panel_xfers, panel_errors;
static uint32_t presses_board, presses_panel, panel_events_sent;   // where the presses came from
static bool panel_ok;
static uint32_t io_ms, status_ms;

static float servo_us, servo_goal_us, servo_step_us;   // pulse now, where it goes, change per ms

static void pin_af(GPIO_TypeDef *g, uint32_t pin, uint32_t af) {
  g->MODER = (g->MODER & ~(3u << (2 * pin))) | (2u << (2 * pin));
  g->OSPEEDR = (g->OSPEEDR & ~(3u << (2 * pin))) | (1u << (2 * pin));
  g->AFR[pin >> 3] = (g->AFR[pin >> 3] & ~(0xFu << (4 * (pin & 7)))) | (af << (4 * (pin & 7)));
}

// Timer kernel clock: HCLK when the APB prescaler is 1, otherwise twice the APB clock.
static uint32_t timer_clk(bool apb2) {
  uint32_t hclk = cyc_per_us * 1000000u;
  uint32_t ppre = apb2 ? (RCC->CFGR2 & RCC_CFGR2_PPRE2) >> RCC_CFGR2_PPRE2_Pos
                       : (RCC->CFGR2 & RCC_CFGR2_PPRE1) >> RCC_CFGR2_PPRE1_Pos;
  if (ppre < 4) return hclk;
  return 2u * hclk / (1u << (ppre - 3));
}

static void act_setup() {
  RCC->AHB2ENR1 |= RCC_AHB2ENR1_GPIOBEN;
  RCC->APB1ENR1 |= RCC_APB1ENR1_TIM4EN;
  (void)RCC->APB1ENR1;
  TIM_TypeDef *t = TIM4;
  t->CR1 = 0;
  act_top = timer_clk(false) / ACT_HZ;
  t->PSC = 0;
  t->ARR = act_top - 1;
  t->CCR3 = 0;
  t->CCR4 = 0;
  t->CCMR2 = (6u << TIM_CCMR2_OC3M_Pos) | TIM_CCMR2_OC3PE | (6u << TIM_CCMR2_OC4M_Pos) | TIM_CCMR2_OC4PE;
  t->CCER = TIM_CCER_CC3E | TIM_CCER_CC4E;
  t->EGR = TIM_EGR_UG;
  t->CR1 = TIM_CR1_ARPE | TIM_CR1_CEN;
  pin_af(GPIOB, 8, 2);                        // TIM4_CH3
  pin_af(GPIOB, 9, 2);                        // TIM4_CH4
}

static void act_set(int32_t pwm, bool brake) {
  if (pwm > max_pwm) pwm = max_pwm;
  if (pwm < -(int32_t)max_pwm) pwm = -(int32_t)max_pwm;
  act_pwm = (int16_t)pwm;
  act_brake = brake;
  // Drive / brake (slow decay): one input stays high, the other is high for the off part of the period,
  // so the motor is shorted between the pulses and its mean voltage is duty x 12 V. Drive / coast (the
  // other input low) let the current flow back through the body diodes against the supply: with the
  // motor's inductance that gave (2 duty - 1) x 12 V, 2.4 V at 0.6, and too little force to move it.
  if (brake) {                                // both inputs high: the outputs are shorted
    TIM4->CCR3 = act_top;
    TIM4->CCR4 = act_top;
  } else if (pwm == 0) {                      // both low: outputs off (coast)
    TIM4->CCR3 = 0;
    TIM4->CCR4 = 0;
  } else if (pwm > 0) {
    TIM4->CCR3 = act_top;
    TIM4->CCR4 = (uint32_t)(1000 - pwm) * act_top / 1000u;
  } else {
    TIM4->CCR4 = act_top;
    TIM4->CCR3 = (uint32_t)(1000 + pwm) * act_top / 1000u;
  }
}

static void fan_setup() {
  RCC->AHB2ENR1 |= RCC_AHB2ENR1_GPIOAEN;
  RCC->APB2ENR |= RCC_APB2ENR_TIM1EN;
  (void)RCC->APB2ENR;
  TIM_TypeDef *t = TIM1;
  t->CR1 = 0;
  fan_top = timer_clk(true) / FAN_HZ;
  t->PSC = 0;
  t->ARR = fan_top - 1;
  t->CCR4 = 0;
  t->CCMR2 = (6u << TIM_CCMR2_OC4M_Pos) | TIM_CCMR2_OC4PE;
  t->CCER = TIM_CCER_CC4E | TIM_CCER_CC4P;    // inverted: low for `duty`, so the fan's line is high
  t->BDTR = TIM_BDTR_MOE;
  t->EGR = TIM_EGR_UG;
  t->CR1 = TIM_CR1_ARPE | TIM_CR1_CEN;
  pin_af(GPIOA, 11, 1);                       // TIM1_CH4
}

static void fan_set(bool on, uint16_t duty) {
  if (duty > 1000) duty = 1000;
  fan_on = on;
  fan_duty = duty;
  TIM1->CCR4 = (uint32_t)duty * fan_top / 1000u;
  digitalWrite(PIN_FAN_EN, on ? HIGH : LOW);
}

static void tach_isr() { tach_pulses++; }

// ---- SG90 on TIM16 CH1N (PB6, AF14): 1 MHz count, 20 ms period, the compare is the pulse in us ----

static void servo_setup() {
  RCC->AHB2ENR1 |= RCC_AHB2ENR1_GPIOBEN;
  RCC->APB2ENR |= RCC_APB2ENR_TIM16EN;
  (void)RCC->APB2ENR;
  TIM_TypeDef *t = TIM16;
  t->CR1 = 0;
  t->DIER = 0;                                // the core's counter driver must not get interrupts
  t->PSC = timer_clk(true) / 1000000u - 1u;
  t->ARR = 20000u - 1u;
  t->CCR1 = 0;                                // no pulse until the first goal
  t->CCMR1 = (6u << TIM_CCMR1_OC1M_Pos) | TIM_CCMR1_OC1PE;
  t->CCER = TIM_CCER_CC1NE;                   // only the N output: with CC1E off it follows OC1REF
  t->BDTR = TIM_BDTR_MOE;
  t->EGR = TIM_EGR_UG;
  t->CR1 = TIM_CR1_ARPE | TIM_CR1_CEN;
  pin_af(GPIOB, 6, 14);                       // TIM16_CH1N
}

static uint16_t servo_clamp(uint16_t us) {
  return us < SERVO_US_MIN ? SERVO_US_MIN : (us > SERVO_US_MAX ? SERVO_US_MAX : us);
}

static void servo_goal(uint16_t us, uint16_t ms) {   // move linearly to `us` over `ms`
  servo_goal_us = servo_clamp(us);
  if (servo_us == 0.0f || ms == 0) {
    servo_us = servo_goal_us;
    servo_step_us = 0.0f;
  } else {
    float d = servo_goal_us - servo_us;
    servo_step_us = (d < 0 ? -d : d) / ms;
  }
  TIM16->CCR1 = (uint32_t)(servo_us + 0.5f);
}

static void servo_tick() {                    // every millisecond
  if (servo_us == servo_goal_us) return;
  float d = servo_goal_us - servo_us;
  if (servo_step_us <= 0.0f || (d < 0 ? -d : d) <= servo_step_us) servo_us = servo_goal_us;
  else servo_us += d < 0 ? -servo_step_us : servo_step_us;
  TIM16->CCR1 = (uint32_t)(servo_us + 0.5f);
}

static void valve_set(bool open) {
  if (open == valve_open) return;
  valve_open = open;
  digitalWrite(PIN_VALVE, open ? HIGH : LOW);
  servo_goal(open ? valve_pos_open : valve_pos_closed, valve_move_ms);
}

static void io_setup() {
  pinMode(PIN_SERVO_TXEN_N, OUTPUT);
  digitalWrite(PIN_SERVO_TXEN_N, LOW);          // the buffer passes D1 to the servo all the time
  pinMode(PIN_FAN_EN, OUTPUT);
  digitalWrite(PIN_FAN_EN, LOW);
  pinMode(PIN_VALVE, OUTPUT);
  digitalWrite(PIN_VALVE, LOW);
  pinMode(PIN_BTN, INPUT_PULLUP);
  pinMode(PIN_TACH, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_TACH), tach_isr, FALLING);
  analogReadResolution(12);
  act_setup();
  fan_setup();
  act_set(0, false);
  fan_set(false, 0);
  servo_setup();
  servo_goal(valve_pos_closed, 0);
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

struct __attribute__((packed)) Status {
  uint32_t ms, frame48;
  uint16_t isense_mv, isense_peak_mv, fan_rpm;
  uint8_t buttons, flags;
  int16_t pwm, servo_pos;
  uint16_t cmd_age_ms;
};

static void send_status(uint32_t now) {
  Status s;
  s.ms = now;
  s.frame48 = frames_now();
  s.isense_mv = sense_mv;
  s.isense_peak_mv = sense_peak_mv;
  s.fan_rpm = fan_rpm;
  s.buttons = btn_presses;
  s.flags = (btn_stable ? 0 : 1) | (valve_open ? 2 : 0) | (fan_on ? 4 : 0) | (cmd_timed_out ? 8 : 0);
  s.pwm = act_brake ? 0 : act_pwm;
  s.servo_pos = (int16_t)(servo_us + 0.5f);
  uint32_t age = now - last_cmd_ms;
  s.cmd_age_ms = age > 65535u ? 65535u : (uint16_t)age;
  send_packet('Y', (const uint8_t *)&s, sizeof(s), nullptr, 0);
}

static void info() {
  char s[200];
  snprintf(s, sizeof(s), "INFO fw yamabiko_fw clock %s cyc_per_us %lu streaming %d frames %lu overruns %lu max_loop_us %lu cmd_bad %lu",
           clk_name, (unsigned long)cyc_per_us, streaming, (unsigned long)frames_now(),
           (unsigned long)overruns, (unsigned long)max_loop_us, (unsigned long)cmd_bad);
  send_text(s);
  snprintf(s, sizeof(s), "IO act_top %lu (%lu Hz) fan_top %lu (%lu Hz) max_pwm %u valve open %u closed %u us move %u ms servo SG90 %d us",
           (unsigned long)act_top, (unsigned long)(timer_clk(false) / act_top), (unsigned long)fan_top,
           (unsigned long)(timer_clk(true) / fan_top), max_pwm, valve_pos_open, valve_pos_closed, valve_move_ms,
           (int)(servo_us + 0.5f));
  send_text(s);
  snprintf(s, sizeof(s), "PANEL i2c4 %s addr 0x%02x timingr 0x%08lx transfers %lu errors %lu page v%u %u bytes "
           "scl %d sda %d isr 0x%05lx presses board %lu panel %lu events %lu",
           panel_ok ? "PF14/PF15" : "off", PANEL_ADDR, (unsigned long)I2C4->TIMINGR, (unsigned long)panel_xfers,
           (unsigned long)panel_errors, panel_ver[panel_cur], panel_len[panel_cur],
           (int)((GPIOF->IDR >> 14) & 1), (int)((GPIOF->IDR >> 15) & 1), (unsigned long)(I2C4->ISR & 0xFFFFFu),
           (unsigned long)presses_board, (unsigned long)presses_panel, (unsigned long)panel_events_sent);
  send_text(s);
}

// Bytes of a command after the sync byte, without the sum (0: not a command).
static uint8_t cmd_len(uint8_t c) {
  switch (c) {
    case 'P': return 5;
    case 'C': return 4;
    case 'F': return 4;
    case 'K': return 7;
    case 'L': return 3;
    case 'U': return 3 + PANEL_CHUNK;
    case 'V': return 5;
    case 'S': case 'X': case 'i': return 1;
    default: return 0;
  }
}

static void run_cmd(const uint8_t *b) {
  uint8_t c = b[0];
  if (c == 'P') {
    pong_frame = frames_now();
    memcpy(&pong_token, b + 1, 4);
    send_pong();                                 // between packets, never inside one
  } else if (c == 'C') {
    int16_t pwm;
    memcpy(&pwm, b + 1, 2);
    act_set(pwm, (b[3] & 2) != 0);
    valve_set((b[3] & 1) != 0);
    last_cmd_ms = millis();
    cmd_timed_out = false;
  } else if (c == 'F') {
    uint16_t d;
    memcpy(&d, b + 2, 2);
    fan_set(b[1] != 0, d);
  } else if (c == 'K') {
    memcpy(&valve_pos_open, b + 1, 2);
    memcpy(&valve_pos_closed, b + 3, 2);
    memcpy(&valve_move_ms, b + 5, 2);
    valve_pos_open = servo_clamp(valve_pos_open);
    valve_pos_closed = servo_clamp(valve_pos_closed);
    servo_goal(valve_open ? valve_pos_open : valve_pos_closed, 300);
  } else if (c == 'L') {
    uint16_t m;
    memcpy(&m, b + 1, 2);
    max_pwm = m > 1000 ? 1000 : m;
    act_set(act_pwm, act_brake);
  } else if (c == 'S') { stream_restart(); streaming = true; }
  else if (c == 'X') streaming = false;
  else if (c == 'i') info();
  else if (c == 'U') {                           // into the page not shown
    uint16_t off;
    memcpy(&off, b + 1, 2);
    uint8_t *page = panel_page[panel_cur ^ 1];
    for (uint32_t i = 0; i < PANEL_CHUNK && off + i < PANEL_MAX; i++) page[off + i] = b[3 + i];
  } else if (c == 'V') {                         // show it
    uint8_t next = panel_cur ^ 1;
    memcpy(&panel_len[next], b + 1, 2);
    memcpy(&panel_ver[next], b + 3, 2);
    if (panel_len[next] > PANEL_MAX) panel_len[next] = PANEL_MAX;
    panel_cur = next;
  }
}

static uint8_t rx[8 + PANEL_CHUNK];              // sync, command, payload, sum
static uint8_t rx_n;

static void commands() {
  while (LINK.available()) {
    uint8_t c = (uint8_t)LINK.read();
    if (rx_n == 0) {
      if (c == CMD_SYNC) rx[rx_n++] = c;
      else cmd_bad++;
      continue;
    }
    rx[rx_n++] = c;
    uint8_t need = cmd_len(rx[1]);
    if (need == 0) {                             // no command after the sync byte: hunt again
      cmd_bad++;
      rx_n = (c == CMD_SYNC) ? 1 : 0;
      continue;
    }
    if (rx_n < need + 2) continue;
    uint8_t sum = 0;
    for (uint8_t i = 1; i <= need; i++) sum += rx[i];
    rx_n = 0;
    if (sum == rx[need + 1]) run_cmd(rx + 1);
    else cmd_bad++;
  }
}

// Every millisecond: current sense, button, servo ramp. Every 10 ms: status. Command watchdog, fan speed.
static void io_tick() {
  uint32_t now = millis();
  if (!cmd_timed_out && now - last_cmd_ms > CMD_TIMEOUT_MS) {
    act_set(0, false);                         // the controller went quiet: stop and shut the air
    valve_set(false);
    fan_set(false, 0);
    cmd_timed_out = true;
  }
  if (now == io_ms) return;
  io_ms = now;
  servo_tick();
  uint32_t mv = (uint32_t)analogRead(A0) * SENSE_MV_FULL / 4095u;
  sense_sum += mv;
  sense_n++;
  if (mv > sense_peak) sense_peak = mv;
  uint8_t level = digitalRead(PIN_BTN) ? 1 : 0;  // debounced: 20 ms the same
  if (level != btn_stable) {
    if (++btn_run >= 20) {
      btn_stable = level;
      btn_run = 0;
      if (!level) { btn_presses++; presses_board++; }
    }
  } else {
    btn_run = 0;
  }
  if (panel_presses_left && (int32_t)(now - panel_press_at) >= 0) {   // the panel's presses count as the button's
    btn_presses++;
    presses_panel++;
    panel_presses_left--;
    panel_press_at = now + PANEL_PRESS_GAP_MS;
  }
  if (now - rpm_ms >= 500u) {                  // 2 pulses per turn: pulses in 0.5 s x 60 = rpm
    uint32_t p = tach_pulses;
    fan_rpm = (uint16_t)((p - tach_last) * 60u);
    tach_last = p;
    rpm_ms = now;
  }
  if (now - status_ms >= STATUS_MS) {
    status_ms = now;
    sense_mv = (uint16_t)(sense_n ? sense_sum / sense_n : 0);
    sense_peak_mv = (uint16_t)sense_peak;
    sense_sum = sense_n = sense_peak = 0;
    send_status(now);
  }
}
// ---- panel (I2C4 target, polled) -----------------------------------------------------

static void panel_setup() {
  RCC->APB1ENR2 |= RCC_APB1ENR2_I2C4EN;
  RCC->AHB2ENR1 |= RCC_AHB2ENR1_GPIOFEN | RCC_AHB2ENR1_GPIODEN;
  (void)RCC->AHB2ENR1;
  GPIOD->MODER |= (3u << (2 * 12)) | (3u << (2 * 13));      // the core's I2C4 pins let go (analog)
  for (uint32_t pin = 14; pin <= 15; pin++) {               // PF14 SCL, PF15 SDA: AF4, open drain, pull-up
    GPIOF->OTYPER |= 1u << pin;                              // (the B board has 4.7 k to 3.3 V as well)
    GPIOF->PUPDR = (GPIOF->PUPDR & ~(3u << (2 * pin))) | (1u << (2 * pin));
    pin_af(GPIOF, pin, 4);
  }
  I2C4->CR1 &= ~I2C_CR1_PE;
  while (I2C4->CR1 & I2C_CR1_PE) {}
  I2C4->CR1 &= ~(I2C_CR1_NOSTRETCH | I2C_CR1_GCEN | I2C_CR1_SBC | I2C_CR1_TXIE | I2C_CR1_RXIE | I2C_CR1_ADDRIE |
                 I2C_CR1_NACKIE | I2C_CR1_STOPIE | I2C_CR1_TCIE | I2C_CR1_ERRIE);   // polled: no interrupts
  if (I2C4->TIMINGR == 0) I2C4->TIMINGR = 0x00F07BFFu;      // only the data setup / hold matter for a target
  I2C4->OAR1 = 0;
  I2C4->OAR2 = 0;
  I2C4->OAR1 = I2C_OAR1_OA1EN | ((uint32_t)PANEL_ADDR << 1);
  I2C4->CR1 |= I2C_CR1_PE;
  panel_ok = true;
}

static uint8_t panel_byte(uint16_t i) {
  uint8_t c = panel_cur;
  if (i == 0) return 'Y';
  if (i == 1) return '1';
  if (i == 2) return panel_ver[c] & 0xFF;
  if (i == 3) return panel_ver[c] >> 8;
  if (i == 4) return panel_len[c] & 0xFF;
  if (i == 5) return panel_len[c] >> 8;
  return (uint16_t)(i - 6) < panel_len[c] ? panel_page[c][i - 6] : 0xFF;
}

static void panel_written(const uint8_t *b, uint8_t n) {
  if (n == 0) return;
  if (b[0] == 0x01 && n >= 2) {                              // presses of the EXEC button
    uint8_t k = b[1] < 1 ? 1 : (b[1] > 2 ? 2 : b[1]);
    if (!panel_presses_left) panel_press_at = millis();
    panel_presses_left = k;
  } else if (b[0] == 0x02 && n >= 2) {
    send_packet('U', b + 1, n - 1, nullptr, 0);
    panel_events_sent++;
  } else if (b[0] == 0x03 && n >= 3) {
    panel_ptr = (uint16_t)(b[1] | (b[2] << 8));
  }
}

// A step of the transfer in progress, every pass of loop(): never waits.  Between bytes the controller is held
// by clock stretching until the next pass (tens of microseconds; a pass with audio to filter takes up to ~2 ms).
// A transfer that makes no progress for PANEL_STALL_MS (the controller gave up) is dropped and the peripheral
// reset.  (The first version served a whole transfer in a loop of its own and held loop() up to 30 ms when the
// Tab5 gave up, which lost link bytes.)
#define PANEL_STALL_MS 20u
static uint8_t panel_w[40];
static uint8_t panel_wn;
static bool panel_active, panel_reading;
static uint32_t panel_seen_ms;

static void panel_reset() {
  I2C4->CR1 &= ~I2C_CR1_PE;
  while (I2C4->CR1 & I2C_CR1_PE) {}
  I2C4->CR1 |= I2C_CR1_PE;
  panel_active = false;
  panel_wn = 0;
  panel_errors++;
}

static void panel_poll() {
  if (!panel_ok) return;
  for (int k = 0; k < 4; k++) {                              // a few events per pass (stop, address, a byte)
    uint32_t isr = I2C4->ISR;
    // A byte received first: its stop can already be pending.  Taken after the stop, the last byte of
    // "0x03 lo hi" stayed in RXDR and became the first byte of the next write: hi 0x01 / 0x02 read as a press /
    // an event (the panel "pressed" by itself once the page passed 262 bytes).
    if (panel_active && !panel_reading && (isr & I2C_ISR_RXNE)) {
      uint8_t v = (uint8_t)I2C4->RXDR;
      if (panel_wn < sizeof(panel_w)) panel_w[panel_wn++] = v;
      panel_seen_ms = millis();
      continue;
    }
    // Then the stop of the last transfer: a stop and the next start can both be pending, and taking the start
    // first then the stop dropped the new transfer with SCL held low (the bus hung).
    if (isr & I2C_ISR_STOPF) {
      I2C4->ICR = I2C_ICR_STOPCF | I2C_ICR_NACKCF;
      if (panel_active && !panel_reading && panel_wn) panel_written(panel_w, panel_wn);
      if (panel_active) panel_xfers++;
      panel_wn = 0;
      panel_active = false;
      continue;
    }
    if (isr & I2C_ISR_ADDR) {                                // start, or a repeated start
      if (panel_active && !panel_reading && panel_wn) panel_written(panel_w, panel_wn);
      panel_wn = 0;
      panel_reading = (isr & I2C_ISR_DIR) != 0;
      if (panel_reading) I2C4->ISR |= I2C_ISR_TXE;           // drop a byte left from an earlier read
      I2C4->ICR = I2C_ICR_ADDRCF;
      panel_active = true;
      panel_seen_ms = millis();
      continue;
    }
    if (!panel_active) {
      if (isr & I2C_ISR_RXNE) (void)I2C4->RXDR;            // a byte from no transfer of ours: drop it
      if (isr & (I2C_ISR_NACKF | I2C_ISR_BERR | I2C_ISR_OVR))
        I2C4->ICR = I2C_ICR_NACKCF | I2C_ICR_BERRCF | I2C_ICR_OVRCF;
      if (!(isr & I2C_ISR_BUSY)) panel_seen_ms = millis();
      else if (millis() - panel_seen_ms > PANEL_STALL_MS) panel_reset();   // busy with no transfer of ours
      return;
    }
    bool moved = false;
    if (!panel_reading && (isr & I2C_ISR_RXNE)) {
      uint8_t v = (uint8_t)I2C4->RXDR;
      if (panel_wn < sizeof(panel_w)) panel_w[panel_wn++] = v;
      moved = true;
    }
    if (panel_reading && (isr & I2C_ISR_TXIS)) { I2C4->TXDR = panel_byte(panel_ptr++); moved = true; }
    if (isr & I2C_ISR_NACKF) I2C4->ICR = I2C_ICR_NACKCF;
    if (isr & (I2C_ISR_BERR | I2C_ISR_OVR)) I2C4->ICR = I2C_ICR_BERRCF | I2C_ICR_OVRCF;
    if (moved) { panel_seen_ms = millis(); continue; }
    if (millis() - panel_seen_ms > PANEL_STALL_MS) panel_reset();   // the controller went away: start over
    return;
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
  // Word counts wrap at 2^32 (12.4 h at 96 k words/s): compare the difference, never the counts.
  // `done + 2 <= total` turned true for about 2^31 frames when `done` wrapped first, and the loop
  // then ran for hours without serving commands or sending status.
  while (total - done >= 2) {
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
  cyc_per_us = (DWT->CYCCNT - c0 + 50000u) / 100000u;

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

  io_setup();
  panel_setup();
  sai_setup();
  stream_restart();
  streaming = false;
  send_text("READY yamabiko_fw");
}

void loop() {
  uint32_t t0 = DWT->CYCCNT;
  commands();
  if (streaming) {
    pump();
  } else {
    dma_total_words();                        // keep counting passes while idle
  }
  io_tick();
  panel_poll();
  uint32_t us = (DWT->CYCCNT - t0) / cyc_per_us;
  if (us > max_loop_us) max_loop_us = us;
}
