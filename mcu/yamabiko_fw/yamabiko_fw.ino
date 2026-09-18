// UNO Q MCU firmware of Yamabiko No.1: the microphone stream of mcu/yamabiko_link plus the IO of the
// A board (hardware/yamabiko_a), driven from the Linux side over the same link.
//
// Audio: unchanged from yamabiko_link (SAI1 -> GPDMA ring -> FIR, decimate to 16 kHz -> 5 ms packets).
//
// IO (pins as in the IO spec; the A board is an UNO shield):
//   D9 / D10  PB8 / PB9  TIM4 CH3 / CH4  TB6643KQ IN1 / IN2, 20 kHz. pwm > 0 drives IN1, pwm < 0 IN2,
//                                        the other input low (drive / coast); brake = both high
//   A0        PA4        ADC            current sense: 0.1 ohm, RC filtered, reported in mV
//   D5        PA11       TIM1 CH4       fan PWM, 25 kHz. An N-MOSFET pulls the fan's PWM line low, so the
//                                        output is inverted (CC4P): the duty set is the fan's duty
//   D6        PB1        GPIO           fan 12 V high-side switch
//   D2        PB3        EXTI           fan tach, 2 pulses per turn
//   D0 / D1   usart1     Serial         SCS0009 (1 Mbaud, half duplex through a 74HC125);
//   D7        PB2        GPIO           the buffer's transmit enable, active low (pulled up while booting)
//   D3        PB0        GPIO           backup solenoid valve (not fitted): follows the valve command
//   D8        PB4        GPIO           EXEC button, active low
//
// The valve is the rotary disc on the SCS0009: the Linux side sends open / closed, the positions are
// set once with 'K'. A command that does not come again within CMD_TIMEOUT_MS stops the actuator
// (coast): a crashed controller must not leave the plunger driving.
//
// Link protocol, little endian. MCU -> Linux, every packet:
//   A5 5A type len(u16) payload[len] sum(u8)       sum = 8-bit sum of type, len and payload
//   type 'A' audio:  seq(u16) first(u32, 16 kHz sample index) samples(i16 x n)
//   type 'P' pong:   token(u32) frame48(u32) overruns(u32) max_loop_us(u32)
//   type 'Y' status, every 10 ms:
//                    ms(u32) frame48(u32) isense_mv(u16, mean of the last 10 ms) isense_peak_mv(u16)
//                    fan_rpm(u16) buttons(u8, presses so far, wraps) flags(u8) pwm(i16, applied, -1000..1000)
//                    servo_pos(i16, -1 = no answer yet) cmd_age_ms(u16)
//                    flags: 1 button held, 2 valve open, 4 fan on, 8 command timed out, 16 servo not answering
//   type 'T' text:   one line (info)
// Linux -> MCU:
//   'S' start streaming, 'X' stop, 'i' info, 'P' + token(u32): ping, answered at once
//   'C' + pwm(i16, -1000..1000) + flags(u8: 1 valve open, 2 brake)   actuator and valve, every control step
//   'F' + on(u8) + duty(u16, 0..1000)                                 fan
//   'K' + open(u16) + closed(u16) + move_ms(u16)                      servo positions of the valve (0..1023)
//   'L' + max_pwm(u16, 0..1000)                                       limit of |pwm|
#include <Arduino.h>
#include <stm32u5xx.h>

#define LINK Serial1
#define LINK_BAUD 921600
#define SERVO Serial
#define SERVO_BAUD 1000000
#define SERVO_ID 1

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
#define STATUS_MS 10u
#define SENSE_MV_FULL 3300u       // ADC full scale

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

// ---- IO of the A board -------------------------------------------------------------

static uint32_t act_top, fan_top;            // timer periods [ticks]
static int16_t act_pwm;                      // applied, -1000..1000
static bool act_brake, valve_open, fan_on, cmd_timed_out = true;
static uint16_t max_pwm = 1000, fan_duty;
static uint32_t last_cmd_ms;
static uint16_t valve_pos_open = 590, valve_pos_closed = 512, valve_move_ms = 50;

static volatile uint32_t tach_pulses;
static uint32_t tach_last, rpm_ms;
static uint16_t fan_rpm;

static uint32_t sense_sum, sense_n, sense_peak;
static uint16_t sense_mv, sense_peak_mv;

static uint8_t btn_presses, btn_stable = 1, btn_run;
static uint32_t io_ms, status_ms, servo_ms;

static uint16_t servo_echo;                  // bytes of our own transmission still to come back
static uint8_t srx[16];
static uint8_t srx_n;
static int16_t servo_pos = -1;
static uint32_t servo_ok_ms;
static bool servo_ok;

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
  if (brake) {                                // both inputs high: the outputs are shorted
    TIM4->CCR3 = act_top;
    TIM4->CCR4 = act_top;
  } else if (pwm >= 0) {
    TIM4->CCR4 = 0;
    TIM4->CCR3 = (uint32_t)pwm * act_top / 1000u;
  } else {
    TIM4->CCR3 = 0;
    TIM4->CCR4 = (uint32_t)(-pwm) * act_top / 1000u;
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

// ---- SCS0009 (Feetech SCS protocol, big endian registers) ----

static void servo_send(const uint8_t *body, uint8_t n) {   // body: ID LEN INST params, no header / checksum
  uint8_t b[24] = {0xFF, 0xFF};
  uint8_t sum = 0;
  for (uint8_t i = 0; i < n; i++) { b[2 + i] = body[i]; sum += body[i]; }
  b[2 + n] = (uint8_t)~sum;
  uint8_t len = n + 3;
  while (SERVO.available()) SERVO.read();     // stale bytes
  srx_n = 0;
  digitalWrite(PIN_SERVO_TXEN_N, LOW);
  SERVO.write(b, len);
  // The receive side hears the bus, our own bytes included: once they are all back, the last one has
  // left the shift register and the transmitter can be let go.
  uint32_t t0 = micros();
  uint8_t got = 0;
  while (got < len && micros() - t0 < 1000u) {
    if (SERVO.available()) { SERVO.read(); got++; }
  }
  if (got < len) delayMicroseconds(20);       // no echo (nothing connected): wait out the last byte
  digitalWrite(PIN_SERVO_TXEN_N, HIGH);
}

static void servo_torque(bool on) {
  const uint8_t body[] = {SERVO_ID, 4, 0x03, 0x28, (uint8_t)(on ? 1 : 0)};
  servo_send(body, sizeof(body));
}

static void servo_goal(uint16_t pos, uint16_t ms) {
  const uint8_t body[] = {SERVO_ID, 9, 0x03, 0x2A, (uint8_t)(pos >> 8), (uint8_t)pos,
                          (uint8_t)(ms >> 8), (uint8_t)ms, 0, 0};
  servo_send(body, sizeof(body));
}

static void servo_ask_position() {
  const uint8_t body[] = {SERVO_ID, 4, 0x02, 0x38, 2};
  servo_send(body, sizeof(body));
}

static void servo_poll() {                    // collect the answer to servo_ask_position
  while (SERVO.available() && srx_n < sizeof(srx)) {
    uint8_t c = SERVO.read();
    if (srx_n < 2 && c != 0xFF) { srx_n = 0; continue; }
    srx[srx_n++] = c;
    if (srx_n == 8) {                         // FF FF ID 04 ERR H L SUM
      uint8_t sum = srx[2] + srx[3] + srx[4] + srx[5] + srx[6];
      if (srx[2] == SERVO_ID && srx[3] == 4 && (uint8_t)~sum == srx[7]) {
        servo_pos = (int16_t)((srx[5] << 8) | srx[6]);
        servo_ok_ms = millis();
        servo_ok = true;
      }
      srx_n = 0;
    }
  }
  if (srx_n >= sizeof(srx)) srx_n = 0;
}

static void valve_set(bool open) {
  if (open == valve_open) return;
  valve_open = open;
  digitalWrite(PIN_VALVE, open ? HIGH : LOW);
  servo_goal(open ? valve_pos_open : valve_pos_closed, valve_move_ms);
}

static void io_setup() {
  pinMode(PIN_SERVO_TXEN_N, OUTPUT);
  digitalWrite(PIN_SERVO_TXEN_N, HIGH);
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
  SERVO.begin(SERVO_BAUD);
  servo_torque(true);
  servo_goal(valve_pos_closed, 200);
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
  s.flags = (btn_stable ? 0 : 1) | (valve_open ? 2 : 0) | (fan_on ? 4 : 0) | (cmd_timed_out ? 8 : 0)
          | (servo_ok ? 0 : 16);
  s.pwm = act_brake ? 0 : act_pwm;
  s.servo_pos = servo_pos;
  uint32_t age = now - last_cmd_ms;
  s.cmd_age_ms = age > 65535u ? 65535u : (uint16_t)age;
  send_packet('Y', (const uint8_t *)&s, sizeof(s), nullptr, 0);
}

static void info() {
  char s[200];
  snprintf(s, sizeof(s), "INFO fw yamabiko_fw clock %s cyc_per_us %lu streaming %d frames %lu overruns %lu max_loop_us %lu",
           clk_name, (unsigned long)cyc_per_us, streaming, (unsigned long)frames_now(),
           (unsigned long)overruns, (unsigned long)max_loop_us);
  send_text(s);
  snprintf(s, sizeof(s), "IO act_top %lu (%lu Hz) fan_top %lu (%lu Hz) max_pwm %u valve open %u closed %u move %u ms servo %s pos %d",
           (unsigned long)act_top, (unsigned long)(timer_clk(false) / act_top), (unsigned long)fan_top,
           (unsigned long)(timer_clk(true) / fan_top), max_pwm, valve_pos_open, valve_pos_closed, valve_move_ms,
           servo_ok ? "ok" : "silent", servo_pos);
  send_text(s);
}

static uint8_t cmd_len(int c) {
  switch (c) {
    case 'P': return 5;
    case 'C': return 4;
    case 'F': return 4;
    case 'K': return 7;
    case 'L': return 3;
    default: return 1;
  }
}

static void commands() {
  while (LINK.available()) {
    int c = LINK.peek();
    uint8_t need = cmd_len(c);
    if (LINK.available() < need) return;       // wait for the whole command
    uint8_t b[8];
    for (uint8_t i = 0; i < need; i++) b[i] = LINK.read();
    if (c == 'P') {
      pong_frame = frames_now();
      memcpy(&pong_token, b + 1, 4);
      send_pong();                               // between packets, never inside one
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
      servo_torque(true);
      servo_goal(valve_open ? valve_pos_open : valve_pos_closed, 200);
    } else if (c == 'L') {
      uint16_t m;
      memcpy(&m, b + 1, 2);
      max_pwm = m > 1000 ? 1000 : m;
      act_set(act_pwm, act_brake);
    } else if (c == 'S') { stream_restart(); streaming = true; }
    else if (c == 'X') streaming = false;
    else if (c == 'i') info();
  }
}

// Every millisecond: current sense, button. Every 10 ms: status. Command watchdog, servo, fan speed.
static void io_tick() {
  uint32_t now = millis();
  servo_poll();
  if (!cmd_timed_out && now - last_cmd_ms > CMD_TIMEOUT_MS) {
    act_set(0, false);                         // the controller went quiet: stop and shut the air
    valve_set(false);
    cmd_timed_out = true;
  }
  if (now == io_ms) return;
  io_ms = now;
  uint32_t mv = (uint32_t)analogRead(A0) * SENSE_MV_FULL / 4095u;
  sense_sum += mv;
  sense_n++;
  if (mv > sense_peak) sense_peak = mv;
  uint8_t level = digitalRead(PIN_BTN) ? 1 : 0;  // debounced: 20 ms the same
  if (level != btn_stable) {
    if (++btn_run >= 20) {
      btn_stable = level;
      btn_run = 0;
      if (!level) btn_presses++;
    }
  } else {
    btn_run = 0;
  }
  if (now - rpm_ms >= 500u) {                  // 2 pulses per turn: pulses in 0.5 s x 60 = rpm
    uint32_t p = tach_pulses;
    fan_rpm = (uint16_t)((p - tach_last) * 60u);
    tach_last = p;
    rpm_ms = now;
  }
  if (now - servo_ms >= 100u) {
    servo_ms = now;
    if (now - servo_ok_ms > 500u) servo_ok = false;
    servo_ask_position();
  }
  if (now - status_ms >= STATUS_MS) {
    status_ms = now;
    sense_mv = (uint16_t)(sense_n ? sense_sum / sense_n : 0);
    sense_peak_mv = (uint16_t)sense_peak;
    sense_sum = sense_n = sense_peak = 0;
    send_status(now);
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
  uint32_t us = (DWT->CYCCNT - t0) / cyc_per_us;
  if (us > max_loop_us) max_loop_us = us;
}
