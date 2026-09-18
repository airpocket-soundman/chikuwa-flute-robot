"""The two Yamabiko No.1 boards as data: parts, nets, schematic and PCB placement.

A board: Arduino UNO R3 shield on the UNO Q (actuator, fan, servo, valve, button).
B board: daughterboard on the UNO Breakout Carrier's J14 / J15 (microphones).
See docs in the published IO spec and BoM; every part is a through-hole part from Akizuki.
"""
from __future__ import annotations

from dataclasses import dataclass, field

NC = None  # pin left unconnected (gets a no-connect flag)


@dataclass
class Part:
    ref: str
    sym: str                 # lib:name
    value: str
    fp: str                  # footprint lib:name ("" = none)
    pins: dict               # pin number -> net name (None = no connect); multi-unit: all units merged
    sch: tuple               # schematic position (x, y) of unit 1; later units are placed to the right
    pcb: tuple | None = None  # (x, y, rot) of pad 1, or None
    units: int = 1
    akizuki: str = ""
    renumber: dict | None = None  # symbol pin renames, e.g. {"G": "1"}
    dnp: bool = False
    device: str = ""              # connectors: what plugs in, printed with the reference
    pin_names: tuple = ()         # connectors: printed next to each pad, pin 1 first
    pin_side: str = ""            # where the pin names go: above / below / left / right
    silk_ref: bool = True         # False: the reference stays off the silkscreen (the UNO Q headers)


FET_GDS = {"G": "1", "D": "2", "S": "3"}

# ----------------------------------------------------------------------------- A board

R_V = "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P2.54mm_Vertical"
C_DISC = "Capacitor_THT:C_Disc_D3.0mm_W1.6mm_P2.50mm"

A_PARTS = [
    # UNO Q R3 headers (positions of the KiCad Arduino_Uno template)
    Part("J101", "Connector_Generic:Conn_01x08", "Power", "Connector_PinHeader_2.54mm:PinHeader_1x08_P2.54mm_Vertical",
         {"1": NC, "2": NC, "3": NC, "4": "+3V3", "5": NC, "6": "GND", "7": "GND", "8": NC},
         (30, 40), (127.94, 97.46, 90), akizuki="100167", silk_ref=False),
    Part("J102", "Connector_Generic:Conn_01x10", "D8-D13/SDA/SCL", "Connector_PinHeader_2.54mm:PinHeader_1x10_P2.54mm_Vertical",
         {"1": NC, "2": NC, "3": NC, "4": "GND", "5": NC, "6": NC, "7": NC,
          "8": "ACT_IN2_MCU", "9": "ACT_IN1_MCU", "10": "BTN_EXEC"},
         (30, 80), (118.796, 49.2, 90), akizuki="100167", silk_ref=False),
    Part("J103", "Connector_Generic:Conn_01x08", "D0-D7", "Connector_PinHeader_2.54mm:PinHeader_1x08_P2.54mm_Vertical",
         {"1": "SERVO_TXEN_N", "2": "FAN_PWR_EN", "3": "FAN_PWM_MCU", "4": NC, "5": "VALVE_EN",
          "6": "FAN_TACH_MCU", "7": "SERVO_TX", "8": "SERVO_RX"},
         (30, 125), (145.72, 49.2, 90), akizuki="100167", silk_ref=False),
    Part("J104", "Connector_Generic:Conn_01x06", "A0-A5", "Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical",
         {"1": "ACT_ISENSE", "2": NC, "3": NC, "4": NC, "5": NC, "6": NC},
         (30, 165), (150.8, 97.46, 90), akizuki="100167", silk_ref=False),

    # 12 V in
    Part("J1", "Connector:Barrel_Jack", "MJ-179PH", "Connector_BarrelJack:BarrelJack_Horizontal",
         {"1": "+12V", "2": "GND"}, (80, 40), (116.0, 56.0, 0), akizuki="106568"),
    Part("C1", "Device:C_Polarized", "470u", "Capacitor_THT:CP_Radial_D10.0mm_P5.00mm",
         {"1": "+12V", "2": "GND"}, (110, 40), (103.8, 69.5, 0), akizuki="117883"),

    # actuator: TB6643KQ, current sense in its GND leg
    Part("U1", "yamabiko:TB6643KQ", "TB6643KQ", "yamabiko:Toshiba_HSIP7_P2.54mm",
         {"1": "ACT_IN1", "2": "ACT_IN2", "3": "ACT_OUT1", "4": "ACT_GND", "5": "ACT_OUT2", "6": NC, "7": "+12V"},
         (170, 40), (121.5, 55.0, 0), akizuki="107688"),
    Part("R2", "Device:R", "1k", R_V, {"1": "ACT_IN1_MCU", "2": "ACT_IN1"}, (140, 34), (135.0, 74.0, 0), akizuki="125102"),
    Part("R3", "Device:R", "1k", R_V, {"1": "ACT_IN2_MCU", "2": "ACT_IN2"}, (140, 50), (141.0, 74.0, 0), akizuki="125102"),
    Part("R1", "Device:R", "0.1 3W", "Resistor_THT:R_Axial_DIN0516_L15.5mm_D5.0mm_P20.32mm_Horizontal",
         {"1": "ACT_GND", "2": "GND"}, (200, 60), (119.6, 62.2, 0), akizuki="111011"),
    Part("R4", "Device:R", "1k", R_V, {"1": "ACT_GND", "2": "ACT_ISENSE"}, (220, 60), (141.0, 68.0, 0), akizuki="125102"),
    Part("C3", "Device:C", "1u", "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm",
         {"1": "ACT_ISENSE", "2": "GND"}, (235, 60), (135.0, 79.5, 0), akizuki="108150"),
    Part("C7", "Device:C", "0.1u", C_DISC, {"1": "+12V", "2": "ACT_GND"}, (200, 40), (135.0, 68.0, 0), akizuki="113582"),
    Part("C2", "Device:C_Polarized", "470u", "Capacitor_THT:CP_Radial_D10.0mm_P5.00mm",
         {"1": "+12V", "2": "GND"}, (215, 40), (123.4, 72.0, 0), akizuki="117883"),
    Part("J2", "Connector_Generic:Conn_01x02", "ACTUATOR", "Connector_JST:JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical",
         {"1": "ACT_OUT1", "2": "ACT_OUT2"}, (200, 25), (144.6, 55.6, 0), akizuki="112247", device="ACTUATOR", pin_names=("M1", "M2"), pin_side="below"),

    # fan: high-side switch, open-drain PWM, tach
    Part("Q1", "Device:Q_PMOS", "MTB060P06I3", "Package_TO_SOT_THT:TO-251-3_Vertical",
         {"1": "FAN_GATE", "2": "FAN_12V", "3": "+12V"}, (80, 100), (121.0, 82.0, 0), akizuki="116095", renumber=FET_GDS),
    Part("R5", "Device:R", "10k", R_V, {"1": "+12V", "2": "FAN_GATE"}, (65, 95), (129.0, 82.0, 0), akizuki="125103"),
    Part("Q2", "Transistor_FET:2N7000", "2N7000", "Package_TO_SOT_THT:TO-92_Inline_Wide",
         {"1": "GND", "2": "Q2_G", "3": "FAN_GATE"}, (80, 125), (120.6, 88.0, 0), akizuki="109723"),
    Part("R6", "Device:R", "100", R_V, {"1": "FAN_PWR_EN", "2": "Q2_G"}, (60, 125), (129.0, 88.0, 0), akizuki="125101"),
    Part("R7", "Device:R", "100k", R_V, {"1": "Q2_G", "2": "GND"}, (65, 140), (135.0, 88.0, 0), akizuki="125104"),
    Part("Q3", "Transistor_FET:2N7000", "2N7000", "Package_TO_SOT_THT:TO-92_Inline_Wide",
         {"1": "GND", "2": "Q3_G", "3": "FAN_PWM"}, (115, 125), (131.0, 93.5, 0), akizuki="109723"),
    Part("R8", "Device:R", "100", R_V, {"1": "FAN_PWM_MCU", "2": "Q3_G"}, (95, 125), (138.8, 93.5, 0), akizuki="125101"),
    Part("R9", "Device:R", "100k", R_V, {"1": "Q3_G", "2": "GND"}, (100, 140), (144.0, 93.5, 0), akizuki="125104"),
    Part("R10", "Device:R", "10k", R_V, {"1": "+3V3", "2": "FAN_TACH"}, (135, 95), (142.2, 83.0, 0), akizuki="125103"),
    Part("R11", "Device:R", "1k", R_V, {"1": "FAN_TACH", "2": "FAN_TACH_MCU"}, (150, 110), (141.5, 88.0, 0), akizuki="125102"),
    Part("J3", "Connector_Generic:Conn_01x04", "FAN", "Connector_JST:JST_XH_B4B-XH-A_1x04_P2.50mm_Vertical",
         {"1": "GND", "2": "FAN_12V", "3": "FAN_TACH", "4": "FAN_PWM"}, (160, 125), (150.8, 91.6, 0), akizuki="112249", device="FAN", pin_names=("GND", "12V", "TACH", "PWM"), pin_side="above"),

    # servo: 6 V regulator, half-duplex buffer
    Part("U2", "Regulator_Linear:L7806", "NJM7806FA", "Package_TO_SOT_THT:TO-220F-3_Vertical",
         {"1": "+12V", "2": "GND", "3": "+6V"}, (200, 110), (103.2, 81.5, 0), akizuki="117193"),
    Part("C6", "Device:C_Polarized", "10u", "Capacitor_THT:CP_Radial_D5.0mm_P2.00mm",
         {"1": "+12V", "2": "GND"}, (180, 125), (103.0, 96.7, 0), akizuki="117897"),
    Part("C8", "Device:C", "0.1u", C_DISC, {"1": "+12V", "2": "GND"}, (190, 140), (112.3, 96.8, 0), akizuki="113582"),
    Part("C4", "Device:C_Polarized", "100u", "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm",
         {"1": "+6V", "2": "GND"}, (225, 125), (114.2, 69.5, 0), akizuki="117877"),
    Part("C9", "Device:C", "0.1u", C_DISC, {"1": "+6V", "2": "GND"}, (235, 140), (112.0, 92.0, 0), akizuki="113582"),
    Part("U3", "74xx:74LS125", "74HC125", "Package_DIP:DIP-14_W7.62mm_Socket",
         {"1": "SERVO_TXEN_N", "2": "SERVO_TX", "3": "SERVO_SIG",
          "4": "GND", "5": "SERVO_RX_IN", "6": "SERVO_RX",
          "10": "+3V3", "9": "GND", "8": NC,
          "13": "+3V3", "12": "GND", "11": NC,
          "7": "GND", "14": "+3V3"},
         (270, 40), (147.5, 71.5, 90), units=5, akizuki="131766"),
    Part("R12", "Device:R", "1k", R_V, {"1": "SERVO_SIG", "2": "SERVO_RX_IN"}, (250, 90), (147.5, 77.0, 0), akizuki="125102"),
    Part("R13", "Device:R", "10k", R_V, {"1": "+3V3", "2": "SERVO_SIG"}, (320, 30), (153.5, 77.0, 0), akizuki="125103"),
    Part("R14", "Device:R", "10k", R_V, {"1": "+3V3", "2": "SERVO_TXEN_N"}, (250, 25), (159.5, 77.0, 0), akizuki="125103"),
    Part("C10", "Device:C", "0.1u", C_DISC, {"1": "+3V3", "2": "GND"}, (330, 90), (147.4, 83.0, 0), akizuki="113582"),
    Part("J4", "Connector_Generic:Conn_01x03", "SCS0009", "Connector_JST:JST_XH_B3B-XH-A_1x03_P2.50mm_Vertical",
         {"1": "GND", "2": "+6V", "3": "SERVO_SIG"}, (345, 40), (154.6, 55.6, 0), akizuki="112248", device="SCS0009", pin_names=("GND", "6V", "SIG"), pin_side="below"),

    # backup valve (2V025), not fitted unless needed
    Part("Q4", "Device:Q_NMOS", "2SK4017", "Package_TO_SOT_THT:TO-251-3_Vertical",
         {"1": "Q4_G", "2": "VALVE_N", "3": "GND"}, (300, 125), (111.3, 88.6, 0), akizuki="107597", renumber=FET_GDS, dnp=True),
    Part("R15", "Device:R", "100", R_V, {"1": "VALVE_EN", "2": "Q4_G"}, (280, 125), (114.3, 78.0, 0), akizuki="125101", dnp=True),
    Part("R16", "Device:R", "100k", R_V, {"1": "Q4_G", "2": "GND"}, (285, 140), (114.3, 84.0, 0), akizuki="125104", dnp=True),
    Part("D1", "Device:D_Schottky", "1N5819", "Diode_THT:D_DO-41_SOD81_P5.08mm_Vertical_AnodeUp",
         {"1": "+12V", "2": "VALVE_N"}, (320, 110), (119.2, 96.2, 0), akizuki="117244", dnp=True),
    Part("J5", "Connector_Generic:Conn_01x02", "VALVE", "Connector_JST:JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical",
         {"1": "+12V", "2": "VALVE_N"}, (345, 125), (103.5, 88.0, 0), akizuki="112247", device="VALVE", pin_names=("12V", "V-"), pin_side="below", dnp=True),

    # button and spare connectors
    Part("SW1", "Switch:SW_Push", "EXEC", "Button_Switch_THT:SW_PUSH_6mm",
         {"1": "BTN_EXEC", "2": "GND"}, (80, 165), (158.3, 81.4, 0), akizuki="103647", device="EXEC"),
    Part("R17", "Device:R", "10k", R_V, {"1": "+3V3", "2": "BTN_EXEC"}, (60, 160), (152.6, 83.0, 0), akizuki="125103"),
    # power flags (schematic only)
    Part("#FLG01", "power:PWR_FLAG", "PWR_FLAG", "", {"1": "+12V"}, (380, 40)),
    Part("#FLG02", "power:PWR_FLAG", "PWR_FLAG", "", {"1": "GND"}, (380, 55)),
    Part("#FLG03", "power:PWR_FLAG", "PWR_FLAG", "", {"1": "+3V3"}, (380, 70)),
    Part("#FLG04", "power:PWR_FLAG", "PWR_FLAG", "", {"1": "ACT_GND"}, (380, 85)),

]

# ----------------------------------------------------------------------------- B board

# Carrier geometry, top view, origin = carrier bottom-right corner placed at (150, 110):
# J14 odd column 25.37 mm and J15 odd column 10.13 mm from the right edge, pin 1 50.8 mm above the bottom edge.
CX, CY = 150.0, 110.0
J14_1 = (CX - 25.37, CY - 50.8)
J15_1 = (CX - 10.13, CY - 50.8)
# nothing left of J14: the UNO Q sits there. The board is trimmed 0.5 mm outside J14's courtyard and
# reaches 13 mm past the carrier's right edge instead (only its edge power pads and mounting holes are below)
B_EDGE = (J14_1[0] - 1.81 - 0.5, CY - 53.34, CX + 13.0, CY)   # x0, y0, x1, y1

J14_GND = ["5", "6", "11", "12", "17", "18", "23", "25", "26", "31", "32", "33", "37"]
j14 = {str(i): NC for i in range(1, 41)}
j14.update({"13": "+3V3", "15": "+3V3", "19": "+1V8", "21": "+1V8"})
j14.update({p: "GND" for p in J14_GND})
j15 = {str(i): NC for i in range(1, 41)}
j15.update({"8": "SAI_SCK", "9": "SAI_WS", "10": "SAI_SD", "23": "GND", "30": "GND", "40": "GND",
            "32": "MI2S_SCK", "34": "MI2S_WS", "36": "MI2S_SD",
            # spares: I2C4 on PF14 / PF15 (not the core's default I2C4 pins) and the OPAMP1 / ADC pins
            "16": "I2C4_SCL", "18": "I2C4_SDA", "20": "PA3", "22": "PA0", "24": "PA1"})

SOCKET = "Connector_PinHeader_2.54mm:PinHeader_2x20_P2.54mm_Vertical"  # pad pattern of the carrier header, top view

B_PARTS = [
    Part("J14", "Connector_Generic:Conn_02x20_Odd_Even", "Carrier J14 (socket, bottom)", SOCKET, j14,
         (40, 40), (J14_1[0], J14_1[1], 0), akizuki="100085"),
    Part("J15", "Connector_Generic:Conn_02x20_Odd_Even", "Carrier J15 (socket, bottom)", SOCKET, j15,
         (40, 120), (J15_1[0], J15_1[1], 0), akizuki="100085"),

    # MIC1: ICS-43434 on the MCU's SAI1, 3.3 V
    Part("J1", "Connector_Generic:Conn_01x06", "MIC1 SAI1 3V3", "Connector_JST:JST_XH_B6B-XH-A_1x06_P2.50mm_Vertical",
         {"1": "+3V3", "2": "GND", "3": "MIC1_SCK", "4": "MIC1_WS", "5": "MIC1_SD", "6": "GND"},
         (160, 40), (133.0, 61.0, 270), akizuki="112251", device="MIC1 SAI1", pin_names=("3V3", "GND", "SCK", "WS", "SD", "LR"), pin_side="right"),
    Part("R1", "Device:R", "33", R_V, {"1": "SAI_SCK", "2": "MIC1_SCK"}, (120, 30), (147.2, 59.5, 270), akizuki="103941"),
    Part("R2", "Device:R", "33", R_V, {"1": "SAI_WS", "2": "MIC1_WS"}, (120, 45), (147.2, 67.5, 270), akizuki="103941"),
    Part("R3", "Device:R", "33", R_V, {"1": "SAI_SD", "2": "MIC1_SD"}, (120, 60), (147.2, 75.5, 270), akizuki="103941"),
    Part("R7", "Device:R", "100k", R_V, {"1": "SAI_SD", "2": "GND"}, (135, 75), (152.2, 59.5, 270), akizuki="125104"),
    Part("C1", "Device:C", "0.1u", C_DISC, {"1": "+3V3", "2": "GND"}, (150, 75), (152.2, 67.5, 270), akizuki="113582"),

    # MIC2: backup route on MI2S0, 1.8 V
    Part("J2", "Connector_Generic:Conn_01x06", "MIC2 MI2S0 1V8", "Connector_JST:JST_XH_B6B-XH-A_1x06_P2.50mm_Vertical",
         {"1": "+1V8", "2": "GND", "3": "MIC2_SCK", "4": "MIC2_WS", "5": "MIC2_SD", "6": "GND"},
         (160, 120), (133.0, 93.5, 270), akizuki="112251", device="MIC2 MI2S0", pin_names=("1V8", "GND", "SCK", "WS", "SD", "LR"), pin_side="right"),
    Part("R4", "Device:R", "33", R_V, {"1": "MI2S_SCK", "2": "MIC2_SCK"}, (120, 110), (147.2, 85.0, 270), akizuki="103941"),
    Part("R5", "Device:R", "33", R_V, {"1": "MI2S_WS", "2": "MIC2_WS"}, (120, 125), (147.2, 93.0, 270), akizuki="103941"),
    Part("R6", "Device:R", "33", R_V, {"1": "MI2S_SD", "2": "MIC2_SD"}, (120, 140), (147.2, 101.0, 270), akizuki="103941"),
    Part("R8", "Device:R", "100k", R_V, {"1": "MI2S_SD", "2": "GND"}, (135, 155), (152.2, 85.0, 270), akizuki="125104"),
    Part("C2", "Device:C", "0.1u", C_DISC, {"1": "+1V8", "2": "GND"}, (150, 155), (152.2, 93.0, 270), akizuki="113582"),

    # spares on the MCU side (3.3 V)
    Part("J3", "Connector_Generic:Conn_01x04", "I2C4 3V3", "Connector_JST:JST_XH_B4B-XH-A_1x04_P2.50mm_Vertical",
         {"1": "+3V3", "2": "GND", "3": "I2C4_SDA", "4": "I2C4_SCL"}, (190, 40), (158.8, 66.5, 270), akizuki="112249",
         device="I2C", pin_names=("3V3", "GND", "SDA", "SCL"), pin_side="left"),
    Part("R9", "Device:R", "4.7k", R_V, {"1": "+3V3", "2": "I2C4_SDA"}, (170, 40), (152.2, 75.5, 270), akizuki="125472"),
    Part("R10", "Device:R", "4.7k", R_V, {"1": "+3V3", "2": "I2C4_SCL"}, (170, 55), (152.2, 101.0, 270), akizuki="125472"),
    Part("J4", "Connector_Generic:Conn_01x05", "GPIO 3V3", "Connector_JST:JST_XH_B5B-XH-A_1x05_P2.50mm_Vertical",
         {"1": "+3V3", "2": "GND", "3": "PA0", "4": "PA1", "5": "PA3"}, (190, 120), (158.8, 84.0, 270), akizuki="112250",
         device="ADC", pin_names=("3V3", "GND", "PA0", "PA1", "PA3"), pin_side="left"),

    Part("#FLG01", "power:PWR_FLAG", "PWR_FLAG", "", {"1": "+3V3"}, (230, 40)),
    Part("#FLG02", "power:PWR_FLAG", "PWR_FLAG", "", {"1": "+1V8"}, (230, 55)),
    Part("#FLG03", "power:PWR_FLAG", "PWR_FLAG", "", {"1": "GND"}, (230, 70)),
]

POWER_NETS = {"GND", "+12V", "+6V", "+3V3", "+1V8"}
WIDE_NETS = {"+12V", "+6V", "ACT_GND", "ACT_OUT1", "ACT_OUT2", "FAN_12V", "VALVE_N"}
