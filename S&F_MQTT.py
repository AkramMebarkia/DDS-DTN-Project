"""
Enhanced MQTT with Spray & Focus DTN Routing

UAVs collect sensor data and relay it toward the sink using Spray & Focus:
  - Spray phase: broadcast copies (token-splitting) to encountered UAVs
  - Focus phase: utility-guided unicast to the best next-hop toward the sink
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ==========================================
# GLOBAL CONFIG
# ==========================================

GLOBAL_QOS     = 1
INITIAL_TOKENS = 8

NUM_UAVS    = 6
SINK_ID     = 0
NUM_SENSORS = 10

SINK_MOBILITY_FRACTION = 0.40
MQTT_TOPIC_LEN         = 16

ZIGBEE_MAX_FRAME   = 127
ZIGBEE_MAX_PAYLOAD = 100


# ==========================================
# PHYSICS / RADIO PROFILES
# ==========================================

class PhyConst:
    # UAVs fly at 90 m absolute altitude. IoT sensors sit at z = 10 m (pole/rooftop),
    # so the effective air-gap between sensor and UAV is 80 m — matching the 80 m
    # UAV-altitude figure stated in the paper.
    H      = 90.0

    P_C    = 5.0
    U_TIP  = 120.0
    V_0    = 4.03
    D_0    = 0.6
    RHO    = 1.225
    AREA   = 0.503
    OMEGA  = 300.0
    R_RAD  = 0.4
    DELTA  = 0.012
    S_SOL  = 0.05
    WEIGHT = 20.0

    SENSOR_PAYLOAD_BYTES    = 64
    WIFI_DATA_PAYLOAD_BYTES = 256
    WIFI_MAX_RATE           = 54_000_000.0
    CONTROL_PAYLOAD_BYTES   = 80


class PHYProfile:
    def __init__(self, name: str, B: float, P_tx: float, N0: float,
                 E_tx_per_bit: float, E_rx_per_bit: float):
        self.name         = name
        self.B            = B
        self.P_tx         = P_tx
        self.N0           = N0
        self.beta0        = None
        self.E_tx_per_bit = E_tx_per_bit
        self.E_rx_per_bit = E_rx_per_bit


# E_tx_per_bit is a system-level measured value that already captures the full
# round-trip energy cost per bit (transmitter + receiver hardware).
# E_rx_per_bit is therefore set to 0 to avoid double-counting; we keep the
# field present so the rest of the code never needs a special-case branch.
#
# ZigBee: ~1 µJ/bit  — Siekkinen et al., IEEE WCNC 2012
# WiFi:   ~200 nJ/bit — Liu & Choi, ACM SIGMETRICS 2023
ZIGBEE = PHYProfile("zigbee", B=250_000.0,    P_tx=0.0774, N0=1e-13,
                    E_tx_per_bit=1e-6, E_rx_per_bit=0)
WIFI   = PHYProfile("wifi",   B=20_000_000.0, P_tx=1.5,    N0=1e-13,
                    E_tx_per_bit=2e-7, E_rx_per_bit=0)

REF_DIST = 100.0


def calibrate_beta0(profile: PHYProfile):
    profile.beta0 = profile.N0 * (REF_DIST ** 2) / profile.P_tx


for _p in (ZIGBEE, WIFI):
    calibrate_beta0(_p)


def shannon_rate_3d(dist_3d: float, profile: PHYProfile) -> float:
    if dist_3d <= 0.0:
        dist_3d = 1e-3
    snr  = (profile.beta0 * profile.P_tx) / (profile.N0 * (dist_3d ** 2))
    rate = profile.B * math.log2(1.0 + snr)
    if profile.name == "zigbee":
        rate = min(rate, 250_000.0)
    elif profile.name == "wifi":
        rate = min(rate, PhyConst.WIFI_MAX_RATE)
    return rate


def link_rate(u_pos: np.ndarray, v_pos: np.ndarray,
              is_ground_to_uav: bool) -> Tuple[float, float, PHYProfile]:
    d    = float(np.linalg.norm(u_pos - v_pos))
    prof = ZIGBEE if is_ground_to_uav else WIFI
    return shannon_rate_3d(d, prof), d, prof


# ==========================================
# MQTT FRAME SIZES
# ==========================================

WIFI_TCP_IP_OVERHEAD      = 40
WIFI_L2_OVERHEAD          = 30
WIFI_TRANSPORT_OVERHEAD   = WIFI_TCP_IP_OVERHEAD + WIFI_L2_OVERHEAD   # 70 bytes
ZIGBEE_L2_OVERHEAD        = 15
ZIGBEE_TRANSPORT_OVERHEAD = ZIGBEE_L2_OVERHEAD

MQTT_FIXED_HDR = 2
MQTT_TOPIC_HDR = 2 + MQTT_TOPIC_LEN
MQTT_PKT_ID    = 2

# Legacy aliases
TCP_IP_OVERHEAD   = WIFI_TCP_IP_OVERHEAD
L2_OVERHEAD       = WIFI_L2_OVERHEAD
BASE_WIRE_OVERHEAD = WIFI_TRANSPORT_OVERHEAD


def mqtt_frame_size_wifi(payload_bytes: int, qos: int) -> int:
    overhead = MQTT_FIXED_HDR + MQTT_TOPIC_HDR + (MQTT_PKT_ID if qos == 1 else 0)
    return payload_bytes + overhead + WIFI_TRANSPORT_OVERHEAD


def mqtt_frame_size_zigbee(payload_bytes: int, qos: int) -> int:
    overhead = MQTT_FIXED_HDR + MQTT_TOPIC_HDR + (MQTT_PKT_ID if qos == 1 else 0)
    return payload_bytes + overhead + ZIGBEE_TRANSPORT_OVERHEAD


def mqtt_puback_size_wifi() -> int:
    return MQTT_FIXED_HDR + MQTT_PKT_ID + WIFI_TRANSPORT_OVERHEAD


def mqtt_puback_size_zigbee() -> int:
    return MQTT_FIXED_HDR + MQTT_PKT_ID + ZIGBEE_TRANSPORT_OVERHEAD


def mqtt_frame_size(payload_bytes: int, qos: int) -> int:
    return mqtt_frame_size_wifi(payload_bytes, qos)


def mqtt_puback_size() -> int:
    return mqtt_puback_size_wifi()


# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class RoutingControlMessage:
    utility_to_sink: float

    def payload_size(self) -> int:
        return 12


@dataclass
class SprayMessage:
    msg_id:        int
    source_id:     int
    creation_time: float
    hop_count:     int
    tokens:        int
    payload:       bytes
    qos:           int
    payload_bytes: int = PhyConst.WIFI_DATA_PAYLOAD_BYTES

    def payload_size(self) -> int:
        return self.payload_bytes


# ==========================================
# MQTT TRANSMISSION
# ==========================================

class MQTTTransmission:
    @staticmethod
    def transmit_control(rate_bps: float, profile: PHYProfile,
                         control_msg: RoutingControlMessage) -> Tuple[bool, float, float, int]:
        if rate_bps < profile.B:
            return False, 0.0, 0.0, 0
        if profile.name == "zigbee":
            frame_bytes = mqtt_frame_size_zigbee(control_msg.payload_size(), qos=1)
            if frame_bytes > ZIGBEE_MAX_FRAME:
                return False, 0.0, 0.0, 0
        else:
            frame_bytes = mqtt_frame_size_wifi(control_msg.payload_size(), qos=1)
        bits = frame_bytes * 8
        return True, bits * profile.E_tx_per_bit, bits * profile.E_rx_per_bit, frame_bytes

    @staticmethod
    def transmit_data(rate_bps: float, profile: PHYProfile,
                      data_msg: SprayMessage) -> Tuple[bool, float, float, int]:
        if rate_bps < profile.B:
            return False, 0.0, 0.0, 0
        if profile.name == "zigbee":
            frame_bytes = mqtt_frame_size_zigbee(data_msg.payload_size(), data_msg.qos)
            if frame_bytes > ZIGBEE_MAX_FRAME:
                return False, 0.0, 0.0, 0
        else:
            frame_bytes = mqtt_frame_size_wifi(data_msg.payload_size(), data_msg.qos)
        bits = frame_bytes * 8
        return True, bits * profile.E_tx_per_bit, bits * profile.E_rx_per_bit, frame_bytes

    @staticmethod
    def transmit_puback(rate_bps: float,
                        profile: PHYProfile) -> Tuple[bool, float, float, int]:
        if rate_bps < profile.B:
            return False, 0.0, 0.0, 0
        frame_bytes = (mqtt_puback_size_zigbee() if profile.name == "zigbee"
                       else mqtt_puback_size_wifi())
        bits = frame_bytes * 8
        return True, bits * profile.E_tx_per_bit, bits * profile.E_rx_per_bit, frame_bytes


# ==========================================
# UAV AGENT
# ==========================================

class MqttUAVAgent:
    MAX_BUFFER = 250

    def __init__(self, uid: int, start_pos, is_sink: bool = False, area_size: float = 500):
        self.id              = uid
        self.pos             = np.array(start_pos, dtype=float)
        self.is_sink         = is_sink
        self.area_size       = area_size
        self.energy          = 300_000.0
        self.radio_tx_energy = 0.0
        self.radio_rx_energy = 0.0
        self.waypoints       = []
        self.buffer:         List[SprayMessage] = []
        self.seen_msgs       = set()
        self.encounter_timers: Dict[int, float] = {i: 9999.0 for i in range(NUM_UAVS)}
        self.encounter_timers[uid] = 0.0
        if self.is_sink:
            self.sink_received_count = 0

    def move(self, dt: float):
        for k in self.encounter_timers:
            self.encounter_timers[k] += dt
        self.encounter_timers[self.id] = 0.0

        if not self.waypoints:
            center = self.area_size / 2
            if self.id == SINK_ID:
                radius = self.area_size * SINK_MOBILITY_FRACTION / 2
                self.waypoints = [
                    np.array([center + random.uniform(-radius, radius),
                               center + random.uniform(-radius, radius),
                               PhyConst.H])
                    for _ in range(5)
                ]
            else:
                self.waypoints = [
                    np.array([random.uniform(100, self.area_size - 100),
                               random.uniform(100, self.area_size - 100),
                               PhyConst.H])
                    for _ in range(5)
                ]

        speed  = 20.0
        target = self.waypoints[0]
        diff   = target - self.pos
        dist   = np.linalg.norm(diff)

        if dist <= speed * dt:
            self.pos = target
            self.waypoints.pop(0)
        else:
            self.pos += (diff / dist) * speed * dt

        self.energy -= self.calc_flight_power(speed) * dt

    @staticmethod
    def calc_flight_power(velocity: float) -> float:
        P0    = (PhyConst.DELTA / 8.0) * PhyConst.RHO * PhyConst.S_SOL * PhyConst.AREA \
                * (PhyConst.OMEGA ** 3) * (PhyConst.R_RAD ** 3)
        P_ind = 1.1 * (PhyConst.WEIGHT ** 1.5) / math.sqrt(2 * PhyConst.RHO * PhyConst.AREA)
        if velocity < 0.1:
            return P0 + P_ind
        term1 = P0 * (1.0 + 3.0 * velocity ** 2 / (PhyConst.U_TIP ** 2))
        term2 = P_ind * math.sqrt(
            math.sqrt(1.0 + (velocity ** 4) / (4 * PhyConst.V_0 ** 4))
            - (velocity ** 2) / (2 * PhyConst.V_0 ** 2)
        )
        term3 = 0.5 * PhyConst.D_0 * PhyConst.RHO * PhyConst.S_SOL * PhyConst.AREA * velocity ** 3
        return term1 + term2 + term3

    def get_utility_message(self) -> RoutingControlMessage:
        return RoutingControlMessage(utility_to_sink=self.encounter_timers[SINK_ID])


# ==========================================
# STANDALONE VERBOSE RUNNER
# ==========================================

def run_mqtt_simulation():
    print("=" * 70)
    print("FIXED MQTT-DTN UAV SIMULATION")
    print("=" * 70)
    print(f"  UAVs: {NUM_UAVS} (Sink: UAV {SINK_ID}) | Sensors: {NUM_SENSORS} | QoS: {GLOBAL_QOS}")
    print(f"  Tokens: {INITIAL_TOKENS} | ZigBee 250 kbps / WiFi 20 Mbps")
    print("=" * 70 + "\n")

    agents: Dict[int, MqttUAVAgent] = {}
    agents[SINK_ID] = MqttUAVAgent(SINK_ID, [250.0, 250.0, PhyConst.H], is_sink=True)
    for i in range(1, NUM_UAVS):
        agents[i] = MqttUAVAgent(i, [random.uniform(50, 450), random.uniform(50, 450), PhyConst.H])

    def generate_spread_sensors(num_sensors, area_size, seed=42):
        rng      = np.random.RandomState(seed)
        margin   = 100
        min_dist = (area_size - 2 * margin) / (num_sensors ** 0.5 + 1)
        positions = []
        for _ in range(num_sensors):
            for attempt in range(1000):
                x, y = rng.uniform(margin, area_size - margin), rng.uniform(margin, area_size - margin)
                if all(np.sqrt((x - px) ** 2 + (y - py) ** 2) >= min_dist
                       for px, py, _ in positions) or attempt == 999:
                    positions.append([x, y, 10.0])
                    break
        return [np.array(p) for p in positions]

    iot_nodes = generate_spread_sensors(NUM_SENSORS, 500, seed=42)

    sim_time    = 0.0
    duration    = 1500.0
    dt          = 0.1
    SENSOR_RATE = 2.0
    MSG_COUNTER = 0
    sensor_queues: List[List[Tuple[int, int, float]]] = [[] for _ in range(NUM_SENSORS)]

    total_generated = total_delivered = 0
    total_generated_qos0 = total_generated_qos1 = 0
    total_delivered_qos0 = total_delivered_qos1 = 0
    uav_relay_events = spray_events = focus_events = sink_delivery_events = 0
    control_messages_sent = 0
    latencies: List[float] = []
    latencies_qos0: List[float] = []
    latencies_qos1: List[float] = []
    hop_counts: List[int] = []

    sensor_tx_energy = sensor_rx_energy = 0.0
    control_tx_energy = control_rx_energy = 0.0
    data_wifi_tx_energy = data_wifi_rx_energy = 0.0
    control_bytes_sent = data_bytes_sent = 0

    print(f"Starting simulation (duration: {duration}s)...\n")

    try:
        while sim_time < duration:
            sim_time += dt

            # Sensor data generation
            for s in range(NUM_SENSORS):
                if random.random() < SENSOR_RATE * dt:
                    MSG_COUNTER += 1
                    sensor_queues[s].append((MSG_COUNTER, GLOBAL_QOS, sim_time))
                    if len(sensor_queues[s]) > 50:
                        sensor_queues[s].pop(0)
                    total_generated += 1
                    if GLOBAL_QOS == 0:
                        total_generated_qos0 += 1
                    else:
                        total_generated_qos1 += 1

            # UAV movement
            for agent in agents.values():
                agent.move(dt)

            # Encounter detection + distance-vector utility update
            for i in range(NUM_UAVS):
                for j in range(i + 1, NUM_UAVS):
                    rate, d, _ = link_rate(agents[i].pos, agents[j].pos, is_ground_to_uav=False)
                    if rate >= WIFI.B:
                        agents[i].encounter_timers[j] = 0.0
                        agents[j].encounter_timers[i] = 0.0
                        t_meet   = d / 20.0
                        t_i_sink = agents[i].encounter_timers[SINK_ID]
                        t_j_sink = agents[j].encounter_timers[SINK_ID]
                        if t_j_sink + t_meet < t_i_sink:
                            agents[i].encounter_timers[SINK_ID] = t_j_sink + t_meet
                        if t_i_sink + t_meet < t_j_sink:
                            agents[j].encounter_timers[SINK_ID] = t_i_sink + t_meet

            # Spray & Focus routing (UAV ↔ UAV)
            for i in range(NUM_UAVS):
                if i == SINK_ID:
                    continue
                ai = agents[i]
                if not ai.buffer:
                    continue

                for j in range(NUM_UAVS):
                    if j == i or j == SINK_ID:
                        continue
                    aj = agents[j]
                    rate, d, prof = link_rate(ai.pos, aj.pos, is_ground_to_uav=False)
                    if rate < prof.B:
                        continue

                    max_bytes          = (rate * dt) / 8.0
                    bytes_this_step    = 0
                    spray_messages     = [m for m in ai.buffer if m.tokens > 1]
                    focus_messages     = [m for m in ai.buffer if m.tokens == 1]

                    # --- Spray phase ---
                    for msg in spray_messages:
                        if msg.msg_id in aj.seen_msgs or len(aj.buffer) >= aj.MAX_BUFFER:
                            continue
                        send_tokens = msg.tokens // 2
                        if send_tokens <= 0:
                            continue
                        frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                        if bytes_this_step + frame_bytes > max_bytes:
                            break

                        success, tx_e, rx_e, db = MQTTTransmission.transmit_data(rate, prof, msg)
                        if not success:
                            continue

                        ai.energy -= tx_e;  ai.radio_tx_energy += tx_e;  data_wifi_tx_energy += tx_e
                        aj.energy -= rx_e;  aj.radio_rx_energy += rx_e;  data_wifi_rx_energy += rx_e
                        data_bytes_sent += db;  bytes_this_step += db

                        if msg.qos == 1:
                            s_ack, ta, ra, _ = MQTTTransmission.transmit_puback(rate, prof)
                            if s_ack:
                                aj.energy -= ta;  aj.radio_tx_energy += ta;  data_wifi_tx_energy += ta
                                ai.energy -= ra;  ai.radio_rx_energy += ra;  data_wifi_rx_energy += ra

                        aj.buffer.append(SprayMessage(
                            msg_id=msg.msg_id, source_id=msg.source_id,
                            creation_time=msg.creation_time, hop_count=msg.hop_count + 1,
                            tokens=send_tokens, payload=msg.payload, qos=msg.qos,
                            payload_bytes=msg.payload_bytes))
                        aj.seen_msgs.add(msg.msg_id)
                        msg.tokens -= send_tokens
                        uav_relay_events += 1;  spray_events += 1

                    # --- Focus phase ---
                    # Exchange utility values before forwarding, then route to the better carrier.
                    if not focus_messages:
                        continue
                    inquiry_msg      = ai.get_utility_message()
                    ctrl_frame_size  = mqtt_frame_size(inquiry_msg.payload_size(), 1)
                    total_ctrl_bytes = 2 * ctrl_frame_size + 2 * mqtt_puback_size()
                    if bytes_this_step + total_ctrl_bytes > max_bytes:
                        continue

                    # Step 1 — inquiry
                    ok, tx_e, rx_e, cb = MQTTTransmission.transmit_control(rate, prof, inquiry_msg)
                    if not ok:
                        continue
                    ai.energy -= tx_e;  ai.radio_tx_energy += tx_e;  control_tx_energy += tx_e
                    aj.energy -= rx_e;  aj.radio_rx_energy += rx_e;  control_rx_energy += rx_e
                    control_messages_sent += 1;  control_bytes_sent += cb;  bytes_this_step += cb

                    s_ack, ta, ra, ab = MQTTTransmission.transmit_puback(rate, prof)
                    if s_ack:
                        aj.energy -= ta;  aj.radio_tx_energy += ta;  control_tx_energy += ta
                        ai.energy -= ra;  ai.radio_rx_energy += ra;  control_rx_energy += ra
                        bytes_this_step += ab

                    # Step 2 — response
                    response_msg = aj.get_utility_message()
                    ok2, tx_e2, rx_e2, cb2 = MQTTTransmission.transmit_control(rate, prof, response_msg)
                    if not ok2:
                        continue
                    aj.energy -= tx_e2;  aj.radio_tx_energy += tx_e2;  control_tx_energy += tx_e2
                    ai.energy -= rx_e2;  ai.radio_rx_energy += rx_e2;  control_rx_energy += rx_e2
                    control_messages_sent += 1;  control_bytes_sent += cb2;  bytes_this_step += cb2

                    s_ack2, ta2, ra2, ab2 = MQTTTransmission.transmit_puback(rate, prof)
                    if s_ack2:
                        ai.energy -= ta2;  ai.radio_tx_energy += ta2;  control_tx_energy += ta2
                        aj.energy -= ra2;  aj.radio_rx_energy += ra2;  control_rx_energy += ra2
                        bytes_this_step += ab2

                    # Step 3 — forward if neighbor is closer to sink
                    my_utility = ai.encounter_timers[SINK_ID]
                    nb_utility = response_msg.utility_to_sink
                    t_meet     = d / 20.0

                    for msg in focus_messages:
                        if (nb_utility - t_meet) >= my_utility:
                            continue
                        if msg.msg_id in aj.seen_msgs or len(aj.buffer) >= aj.MAX_BUFFER:
                            continue
                        frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                        if bytes_this_step + frame_bytes > max_bytes:
                            break

                        ok3, tx_e3, rx_e3, db3 = MQTTTransmission.transmit_data(rate, prof, msg)
                        if not ok3:
                            continue

                        ai.energy -= tx_e3;  ai.radio_tx_energy += tx_e3;  data_wifi_tx_energy += tx_e3
                        aj.energy -= rx_e3;  aj.radio_rx_energy += rx_e3;  data_wifi_rx_energy += rx_e3
                        data_bytes_sent += db3;  bytes_this_step += db3

                        if msg.qos == 1:
                            s_ack3, ta3, ra3, _ = MQTTTransmission.transmit_puback(rate, prof)
                            if s_ack3:
                                aj.energy -= ta3;  aj.radio_tx_energy += ta3;  data_wifi_tx_energy += ta3
                                ai.energy -= ra3;  ai.radio_rx_energy += ra3;  data_wifi_rx_energy += ra3

                        aj.buffer.append(SprayMessage(
                            msg_id=msg.msg_id, source_id=msg.source_id,
                            creation_time=msg.creation_time, hop_count=msg.hop_count + 1,
                            tokens=1, payload=msg.payload, qos=msg.qos,
                            payload_bytes=msg.payload_bytes))
                        aj.seen_msgs.add(msg.msg_id)
                        if msg in ai.buffer:
                            ai.buffer.remove(msg)
                        uav_relay_events += 1;  focus_events += 1

            # UAV → Sink delivery
            sink = agents[SINK_ID]
            for i in range(1, NUM_UAVS):
                ai = agents[i]
                if not ai.buffer:
                    continue
                rate_bps, _, prof = link_rate(ai.pos, sink.pos, is_ground_to_uav=False)
                if rate_bps < prof.B:
                    continue

                max_bytes      = (rate_bps * dt) / 8.0
                bytes_sent     = 0
                msgs_to_remove = []

                for msg in ai.buffer:
                    if msg.msg_id in sink.seen_msgs:
                        msgs_to_remove.append(msg);  continue
                    frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                    if bytes_sent + frame_bytes > max_bytes:
                        break
                    ok, tx_e, rx_e, _ = MQTTTransmission.transmit_data(rate_bps, prof, msg)
                    if not ok:
                        continue

                    ai.energy -= tx_e;    ai.radio_tx_energy += tx_e;    data_wifi_tx_energy += tx_e
                    sink.energy -= rx_e;  sink.radio_rx_energy += rx_e;  data_wifi_rx_energy += rx_e

                    if msg.qos == 1:
                        s_ack, ta, ra, _ = MQTTTransmission.transmit_puback(rate_bps, prof)
                        if s_ack:
                            sink.energy -= ta;  sink.radio_tx_energy += ta;  data_wifi_tx_energy += ta
                            ai.energy -= ra;    ai.radio_rx_energy += ra;    data_wifi_rx_energy += ra

                    sink.seen_msgs.add(msg.msg_id)
                    sink.sink_received_count += 1
                    total_delivered += 1;  sink_delivery_events += 1
                    hop_counts.append(msg.hop_count)
                    lat = sim_time - msg.creation_time
                    latencies.append(lat)
                    (latencies_qos0 if msg.qos == 0 else latencies_qos1).append(lat)
                    (total_delivered_qos0 if msg.qos == 0 else total_delivered_qos1)
                    bytes_sent += frame_bytes;  msgs_to_remove.append(msg)

                    if total_delivered % 50 == 0:
                        print(f"[t={sim_time:.1f}s] Delivered: {total_delivered} | "
                              f"PDR: {100*total_delivered/max(1,total_generated):.1f}%")

                for m in msgs_to_remove:
                    if m in ai.buffer:
                        ai.buffer.remove(m)

            # Periodic duplicate purge
            if total_delivered > 0 and int(sim_time * 10) % 10 == 0:
                for uid, a in agents.items():
                    if uid == SINK_ID or not a.buffer:
                        continue
                    a.buffer = [m for m in a.buffer if m.msg_id not in sink.seen_msgs]

            # Sensor → UAV upload (ZigBee)
            for s, src_pos in enumerate(iot_nodes):
                if not sensor_queues[s]:
                    continue
                best_uav = min([k for k in range(NUM_UAVS) if k != SINK_ID],
                               key=lambda k: np.linalg.norm(agents[k].pos - src_pos))
                rate, _, prof = link_rate(src_pos, agents[best_uav].pos, is_ground_to_uav=True)
                msg_id, qos_val, t0 = sensor_queues[s][0]

                if rate < prof.B:
                    if qos_val == 0:
                        sensor_queues[s].pop(0)
                    continue

                frame_bytes = mqtt_frame_size_zigbee(PhyConst.SENSOR_PAYLOAD_BYTES, qos_val)
                if frame_bytes > (rate * dt) / 8.0:
                    if qos_val == 0:
                        sensor_queues[s].pop(0)
                    continue

                bits = frame_bytes * 8
                sensor_tx_energy                     += bits * prof.E_tx_per_bit
                agents[best_uav].energy              -= bits * prof.E_rx_per_bit
                agents[best_uav].radio_rx_energy     += bits * prof.E_rx_per_bit

                if qos_val == 1:
                    pb_bits = mqtt_puback_size_zigbee() * 8
                    agents[best_uav].energy          -= pb_bits * prof.E_tx_per_bit
                    agents[best_uav].radio_tx_energy += pb_bits * prof.E_tx_per_bit
                    sensor_rx_energy                 += pb_bits * prof.E_rx_per_bit

                if len(agents[best_uav].buffer) < agents[best_uav].MAX_BUFFER:
                    sensor_queues[s].pop(0)
                    agents[best_uav].buffer.append(SprayMessage(
                        msg_id=msg_id, source_id=s, creation_time=t0,
                        hop_count=0, tokens=INITIAL_TOKENS, payload=b"SENSOR_DATA",
                        qos=qos_val, payload_bytes=PhyConst.SENSOR_PAYLOAD_BYTES))
                    agents[best_uav].seen_msgs.add(msg_id)
                elif qos_val == 0:
                    sensor_queues[s].pop(0)

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")

    # --- Final report ---
    total_uav_tx    = sum(a.radio_tx_energy for a in agents.values())
    total_uav_rx    = sum(a.radio_rx_energy for a in agents.values())
    total_uav_radio = total_uav_tx + total_uav_rx

    total_control_energy    = control_tx_energy + control_rx_energy
    total_data_wifi_energy  = data_wifi_tx_energy + data_wifi_rx_energy
    total_sensor_zigbee     = sensor_tx_energy + sensor_rx_energy
    grand_total             = total_control_energy + total_data_wifi_energy + total_sensor_zigbee

    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    if total_generated > 0:
        pdr = 100.0 * total_delivered / total_generated
        print(f"\n  PDR:                    {pdr:.2f}%")
    if latencies:
        print(f"  Avg Latency:            {np.mean(latencies):.3f} s")
        print(f"  Median Latency:         {np.median(latencies):.3f} s")
    if hop_counts:
        print(f"  Avg Hops:               {np.mean(hop_counts):.2f}")
    if total_delivered > 0:
        print(f"  Overhead Factor:        {(uav_relay_events + total_delivered) / total_delivered:.2f}x")
        print(f"  Energy/Msg:             {(total_uav_radio / total_delivered) * 1000:.4f} mJ")
    print(f"\n  UAV↔UAV Relays:         {uav_relay_events} (spray={spray_events}, focus={focus_events})")
    print(f"  UAV→Sink Deliveries:    {sink_delivery_events}")
    print(f"  Control Messages:       {control_messages_sent}")
    if grand_total > 0:
        print(f"\n  Energy Distribution:")
        print(f"    Control (WiFi):       {100*total_control_energy/grand_total:.1f}%")
        print(f"    Data (WiFi):          {100*total_data_wifi_energy/grand_total:.1f}%")
        print(f"    Data (ZigBee):        {100*total_sensor_zigbee/grand_total:.1f}%")
    print("\n" + "=" * 70)


# ==========================================
# BATCH SIMULATION RUNNER
# ==========================================

def run_simulation(config: dict, verbose: bool = False) -> dict:
    """Run simulation with the given configuration and return result metrics."""
    global NUM_UAVS, NUM_SENSORS, INITIAL_TOKENS, GLOBAL_QOS, SINK_ID

    NUM_UAVS       = config.get("NUM_UAVS",       8)
    NUM_SENSORS    = config.get("NUM_SENSORS",     6)
    INITIAL_TOKENS = config.get("INITIAL_TOKENS", 10)
    GLOBAL_QOS     = config.get("GLOBAL_QOS",      1)
    AREA_SIZE      = config.get("AREA_SIZE",      750)
    SINK_MOBILE    = config.get("SINK_MOBILE",    True)
    BUFFER_POLICY  = config.get("BUFFER_POLICY", "SMART")
    duration       = config.get("DURATION",     1500.0)
    dt             = 0.1

    if "MAX_BUFFER" in config:
        MqttUAVAgent.MAX_BUFFER = config["MAX_BUFFER"]
    if "WIFI_PAYLOAD_BYTES" in config:
        PhyConst.WIFI_DATA_PAYLOAD_BYTES = config["WIFI_PAYLOAD_BYTES"]

    SINK_ID  = 0
    SINK_IDS = [SINK_ID]

    agents: Dict[int, MqttUAVAgent] = {}
    sink_pos = ([random.uniform(100, AREA_SIZE - 100),
                 random.uniform(100, AREA_SIZE - 100), PhyConst.H]
                if SINK_MOBILE else [AREA_SIZE / 2, AREA_SIZE / 2, PhyConst.H])
    agents[SINK_ID] = MqttUAVAgent(SINK_ID, sink_pos, is_sink=not SINK_MOBILE, area_size=AREA_SIZE)
    for i in range(1, NUM_UAVS):
        agents[i] = MqttUAVAgent(i, [random.uniform(50, AREA_SIZE - 50),
                                     random.uniform(50, AREA_SIZE - 50), PhyConst.H],
                                 area_size=AREA_SIZE)

    initial_energy = {uid: a.energy for uid, a in agents.items()}

    def generate_spread_sensors(num_sensors, area_size, seed=42):
        rng      = np.random.RandomState(seed)
        margin   = 100
        min_dist = (area_size - 2 * margin) / (num_sensors ** 0.5 + 1)
        positions = []
        for _ in range(num_sensors):
            for attempt in range(1000):
                x, y = rng.uniform(margin, area_size - margin), rng.uniform(margin, area_size - margin)
                if all(np.sqrt((x - px) ** 2 + (y - py) ** 2) >= min_dist
                       for px, py, _ in positions) or attempt == 999:
                    positions.append([x, y, 10.0])
                    break
        return [np.array(p) for p in positions]

    iot_nodes = generate_spread_sensors(NUM_SENSORS, AREA_SIZE, seed=42)

    sim_time    = 0.0
    SENSOR_RATE = 2.0
    MSG_COUNTER = 0
    sensor_queues: List[List[Tuple[int, int, float]]] = [[] for _ in range(NUM_SENSORS)]

    total_generated = total_delivered = 0
    uav_relay_events = spray_events = focus_events = sink_delivery_events = 0
    control_messages_sent = 0
    direct_deliveries = relayed_deliveries = 0
    latencies: List[float] = []
    hop_counts: List[int]  = []

    sensor_tx_energy = sensor_rx_energy = 0.0
    control_tx_energy = control_rx_energy = 0.0
    data_wifi_tx_energy = data_wifi_rx_energy = 0.0

    while sim_time < duration:
        sim_time += dt

        # Sensor data generation
        for s in range(NUM_SENSORS):
            if random.random() < SENSOR_RATE * dt:
                MSG_COUNTER += 1
                sensor_queues[s].append((MSG_COUNTER, GLOBAL_QOS, sim_time))
                if len(sensor_queues[s]) > 50:
                    sensor_queues[s].pop(0)
                total_generated += 1

        # UAV movement
        for uid, agent in agents.items():
            if uid in SINK_IDS and not SINK_MOBILE:
                continue
            agent.move(dt)

        # Encounter detection + distance-vector utility update
        for i in range(NUM_UAVS):
            for j in range(i + 1, NUM_UAVS):
                rate, d, _ = link_rate(agents[i].pos, agents[j].pos, is_ground_to_uav=False)
                if rate >= WIFI.B:
                    agents[i].encounter_timers[j] = 0.0
                    agents[j].encounter_timers[i] = 0.0
                    t_meet   = d / 20.0
                    t_i_sink = min(agents[i].encounter_timers.get(sid, 9999.0) for sid in SINK_IDS)
                    t_j_sink = min(agents[j].encounter_timers.get(sid, 9999.0) for sid in SINK_IDS)
                    for sid in SINK_IDS:
                        if t_j_sink + t_meet < agents[i].encounter_timers.get(sid, 9999.0):
                            agents[i].encounter_timers[sid] = t_j_sink + t_meet
                        if t_i_sink + t_meet < agents[j].encounter_timers.get(sid, 9999.0):
                            agents[j].encounter_timers[sid] = t_i_sink + t_meet
        for i in range(NUM_UAVS):
            for sid in SINK_IDS:
                rate, _, _ = link_rate(agents[i].pos, agents[sid].pos, is_ground_to_uav=False)
                if rate >= WIFI.B:
                    agents[i].encounter_timers[sid] = 0.0

        # UAV → Sink delivery (runs BEFORE S&F so freshly arrived messages go out immediately)
        all_delivered_ids: set = set()
        for sid in SINK_IDS:
            all_delivered_ids.update(agents[sid].seen_msgs)

        for i in range(NUM_UAVS):
            if i in SINK_IDS:
                continue
            ai = agents[i]
            if not ai.buffer:
                continue
            for sid in SINK_IDS:
                sink     = agents[sid]
                rate_bps, _, prof = link_rate(ai.pos, sink.pos, is_ground_to_uav=False)
                if rate_bps < prof.B:
                    continue
                max_bytes      = (rate_bps * dt) / 8.0
                bytes_sent     = 0
                msgs_to_remove = []
                for msg in ai.buffer:
                    if msg.msg_id in all_delivered_ids:
                        msgs_to_remove.append(msg);  continue
                    frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                    if bytes_sent + frame_bytes > max_bytes:
                        break
                    ok, tx_e, rx_e, _ = MQTTTransmission.transmit_data(rate_bps, prof, msg)
                    if not ok:
                        continue
                    ai.energy -= tx_e;    ai.radio_tx_energy += tx_e;    data_wifi_tx_energy += tx_e
                    sink.energy -= rx_e;  sink.radio_rx_energy += rx_e;  data_wifi_rx_energy += rx_e
                    if msg.qos == 1:
                        s_ack, ta, ra, _ = MQTTTransmission.transmit_puback(rate_bps, prof)
                        if s_ack:
                            sink.energy -= ta;  sink.radio_tx_energy += ta;  data_wifi_tx_energy += ta
                            ai.energy -= ra;    ai.radio_rx_energy += ra;    data_wifi_rx_energy += ra
                    sink.seen_msgs.add(msg.msg_id)
                    all_delivered_ids.add(msg.msg_id)
                    total_delivered += 1;  sink_delivery_events += 1
                    hop_counts.append(msg.hop_count)
                    latencies.append(sim_time - msg.creation_time)
                    bytes_sent += frame_bytes;  msgs_to_remove.append(msg)
                    (relayed_deliveries if msg.hop_count > 0 else direct_deliveries)
                for m in msgs_to_remove:
                    if m in ai.buffer:
                        ai.buffer.remove(m)

        # Periodic duplicate purge
        if total_delivered > 0 and int(sim_time * 10) % 10 == 0:
            for uid, a in agents.items():
                if uid in SINK_IDS or not a.buffer:
                    continue
                a.buffer = [m for m in a.buffer if m.msg_id not in all_delivered_ids]

        # Spray & Focus routing (UAV ↔ UAV, runs AFTER sink delivery)
        for i in range(NUM_UAVS):
            if i in SINK_IDS:
                continue
            ai = agents[i]
            if not ai.buffer:
                continue
            for j in range(NUM_UAVS):
                if j == i or j in SINK_IDS:
                    continue
                aj = agents[j]
                rate, d, prof = link_rate(ai.pos, aj.pos, is_ground_to_uav=False)
                if rate < prof.B:
                    continue

                max_bytes       = (rate * dt) / 8.0
                bytes_this_step = 0
                spray_messages  = [m for m in ai.buffer if m.tokens > 1]
                focus_messages  = [m for m in ai.buffer if m.tokens == 1]

                # --- Spray ---
                for msg in spray_messages:
                    if msg.msg_id in aj.seen_msgs or len(aj.buffer) >= aj.MAX_BUFFER:
                        continue
                    send_tokens = msg.tokens // 2
                    if send_tokens <= 0:
                        continue
                    frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                    if bytes_this_step + frame_bytes > max_bytes:
                        break
                    ok, tx_e, rx_e, db = MQTTTransmission.transmit_data(rate, prof, msg)
                    if not ok:
                        continue
                    ai.energy -= tx_e;  ai.radio_tx_energy += tx_e;  data_wifi_tx_energy += tx_e
                    aj.energy -= rx_e;  aj.radio_rx_energy += rx_e;  data_wifi_rx_energy += rx_e
                    bytes_this_step += db
                    if msg.qos == 1:
                        s_ack, ta, ra, _ = MQTTTransmission.transmit_puback(rate, prof)
                        if s_ack:
                            aj.energy -= ta;  aj.radio_tx_energy += ta;  data_wifi_tx_energy += ta
                            ai.energy -= ra;  ai.radio_rx_energy += ra;  data_wifi_rx_energy += ra
                    aj.buffer.append(SprayMessage(
                        msg_id=msg.msg_id, source_id=msg.source_id,
                        creation_time=msg.creation_time, hop_count=msg.hop_count + 1,
                        tokens=send_tokens, payload=msg.payload, qos=msg.qos,
                        payload_bytes=msg.payload_bytes))
                    aj.seen_msgs.add(msg.msg_id)
                    msg.tokens -= send_tokens
                    uav_relay_events += 1;  spray_events += 1

                # --- Focus ---
                if not focus_messages:
                    continue
                inquiry_msg      = ai.get_utility_message()
                total_ctrl_bytes = 2 * mqtt_frame_size(inquiry_msg.payload_size(), 1) + 2 * mqtt_puback_size()
                if bytes_this_step + total_ctrl_bytes > max_bytes:
                    continue

                ok, tx_e, rx_e, cb = MQTTTransmission.transmit_control(rate, prof, inquiry_msg)
                if not ok:
                    continue
                ai.energy -= tx_e;  ai.radio_tx_energy += tx_e;  control_tx_energy += tx_e
                aj.energy -= rx_e;  aj.radio_rx_energy += rx_e;  control_rx_energy += rx_e
                control_messages_sent += 1;  bytes_this_step += cb

                s_ack, ta, ra, ab = MQTTTransmission.transmit_puback(rate, prof)
                if s_ack:
                    aj.energy -= ta;  aj.radio_tx_energy += ta;  control_tx_energy += ta
                    ai.energy -= ra;  ai.radio_rx_energy += ra;  control_rx_energy += ra
                    bytes_this_step += ab

                response_msg = aj.get_utility_message()
                ok2, tx_e2, rx_e2, cb2 = MQTTTransmission.transmit_control(rate, prof, response_msg)
                if not ok2:
                    continue
                aj.energy -= tx_e2;  aj.radio_tx_energy += tx_e2;  control_tx_energy += tx_e2
                ai.energy -= rx_e2;  ai.radio_rx_energy += rx_e2;  control_rx_energy += rx_e2
                control_messages_sent += 1;  bytes_this_step += cb2

                s_ack2, ta2, ra2, ab2 = MQTTTransmission.transmit_puback(rate, prof)
                if s_ack2:
                    ai.energy -= ta2;  ai.radio_tx_energy += ta2;  control_tx_energy += ta2
                    aj.energy -= ra2;  aj.radio_rx_energy += ra2;  control_rx_energy += ra2
                    bytes_this_step += ab2

                my_utility = min(ai.encounter_timers.get(sid, 9999.0) for sid in SINK_IDS)
                nb_utility = response_msg.utility_to_sink
                t_meet     = d / 20.0

                for msg in focus_messages:
                    if (nb_utility - t_meet) >= my_utility:
                        continue
                    if msg.msg_id in aj.seen_msgs or len(aj.buffer) >= aj.MAX_BUFFER:
                        continue
                    frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                    if bytes_this_step + frame_bytes > max_bytes:
                        break
                    ok3, tx_e3, rx_e3, db3 = MQTTTransmission.transmit_data(rate, prof, msg)
                    if not ok3:
                        continue
                    ai.energy -= tx_e3;  ai.radio_tx_energy += tx_e3;  data_wifi_tx_energy += tx_e3
                    aj.energy -= rx_e3;  aj.radio_rx_energy += rx_e3;  data_wifi_rx_energy += rx_e3
                    bytes_this_step += db3
                    if msg.qos == 1:
                        s_ack3, ta3, ra3, _ = MQTTTransmission.transmit_puback(rate, prof)
                        if s_ack3:
                            aj.energy -= ta3;  aj.radio_tx_energy += ta3;  data_wifi_tx_energy += ta3
                            ai.energy -= ra3;  ai.radio_rx_energy += ra3;  data_wifi_rx_energy += ra3
                    aj.buffer.append(SprayMessage(
                        msg_id=msg.msg_id, source_id=msg.source_id,
                        creation_time=msg.creation_time, hop_count=msg.hop_count + 1,
                        tokens=1, payload=msg.payload, qos=msg.qos,
                        payload_bytes=msg.payload_bytes))
                    aj.seen_msgs.add(msg.msg_id)
                    if msg in ai.buffer:
                        ai.buffer.remove(msg)
                    uav_relay_events += 1;  focus_events += 1

        # Sensor → UAV upload (ZigBee, sink excluded)
        for s, src_pos in enumerate(iot_nodes):
            if not sensor_queues[s]:
                continue
            best_uav = min([k for k in range(NUM_UAVS) if k not in SINK_IDS],
                           key=lambda k: np.linalg.norm(agents[k].pos - src_pos))
            rate, _, prof = link_rate(src_pos, agents[best_uav].pos, is_ground_to_uav=True)
            msg_id, qos_val, t0 = sensor_queues[s][0]

            if rate < prof.B:
                if qos_val == 0:
                    sensor_queues[s].pop(0)
                continue

            temp_msg = SprayMessage(
                msg_id=msg_id, source_id=s, creation_time=t0, hop_count=0,
                tokens=INITIAL_TOKENS, payload=b"SENSOR_DATA", qos=qos_val,
                payload_bytes=PhyConst.SENSOR_PAYLOAD_BYTES)

            frame_bytes = mqtt_frame_size_zigbee(temp_msg.payload_size(), qos_val)
            if frame_bytes > (rate * dt) / 8.0:
                if qos_val == 0:
                    sensor_queues[s].pop(0)
                continue

            ok, tx_e, rx_e, _ = MQTTTransmission.transmit_data(rate, prof, temp_msg)
            if not ok:
                if qos_val == 0:
                    sensor_queues[s].pop(0)
                continue

            sensor_tx_energy                     += tx_e
            agents[best_uav].energy              -= rx_e
            agents[best_uav].radio_rx_energy     += rx_e

            if qos_val == 1:
                s_ack, ta, ra, _ = MQTTTransmission.transmit_puback(rate, prof)
                if s_ack:
                    agents[best_uav].energy          -= ta
                    agents[best_uav].radio_tx_energy += ta
                    sensor_rx_energy                 += ra

            sensor_queues[s].pop(0)

            if len(agents[best_uav].buffer) < agents[best_uav].MAX_BUFFER:
                agents[best_uav].buffer.append(temp_msg)
                agents[best_uav].seen_msgs.add(msg_id)
            else:
                if BUFFER_POLICY == "SMART":
                    for idx, victim in enumerate(agents[best_uav].buffer):
                        if victim.hop_count > 0:
                            agents[best_uav].buffer.pop(idx)
                            agents[best_uav].buffer.append(temp_msg)
                            agents[best_uav].seen_msgs.add(msg_id)
                            break
                elif BUFFER_POLICY == "FIFO":
                    agents[best_uav].buffer.pop(0)
                    agents[best_uav].buffer.append(temp_msg)
                    agents[best_uav].seen_msgs.add(msg_id)

    # --- Results ---
    total_uav_radio        = sum(a.radio_tx_energy + a.radio_rx_energy for a in agents.values())
    total_control_energy   = control_tx_energy + control_rx_energy
    total_data_wifi_energy = data_wifi_tx_energy + data_wifi_rx_energy
    total_data_zigbee      = sensor_tx_energy + sensor_rx_energy
    relayed_hop_counts     = [h for h in hop_counts if h > 0]

    results = {
        "pdr":                    100.0 * total_delivered / max(1, total_generated),
        "avg_latency":            float(np.mean(latencies))           if latencies           else 0.0,
        "median_latency":         float(np.median(latencies))         if latencies           else 0.0,
        "avg_hops":               float(np.mean(hop_counts))          if hop_counts          else 0.0,
        "avg_hops_relayed":       float(np.mean(relayed_hop_counts))  if relayed_hop_counts  else 0.0,
        "direct_deliveries":      direct_deliveries,
        "relayed_deliveries":     relayed_deliveries,
        "direct_delivery_ratio":  100.0 * direct_deliveries / max(1, total_delivered),
        "overhead_factor":        (uav_relay_events + total_delivered) / max(1, total_delivered),
        "total_generated":        total_generated,
        "total_delivered":        total_delivered,
        "uav_relay_events":       uav_relay_events,
        "spray_events":           spray_events,
        "focus_events":           focus_events,
        "sink_delivery_events":   sink_delivery_events,
        "control_messages_sent":  control_messages_sent,
        "control_energy":         total_control_energy,
        "data_wifi_energy":       total_data_wifi_energy,
        "data_zigbee_energy":     total_data_zigbee,
        "total_uav_radio_energy": total_uav_radio,
        "energy_per_msg_mJ":      (total_uav_radio / max(1, total_delivered)) * 1000,
        "sink_flight_energy_kJ":  (
            initial_energy[SINK_ID]
            - agents[SINK_ID].energy
            - agents[SINK_ID].radio_tx_energy
            - agents[SINK_ID].radio_rx_energy
        ) / 1000.0,
        "total_system_energy_kJ": sum(
            initial_energy[uid] - agents[uid].energy for uid in agents
        ) / 1000.0,
    }

    if verbose:
        print(f"  PDR: {results['pdr']:.1f}% | Latency: {results['avg_latency']:.2f}s | "
              f"Hops: {results['avg_hops']:.2f}")
    return results


if __name__ == "__main__":
    run_mqtt_simulation()
