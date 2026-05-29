"""
BASELINE MQTT Simulation (No DTN Routing)

Simulates standard MQTT over a mobile ad-hoc UAV network without DTN enhancement.
UAVs collect sensor data and deliver it directly to the sink — no UAV-to-UAV relay.
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ==========================================
# GLOBAL CONFIG
# ==========================================

GLOBAL_QOS      = 1
NUM_UAVS        = 6
SINK_ID         = 0
NUM_SENSORS     = 10
SINK_MOBILITY_FRACTION = 0.40

MQTT_TOPIC_LEN  = 16
TCP_IP_OVERHEAD = 40
L2_OVERHEAD     = 30

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

    SENSOR_PAYLOAD_BYTES   = 64
    WIFI_DATA_PAYLOAD_BYTES = 256
    WIFI_MAX_RATE          = 54_000_000.0
    CONTROL_PAYLOAD_BYTES  = 80


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


def calibrate_beta0(prof: PHYProfile, target_snr_dB: float = 0.0,
                    ref_dist: float = REF_DIST):
    prof.beta0 = prof.N0 * (REF_DIST ** 2) / prof.P_tx


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


def link_rate(pos1, pos2, is_ground_to_uav: bool):
    d    = float(np.linalg.norm(np.array(pos1) - np.array(pos2)))
    prof = ZIGBEE if is_ground_to_uav else WIFI
    return shannon_rate_3d(d, prof), d, prof


calibrate_beta0(ZIGBEE)
calibrate_beta0(WIFI)


# ==========================================
# MQTT FRAME SIZES
# ==========================================

ZIGBEE_L2_OVERHEAD = 15


def mqtt_frame_size_wifi(payload_bytes: int, qos: int) -> int:
    header = 2 + 2 + MQTT_TOPIC_LEN + (2 if qos > 0 else 0)
    return TCP_IP_OVERHEAD + L2_OVERHEAD + header + payload_bytes


def mqtt_frame_size_zigbee(payload_bytes: int, qos: int) -> int:
    header = 2 + 2 + MQTT_TOPIC_LEN + (2 if qos > 0 else 0)
    return ZIGBEE_L2_OVERHEAD + header + payload_bytes


def mqtt_puback_size_wifi() -> int:
    return TCP_IP_OVERHEAD + L2_OVERHEAD + 4


def mqtt_puback_size_zigbee() -> int:
    return ZIGBEE_L2_OVERHEAD + 4


def mqtt_frame_size(payload_bytes: int, qos: int) -> int:
    return mqtt_frame_size_wifi(payload_bytes, qos)


def mqtt_puback_size() -> int:
    return mqtt_puback_size_wifi()


# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class BaselineMessage:
    msg_id:        int
    source_id:     int
    creation_time: float
    hop_count:     int
    payload:       bytes
    qos:           int
    payload_bytes: int = PhyConst.WIFI_DATA_PAYLOAD_BYTES

    def payload_size(self) -> int:
        return self.payload_bytes


class MQTTTransmission:
    @staticmethod
    def transmit_data(rate_bps: float, prof: PHYProfile, msg):
        if rate_bps < prof.B:
            return False, 0.0, 0.0, 0

        if prof.name == "zigbee":
            frame_bytes = mqtt_frame_size_zigbee(msg.payload_size(), msg.qos)
            if frame_bytes > ZIGBEE_MAX_FRAME:
                return False, 0.0, 0.0, 0
        else:
            frame_bytes = mqtt_frame_size_wifi(msg.payload_size(), msg.qos)

        bits      = frame_bytes * 8
        tx_energy = bits * prof.E_tx_per_bit
        rx_energy = bits * prof.E_rx_per_bit
        return True, tx_energy, rx_energy, frame_bytes

    @staticmethod
    def transmit_puback(rate_bps: float, prof: PHYProfile):
        if rate_bps < prof.B:
            return False, 0.0, 0.0, 0

        ack_bytes = (mqtt_puback_size_zigbee() if prof.name == "zigbee"
                     else mqtt_puback_size_wifi())
        bits      = ack_bytes * 8
        tx_energy = bits * prof.E_tx_per_bit
        rx_energy = bits * prof.E_rx_per_bit
        return True, tx_energy, rx_energy, ack_bytes


# ==========================================
# UAV AGENT
# ==========================================

class BaselineUAVAgent:
    MAX_BUFFER = 250

    def __init__(self, uid: int, pos: List[float], is_sink: bool = False,
                 area_size: float = 500):
        self.id              = uid
        self.pos             = np.array(pos, dtype=float)
        self.energy          = 300_000.0
        self.is_sink         = is_sink
        self.area_size       = area_size
        self.buffer:         List[BaselineMessage] = []
        self.seen_msgs       = set()
        self.radio_tx_energy = 0.0
        self.radio_rx_energy = 0.0
        self.sink_received_count = 0
        self.vel             = np.array([0.0, 0.0, 0.0])
        self._waypoint_timer = 0.0

    def move(self, dt: float):
        if not hasattr(self, 'waypoints') or not self.waypoints:
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
        step   = speed * dt

        if dist <= step:
            self.pos = target.copy()
            self.waypoints.pop(0)
        else:
            self.pos += (diff / dist) * step

        self.energy -= self.flight_power(speed) * dt

    def flight_power(self, velocity: float) -> float:
        term1 = PhyConst.P_C * (1 + (3 * velocity ** 2) / (PhyConst.U_TIP ** 2))
        term2 = PhyConst.WEIGHT * (
            math.sqrt(1 + (velocity ** 4) / (4 * PhyConst.V_0 ** 4))
            - (velocity ** 2) / (2 * PhyConst.V_0 ** 2)
        )
        term3 = 0.5 * PhyConst.D_0 * PhyConst.RHO * PhyConst.S_SOL * PhyConst.AREA * (velocity ** 3)
        return term1 + term2 + term3


# ==========================================
# BASELINE SIMULATION
# ==========================================

def run_baseline_simulation(config: dict, verbose: bool = False) -> dict:
    """
    Direct-delivery only (no DTN routing).
    Messages are buffered on the collecting UAV until it reaches the sink.
    """
    global NUM_UAVS, NUM_SENSORS, GLOBAL_QOS, SINK_ID

    NUM_UAVS    = config.get("NUM_UAVS",    8)
    NUM_SENSORS = config.get("NUM_SENSORS", 6)
    GLOBAL_QOS  = config.get("GLOBAL_QOS",  1)
    SINK_ID     = 0

    AREA_SIZE   = config.get("AREA_SIZE",  750)
    SINK_MOBILE = config.get("SINK_MOBILE", True)
    duration    = config.get("DURATION",  3000.0)
    dt          = 0.1

    if "WIFI_PAYLOAD_BYTES" in config:
        PhyConst.WIFI_DATA_PAYLOAD_BYTES = config["WIFI_PAYLOAD_BYTES"]
    if "MAX_BUFFER" in config:
        BaselineUAVAgent.MAX_BUFFER = config["MAX_BUFFER"]

    # --- Initialise agents ---
    agents: Dict[int, BaselineUAVAgent] = {}
    sink_pos = ([random.uniform(100, AREA_SIZE - 100),
                 random.uniform(100, AREA_SIZE - 100), PhyConst.H]
                if SINK_MOBILE else [AREA_SIZE / 2, AREA_SIZE / 2, PhyConst.H])
    agents[SINK_ID] = BaselineUAVAgent(SINK_ID, sink_pos,
                                       is_sink=not SINK_MOBILE, area_size=AREA_SIZE)
    for i in range(1, NUM_UAVS):
        agents[i] = BaselineUAVAgent(
            i, [random.uniform(100, AREA_SIZE - 100),
                random.uniform(100, AREA_SIZE - 100), PhyConst.H],
            area_size=AREA_SIZE)

    initial_energy = {uid: agent.energy for uid, agent in agents.items()}

    # --- Sensor positions ---
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

    # --- State ---
    sim_time    = 0.0
    SENSOR_RATE = 2.0
    MSG_COUNTER = 0
    sensor_queues: List[List[Tuple[int, int, float]]] = [[] for _ in range(NUM_SENSORS)]

    total_generated = total_delivered = sink_delivery_events = 0
    latencies: List[float] = []
    hop_counts: List[int]  = []

    sensor_tx_energy = sensor_rx_energy = 0.0
    data_wifi_tx_energy = data_wifi_rx_energy = 0.0

    # --- Main loop ---
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
        for agent in agents.values():
            agent.move(dt)

        # Direct delivery: UAV → Sink (no relay)
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
                    msgs_to_remove.append(msg)
                    continue

                frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                if bytes_sent + frame_bytes > max_bytes:
                    break

                success, tx_e, rx_e, _ = MQTTTransmission.transmit_data(rate_bps, prof, msg)
                if not success:
                    continue

                ai.energy   -= tx_e;  ai.radio_tx_energy   += tx_e;  data_wifi_tx_energy += tx_e
                sink.energy -= rx_e;  sink.radio_rx_energy += rx_e;  data_wifi_rx_energy += rx_e

                if msg.qos == 1:
                    succ_ack, tx_ack, rx_ack, _ = MQTTTransmission.transmit_puback(rate_bps, prof)
                    if succ_ack:
                        sink.energy -= tx_ack;  sink.radio_tx_energy += tx_ack;  data_wifi_tx_energy += tx_ack
                        ai.energy   -= rx_ack;  ai.radio_rx_energy   += rx_ack;  data_wifi_rx_energy += rx_ack

                sink.seen_msgs.add(msg.msg_id)
                total_delivered      += 1
                sink_delivery_events += 1
                hop_counts.append(msg.hop_count)
                latencies.append(sim_time - msg.creation_time)
                bytes_sent += frame_bytes
                msgs_to_remove.append(msg)

            for m in msgs_to_remove:
                if m in ai.buffer:
                    ai.buffer.remove(m)

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

            frame_bytes    = mqtt_frame_size_zigbee(PhyConst.SENSOR_PAYLOAD_BYTES, qos_val)
            max_bytes      = (rate * dt) / 8.0
            if frame_bytes > max_bytes:
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
                agents[best_uav].buffer.append(BaselineMessage(
                    msg_id=msg_id, source_id=s, creation_time=t0,
                    hop_count=0, payload=b"SENSOR_DATA", qos=qos_val,
                    payload_bytes=PhyConst.SENSOR_PAYLOAD_BYTES))
                agents[best_uav].seen_msgs.add(msg_id)
            elif qos_val == 0:
                sensor_queues[s].pop(0)

    # --- Results ---
    total_uav_radio = sum(a.radio_tx_energy + a.radio_rx_energy for a in agents.values())

    return {
        "pdr":                    100.0 * total_delivered / max(1, total_generated),
        "avg_latency":            float(np.mean(latencies))   if latencies   else 0.0,
        "median_latency":         float(np.median(latencies)) if latencies   else 0.0,
        "avg_hops":               float(np.mean(hop_counts))  if hop_counts  else 0.0,
        "overhead_factor":        1.0,
        "total_generated":        total_generated,
        "total_delivered":        total_delivered,
        "uav_relay_events":       0,
        "sink_delivery_events":   sink_delivery_events,
        "control_messages_sent":  0,
        "control_energy":         0.0,
        "data_wifi_energy":       data_wifi_tx_energy + data_wifi_rx_energy,
        "data_zigbee_energy":     sensor_tx_energy + sensor_rx_energy,
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


if __name__ == "__main__":
    config = {"NUM_UAVS": 8, "NUM_SENSORS": 6, "GLOBAL_QOS": 1}
    result = run_baseline_simulation(config, verbose=True)
    print(f"\nBaseline MQTT Results:")
    print(f"  PDR:          {result['pdr']:.2f}%")
    print(f"  Avg Latency:  {result['avg_latency']:.2f}s")
    print(f"  Energy/Msg:   {result['energy_per_msg_mJ']:.2f} mJ")
