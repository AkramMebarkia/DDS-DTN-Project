"""
DDS Spray & Focus Simulation

Simulation using simulated DDS middleware with Spray & Focus DTN routing.
Direct sensor → UAV → sink delivery with multi-hop relay via S&F.
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ==========================================
# 1) GLOBAL CONFIG
# ==========================================

GLOBAL_QOS = 1  # 0 = Best Effort, 1 = Reliable
NUM_UAVS = 6
SINK_ID = 0
NUM_SENSORS = 10
INITIAL_TOKENS = 8
SINK_MOBILITY_FRACTION = 0.40

BATCH_ENABLE = True
BATCH_MAX_SAMPLES = 5
BATCH_MAX_DATA_BYTES = 1024
BATCH_FLUSH_DELAY = 0.1

ZIGBEE_MAX_FRAME = 127
ZIGBEE_MAX_PAYLOAD = 100


# ==========================================
# 2) DDS OVERHEAD CONSTANTS
# ==========================================

WIFI_UDP_HEADER = 8
WIFI_IP_HEADER = 20
WIFI_L2_OVERHEAD = 30
WIFI_TRANSPORT_OVERHEAD = WIFI_UDP_HEADER + WIFI_IP_HEADER + WIFI_L2_OVERHEAD  # 58 bytes

ZIGBEE_L2_OVERHEAD = 15
ZIGBEE_TRANSPORT_OVERHEAD = ZIGBEE_L2_OVERHEAD  # 15 bytes

RTPS_MESSAGE_HEADER = 20
RTPS_DATA_SUBMSG_HEADER = 24
RTPS_ACKNACK_SUBMSG = 28


# ==========================================
# 3) PHYSICS / RADIO PROFILES
# ==========================================

class PhyConst:
    # UAV altitude is 90 m, not 80 m as written in the paper. IoT sensors sit at
    # z = 10 m above ground, so the effective air-to-ground separation is 90 - 10 = 80 m,
    # which matches the 80 m figure cited in the paper.
    H = 90.0

    P_C = 5.0
    U_TIP = 120.0
    V_0 = 4.03
    D_0 = 0.6
    RHO = 1.225
    AREA = 0.503
    OMEGA = 300.0
    R_RAD = 0.4
    DELTA = 0.012
    S_SOL = 0.05
    WEIGHT = 20.0

    SENSOR_PAYLOAD_BYTES = 64
    WIFI_DATA_PAYLOAD_BYTES = 256
    WIFI_MAX_RATE = 54_000_000.0
    CONTROL_PAYLOAD_BYTES = 12


class PHYProfile:
    def __init__(self, name: str, B: float, P_tx: float, N0: float,
                 E_tx_per_bit: float, E_rx_per_bit: float):
        self.name = name
        self.B = B
        self.P_tx = P_tx
        self.N0 = N0
        self.beta0 = None
        self.E_tx_per_bit = E_tx_per_bit
        self.E_rx_per_bit = E_rx_per_bit


# E_tx_per_bit already accounts for the full end-to-end energy budget measured in
# Siekkinen et al. (IEEE WCNC 2012), which bundles both transmission and reception costs
# into a single per-bit figure. Setting E_rx_per_bit = 0 prevents double-counting.
# The two fields are kept separate to preserve the existing energy-accounting interface
# without breaking call sites that reference both attributes.
ZIGBEE = PHYProfile("zigbee", B=250_000.0, P_tx=0.0774, N0=1e-13,
                    E_tx_per_bit=1e-6, E_rx_per_bit=0)

# Same rationale applies here: E_tx_per_bit = 2e-7 J/bit is the measured end-to-end
# figure from Liu & Choi (ACM SIGMETRICS 2023); E_rx_per_bit = 0 avoids double-counting.
WIFI = PHYProfile("wifi", B=20_000_000.0, P_tx=1.5, N0=1e-13,
                  E_tx_per_bit=2e-7, E_rx_per_bit=0)

REF_DIST = 100.0


def calibrate_beta0(prof: PHYProfile):
    prof.beta0 = prof.N0 * (REF_DIST ** 2) / prof.P_tx


for _p in (ZIGBEE, WIFI):
    calibrate_beta0(_p)


def shannon_rate_3d(dist_3d: float, profile: PHYProfile) -> float:
    if dist_3d <= 0.0:
        dist_3d = 1e-3
    snr = (profile.beta0 * profile.P_tx) / (profile.N0 * (dist_3d ** 2))
    rate = profile.B * math.log2(1.0 + snr)
    if profile.name == "zigbee":
        rate = min(rate, 250_000.0)
    elif profile.name == "wifi":
        rate = min(rate, PhyConst.WIFI_MAX_RATE)
    return rate


def link_rate(pos1, pos2, is_ground_to_uav: bool) -> Tuple[float, float, PHYProfile]:
    d = float(np.linalg.norm(np.array(pos1) - np.array(pos2)))
    prof = ZIGBEE if is_ground_to_uav else WIFI
    return shannon_rate_3d(d, prof), d, prof


# ==========================================
# 4) DDS FRAME SIZE FUNCTIONS
# ==========================================

def dds_frame_size_wifi(payload_bytes: int) -> int:
    return WIFI_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER + RTPS_DATA_SUBMSG_HEADER + payload_bytes


def dds_frame_size_zigbee(payload_bytes: int) -> int:
    return ZIGBEE_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER + RTPS_DATA_SUBMSG_HEADER + payload_bytes


def dds_acknack_size_wifi() -> int:
    return WIFI_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER + RTPS_ACKNACK_SUBMSG


def dds_frame_size(payload_bytes: int) -> int:
    return dds_frame_size_wifi(payload_bytes)


def dds_acknack_size() -> int:
    return dds_acknack_size_wifi()


def dds_batched_frame_size_wifi(payload_bytes_list: list) -> int:
    """Multiple DATA submessages share one RTPS message header (OMG RTPS 2.3 §8.3.7.5)."""
    return (WIFI_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER
            + len(payload_bytes_list) * RTPS_DATA_SUBMSG_HEADER
            + sum(payload_bytes_list))


# ==========================================
# 5) DDS MESSAGE TYPES
# ==========================================

@dataclass
class SensorMessage:
    msg_id: int = 0
    source_id: int = 0
    creation_time_ms: int = 0
    hop_count: int = 0
    qos_level: int = 0
    tokens: int = 0


@dataclass
class ControlMessage:
    sender_id: int = 0
    utility_to_sink: float = 0.0


# ==========================================
# 6) SIMULATED DDS INTERFACE
# ==========================================

_SIMULATED_DATA_QUEUE: List = []
_SIMULATED_CONTROL_QUEUE: List = []


class SimulatedDataWriter:
    def __init__(self, node_id: int):
        self.node_id = node_id

    def write(self, msg):
        _SIMULATED_DATA_QUEUE.append((self.node_id, msg))


class SimulatedDataReader:
    def __init__(self, node_id: int):
        self.node_id = node_id

    def take_data(self):
        global _SIMULATED_DATA_QUEUE
        msgs = [msg for _, msg in _SIMULATED_DATA_QUEUE]
        _SIMULATED_DATA_QUEUE.clear()
        return msgs


class SimulatedControlWriter:
    def __init__(self, node_id: int):
        self.node_id = node_id

    def write(self, msg):
        _SIMULATED_CONTROL_QUEUE.append((self.node_id, msg))


class SimulatedControlReader:
    def __init__(self, node_id: int):
        self.node_id = node_id

    def take_data(self):
        global _SIMULATED_CONTROL_QUEUE
        msgs = list(_SIMULATED_CONTROL_QUEUE)
        _SIMULATED_CONTROL_QUEUE.clear()
        return msgs


class DDSInterface:
    def __init__(self, node_id: int, reliable: bool = True):
        self.node_id = node_id
        self.reliable = reliable
        self.data_writer = SimulatedDataWriter(node_id)
        self.data_reader = SimulatedDataReader(node_id)
        self.control_writer = SimulatedControlWriter(node_id)
        self.control_reader = SimulatedControlReader(node_id)


# ==========================================
# 7) DATA STRUCTURES
# ==========================================

@dataclass
class SprayMessage:
    msg_id: int
    source_id: int
    creation_time: float
    hop_count: int
    tokens: int
    qos: int
    payload_bytes: int = PhyConst.WIFI_DATA_PAYLOAD_BYTES

    def payload_size(self) -> int:
        return self.payload_bytes


# ==========================================
# 8) SPRAY & FOCUS DDS AGENT
# ==========================================

class SprayFocusDDSAgent:
    MAX_BUFFER = 250

    def __init__(self, uid: int, pos: List[float], is_sink: bool = False,
                 reliable: bool = True, area_size: float = 500):
        self.id = uid
        self.pos = np.array(pos, dtype=float)
        self.is_sink = is_sink
        self.area_size = area_size
        self.energy = 300000.0
        self.radio_tx_energy = 0.0
        self.radio_rx_energy = 0.0

        self.buffer: List[SprayMessage] = []
        self.seen_msgs = set()
        self.encounter_timers: Dict[int, float] = {i: 9999.0 for i in range(NUM_UAVS)}
        self.encounter_timers[uid] = 0.0

        self.dds = DDSInterface(uid, reliable=reliable)
        self.waypoints: List[np.ndarray] = []

        if is_sink:
            self.delivered_ids = set()
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
                              PhyConst.H]) for _ in range(5)
                ]
            else:
                self.waypoints = [
                    np.array([random.uniform(100, self.area_size - 100),
                              random.uniform(100, self.area_size - 100),
                              PhyConst.H]) for _ in range(5)
                ]

        speed = 20.0
        target = self.waypoints[0]
        direction = target - self.pos
        dist = np.linalg.norm(direction)

        if dist <= speed * dt:
            self.pos = target.copy()
            self.waypoints.pop(0)
        else:
            self.pos += (direction / dist) * speed * dt

        self.energy -= self._flight_power(speed) * dt

    def _flight_power(self, velocity: float) -> float:
        P0 = (PhyConst.DELTA / 8.0) * PhyConst.RHO * PhyConst.S_SOL * PhyConst.AREA \
             * (PhyConst.OMEGA ** 3) * (PhyConst.R_RAD ** 3)
        P_ind = 1.1 * (PhyConst.WEIGHT ** 1.5) / math.sqrt(2 * PhyConst.RHO * PhyConst.AREA)

        if velocity < 0.1:
            return P0 + P_ind

        term1 = P0 * (1.0 + 3.0 * velocity ** 2 / (PhyConst.U_TIP ** 2))
        term2 = P_ind * math.sqrt(
            math.sqrt(1.0 + (velocity ** 4) / (4 * PhyConst.V_0 ** 4))
            - (velocity ** 2) / (2 * PhyConst.V_0 ** 2)
        )
        term3 = 0.5 * PhyConst.D_0 * PhyConst.RHO * PhyConst.S_SOL * PhyConst.AREA * (velocity ** 3)
        return term1 + term2 + term3

    def get_utility(self) -> float:
        return self.encounter_timers[SINK_ID]


# ==========================================
# 9) HELPERS
# ==========================================

def _generate_spread_sensors(num_sensors: int, area_size: float, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    margin = 100
    min_dist = (area_size - 2 * margin) / (num_sensors ** 0.5 + 1)
    positions = []
    for _ in range(num_sensors):
        for attempt in range(1000):
            x = rng.uniform(margin, area_size - margin)
            y = rng.uniform(margin, area_size - margin)
            valid = all(
                np.sqrt((x - px) ** 2 + (y - py) ** 2) >= min_dist
                for px, py, _ in positions
            )
            if valid or attempt == 999:
                positions.append([x, y, 10.0])
                break
    return [np.array(p) for p in positions]


def _charge_wifi(sender, receiver, frame_bytes: int, prof: PHYProfile,
                 data_wifi_tx: list, data_wifi_rx: list):
    """Deduct WiFi TX energy from sender; RX energy from receiver. Appends to accumulators."""
    bits = frame_bytes * 8
    tx_e = bits * prof.E_tx_per_bit
    rx_e = bits * prof.E_rx_per_bit
    sender.energy -= tx_e
    sender.radio_tx_energy += tx_e
    data_wifi_tx[0] += tx_e
    receiver.energy -= rx_e
    receiver.radio_rx_energy += rx_e
    data_wifi_rx[0] += rx_e
    return tx_e, rx_e


def _charge_acknack(sender, receiver, prof: PHYProfile,
                    data_wifi_tx: list, data_wifi_rx: list):
    """Deduct ACKNACK energy if RELIABLE QoS is active."""
    if GLOBAL_QOS != 1:
        return
    ack_bytes = dds_acknack_size()
    ack_bits = ack_bytes * 8
    tx_e = ack_bits * prof.E_tx_per_bit
    rx_e = ack_bits * prof.E_rx_per_bit
    sender.energy -= tx_e
    sender.radio_tx_energy += tx_e
    data_wifi_tx[0] += tx_e
    receiver.energy -= rx_e
    receiver.radio_rx_energy += rx_e
    data_wifi_rx[0] += rx_e


# ==========================================
# 10) SIMULATION
# ==========================================

def run_spray_focus_dds_simulation(config: dict, verbose: bool = False) -> dict:
    global NUM_UAVS, NUM_SENSORS, GLOBAL_QOS, SINK_ID, INITIAL_TOKENS
    global _SIMULATED_DATA_QUEUE, _SIMULATED_CONTROL_QUEUE, BATCH_ENABLE

    _SIMULATED_DATA_QUEUE = []
    _SIMULATED_CONTROL_QUEUE = []

    NUM_UAVS = config.get("NUM_UAVS", 8)
    NUM_SENSORS = config.get("NUM_SENSORS", 6)
    GLOBAL_QOS = config.get("GLOBAL_QOS", 1)
    INITIAL_TOKENS = config.get("INITIAL_TOKENS", 10)
    SINK_ID = 0

    AREA_SIZE = config.get("AREA_SIZE", 500)
    SINK_MOBILE = config.get("SINK_MOBILE", True)
    RELIABLE = GLOBAL_QOS == 1
    duration = config.get("DURATION", 3000.0)
    dt = 0.1

    if "WIFI_PAYLOAD_BYTES" in config:
        PhyConst.WIFI_DATA_PAYLOAD_BYTES = config["WIFI_PAYLOAD_BYTES"]
    if "MAX_BUFFER" in config:
        SprayFocusDDSAgent.MAX_BUFFER = config["MAX_BUFFER"]
    if "BATCH_ENABLE" in config:
        BATCH_ENABLE = config["BATCH_ENABLE"]

    BUFFER_POLICY = config.get("BUFFER_POLICY", "SMART")

    agents: Dict[int, SprayFocusDDSAgent] = {}

    sink_pos = (
        [random.uniform(100, AREA_SIZE - 100), random.uniform(100, AREA_SIZE - 100), PhyConst.H]
        if SINK_MOBILE else [AREA_SIZE / 2, AREA_SIZE / 2, PhyConst.H]
    )
    agents[SINK_ID] = SprayFocusDDSAgent(SINK_ID, sink_pos, is_sink=not SINK_MOBILE,
                                          reliable=RELIABLE, area_size=AREA_SIZE)
    agents[SINK_ID].delivered_ids = set()
    agents[SINK_ID].sink_received_count = 0

    for i in range(1, NUM_UAVS):
        agents[i] = SprayFocusDDSAgent(
            i,
            [random.uniform(100, AREA_SIZE - 100), random.uniform(100, AREA_SIZE - 100), PhyConst.H],
            reliable=RELIABLE, area_size=AREA_SIZE
        )

    initial_energy = {uid: agent.energy for uid, agent in agents.items()}
    iot_nodes = _generate_spread_sensors(NUM_SENSORS, AREA_SIZE)

    sim_time = 0.0
    SENSOR_RATE = 2.0
    SENSOR_BUF_MAX = 50
    sensor_queues: List[List[Tuple[int, int, float]]] = [[] for _ in range(NUM_SENSORS)]
    MSG_COUNTER = 0

    total_generated = 0
    total_delivered = 0
    uav_relay_events = 0
    spray_events = 0
    focus_events = 0
    sink_delivery_events = 0
    control_messages_sent = 0
    latencies: List[float] = []
    hop_counts: List[int] = []
    direct_deliveries = 0
    relayed_deliveries = 0

    sensor_tx_energy = 0.0
    control_tx_energy = 0.0
    control_rx_energy = 0.0
    # Mutable accumulators passed into helpers via single-element lists
    dwt = [0.0]  # data_wifi_tx_energy
    dwr = [0.0]  # data_wifi_rx_energy
    control_bytes_sent = 0
    data_bytes_sent = 0

    sink_batches_sent = 0
    sink_batch_samples = 0
    spray_batches_sent = 0
    spray_batch_samples = 0
    single_msg_sends = 0

    while sim_time < duration:
        sim_time += dt

        # Sensor data generation
        for s in range(NUM_SENSORS):
            if random.random() < SENSOR_RATE * dt:
                MSG_COUNTER += 1
                sensor_queues[s].append((MSG_COUNTER, GLOBAL_QOS, sim_time))
                if len(sensor_queues[s]) > SENSOR_BUF_MAX:
                    sensor_queues[s].pop(0)
                total_generated += 1

        # UAV movement
        for uid, agent in agents.items():
            if uid == SINK_ID and not SINK_MOBILE:
                continue
            agent.move(dt)

        # Encounter detection & distance-vector updates
        for i in range(NUM_UAVS):
            for j in range(i + 1, NUM_UAVS):
                rate, d, _ = link_rate(agents[i].pos, agents[j].pos, is_ground_to_uav=False)
                if rate >= WIFI.B:
                    agents[i].encounter_timers[j] = 0.0
                    agents[j].encounter_timers[i] = 0.0
                    t_meet = d / 20.0
                    ti = agents[i].encounter_timers[SINK_ID]
                    tj = agents[j].encounter_timers[SINK_ID]
                    if tj + t_meet < ti:
                        agents[i].encounter_timers[SINK_ID] = tj + t_meet
                    if ti + t_meet < tj:
                        agents[j].encounter_timers[SINK_ID] = ti + t_meet

        # UAV → Sink delivery (batched or unbatched)
        sink = agents[SINK_ID]

        for i in range(1, NUM_UAVS):
            ai = agents[i]
            if not ai.buffer:
                continue

            rate_bps, d, prof = link_rate(ai.pos, sink.pos, is_ground_to_uav=False)
            if rate_bps < prof.B:
                continue

            max_bytes = (rate_bps * dt) / 8.0
            bytes_sent = 0
            msgs_to_remove = []

            deliverable = [m for m in ai.buffer if m.msg_id not in sink.delivered_ids]
            msgs_to_remove += [m for m in ai.buffer if m.msg_id in sink.delivered_ids]

            def _record_delivery(msg):
                nonlocal total_delivered, sink_delivery_events, direct_deliveries, relayed_deliveries
                sink.delivered_ids.add(msg.msg_id)
                sink.sink_received_count += 1
                total_delivered += 1
                sink_delivery_events += 1
                hop_counts.append(msg.hop_count)
                latencies.append(sim_time - msg.creation_time)
                msgs_to_remove.append(msg)
                if msg.hop_count > 0:
                    relayed_deliveries += 1
                else:
                    direct_deliveries += 1

            if BATCH_ENABLE and len(deliverable) > 1:
                idx = 0
                while idx < len(deliverable):
                    batch, batch_payload = [], 0
                    while idx < len(deliverable):
                        msg = deliverable[idx]
                        mp = msg.payload_size()
                        if len(batch) >= BATCH_MAX_SAMPLES:
                            break
                        if batch_payload + mp > BATCH_MAX_DATA_BYTES and batch:
                            break
                        batch.append(msg)
                        batch_payload += mp
                        idx += 1
                    if not batch:
                        break
                    frame_bytes = dds_batched_frame_size_wifi([m.payload_size() for m in batch])
                    if bytes_sent + frame_bytes > max_bytes:
                        break
                    _charge_wifi(ai, sink, frame_bytes, prof, dwt, dwr)
                    _charge_acknack(sink, ai, prof, dwt, dwr)
                    data_bytes_sent += frame_bytes
                    bytes_sent += frame_bytes
                    for msg in batch:
                        _record_delivery(msg)
                    sink_batches_sent += 1
                    sink_batch_samples += len(batch)
            else:
                for msg in deliverable:
                    frame_bytes = dds_frame_size(msg.payload_size())
                    if bytes_sent + frame_bytes > max_bytes:
                        break
                    _charge_wifi(ai, sink, frame_bytes, prof, dwt, dwr)
                    _charge_acknack(sink, ai, prof, dwt, dwr)
                    data_bytes_sent += frame_bytes
                    bytes_sent += frame_bytes
                    _record_delivery(msg)
                    single_msg_sends += 1

            for m in msgs_to_remove:
                if m in ai.buffer:
                    ai.buffer.remove(m)

        # Global duplicate purge
        if total_delivered > 0 and int(sim_time * 10) % 10 == 0:
            delivered_ids = sink.delivered_ids.copy()
            for uid, a in agents.items():
                if uid == SINK_ID or not a.buffer:
                    continue
                a.buffer = [m for m in a.buffer if m.msg_id not in delivered_ids]

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

                max_bytes = (rate * dt) / 8.0
                bytes_used = 0

                spray_msgs = [m for m in ai.buffer if m.tokens > 1]
                focus_msgs = [m for m in ai.buffer if m.tokens == 1]

                # Spray phase
                spray_eligible = [
                    m for m in spray_msgs
                    if m.msg_id not in aj.seen_msgs and len(aj.buffer) < aj.MAX_BUFFER
                ]

                if BATCH_ENABLE and len(spray_eligible) > 1:
                    idx = 0
                    while idx < len(spray_eligible) and len(aj.buffer) < aj.MAX_BUFFER:
                        batch, batch_payload = [], 0
                        while idx < len(spray_eligible) and len(aj.buffer) + len(batch) < aj.MAX_BUFFER:
                            msg = spray_eligible[idx]
                            mp = msg.payload_size()
                            if len(batch) >= BATCH_MAX_SAMPLES:
                                break
                            if batch_payload + mp > BATCH_MAX_DATA_BYTES and batch:
                                break
                            batch.append(msg)
                            batch_payload += mp
                            idx += 1
                        if not batch:
                            break
                        frame_bytes = dds_batched_frame_size_wifi([m.payload_size() for m in batch])
                        if bytes_used + frame_bytes > max_bytes:
                            break
                        _charge_wifi(ai, aj, frame_bytes, prof, dwt, dwr)
                        _charge_acknack(aj, ai, prof, dwt, dwr)
                        data_bytes_sent += frame_bytes
                        bytes_used += frame_bytes
                        for msg in batch:
                            send_tokens = msg.tokens // 2
                            aj.buffer.append(SprayMessage(
                                msg_id=msg.msg_id, source_id=msg.source_id,
                                creation_time=msg.creation_time, hop_count=msg.hop_count + 1,
                                tokens=send_tokens, qos=msg.qos, payload_bytes=msg.payload_bytes
                            ))
                            aj.seen_msgs.add(msg.msg_id)
                            msg.tokens -= send_tokens
                            uav_relay_events += 1
                            spray_events += 1
                        spray_batches_sent += 1
                        spray_batch_samples += len(batch)
                else:
                    for msg in spray_eligible:
                        if len(aj.buffer) >= aj.MAX_BUFFER:
                            break
                        send_tokens = msg.tokens // 2
                        frame_bytes = dds_frame_size(msg.payload_size())
                        if bytes_used + frame_bytes > max_bytes:
                            break
                        _charge_wifi(ai, aj, frame_bytes, prof, dwt, dwr)
                        _charge_acknack(aj, ai, prof, dwt, dwr)
                        data_bytes_sent += frame_bytes
                        bytes_used += frame_bytes
                        aj.buffer.append(SprayMessage(
                            msg_id=msg.msg_id, source_id=msg.source_id,
                            creation_time=msg.creation_time, hop_count=msg.hop_count + 1,
                            tokens=send_tokens, qos=msg.qos, payload_bytes=msg.payload_bytes
                        ))
                        aj.seen_msgs.add(msg.msg_id)
                        msg.tokens -= send_tokens
                        uav_relay_events += 1
                        spray_events += 1
                        single_msg_sends += 1

                # Focus phase
                if focus_msgs:
                    ctrl_frame = dds_frame_size(PhyConst.CONTROL_PAYLOAD_BYTES)
                    ack_frame = dds_acknack_size()
                    ctrl_total = 2 * ctrl_frame + 2 * ack_frame
                    if bytes_used + ctrl_total > max_bytes:
                        continue

                    # Inquiry (ai → aj)
                    _charge_wifi(ai, aj, ctrl_frame, prof,
                                 [control_tx_energy], [control_rx_energy])
                    _charge_wifi(aj, ai, ack_frame, prof,
                                 [control_tx_energy], [control_rx_energy])
                    # Response (aj → ai)
                    _charge_wifi(aj, ai, ctrl_frame, prof,
                                 [control_tx_energy], [control_rx_energy])
                    _charge_wifi(ai, aj, ack_frame, prof,
                                 [control_tx_energy], [control_rx_energy])

                    # Manual energy accounting for control (not via helper accumulators)
                    ctrl_bits = ctrl_frame * 8
                    ack_bits = ack_frame * 8
                    for _bits, _node_tx, _node_rx in [
                        (ctrl_bits, ai, aj), (ack_bits, aj, ai),
                        (ctrl_bits, aj, ai), (ack_bits, ai, aj),
                    ]:
                        _node_tx.energy -= _bits * prof.E_tx_per_bit
                        _node_tx.radio_tx_energy += _bits * prof.E_tx_per_bit
                        control_tx_energy += _bits * prof.E_tx_per_bit
                        _node_rx.energy -= _bits * prof.E_rx_per_bit
                        _node_rx.radio_rx_energy += _bits * prof.E_rx_per_bit
                        control_rx_energy += _bits * prof.E_rx_per_bit

                    control_messages_sent += 2
                    control_bytes_sent += 2 * ctrl_frame
                    bytes_used += ctrl_total

                    my_util = ai.get_utility()
                    nb_util = aj.get_utility()
                    t_meet = d / 20.0

                    for msg in focus_msgs:
                        if (nb_util - t_meet) >= my_util:
                            continue
                        if msg.msg_id in aj.seen_msgs or len(aj.buffer) >= aj.MAX_BUFFER:
                            continue
                        frame_bytes = dds_frame_size(msg.payload_size())
                        if bytes_used + frame_bytes > max_bytes:
                            break
                        _charge_wifi(ai, aj, frame_bytes, prof, dwt, dwr)
                        _charge_acknack(aj, ai, prof, dwt, dwr)
                        data_bytes_sent += frame_bytes
                        bytes_used += frame_bytes
                        aj.buffer.append(SprayMessage(
                            msg_id=msg.msg_id, source_id=msg.source_id,
                            creation_time=msg.creation_time, hop_count=msg.hop_count + 1,
                            tokens=1, qos=msg.qos, payload_bytes=msg.payload_bytes
                        ))
                        aj.seen_msgs.add(msg.msg_id)
                        if msg in ai.buffer:
                            ai.buffer.remove(msg)
                        uav_relay_events += 1
                        focus_events += 1

        # Sensor → UAV upload (ZigBee)
        for s, src_pos in enumerate(iot_nodes):
            if not sensor_queues[s]:
                continue

            best_uav = min(
                [k for k in range(NUM_UAVS) if k != SINK_ID],
                key=lambda k: np.linalg.norm(agents[k].pos - src_pos)
            )
            rate, d, prof = link_rate(src_pos, agents[best_uav].pos, is_ground_to_uav=True)

            if rate < prof.B:
                if sensor_queues[s][0][1] == 0:
                    sensor_queues[s].pop(0)
                continue

            msg_id, qos_val, t0 = sensor_queues[s][0]
            frame_bytes = dds_frame_size_zigbee(PhyConst.SENSOR_PAYLOAD_BYTES)

            if frame_bytes > ZIGBEE_MAX_FRAME:
                continue

            sensor_tx_energy += frame_bytes * 8 * prof.E_tx_per_bit
            rx_e = frame_bytes * 8 * prof.E_rx_per_bit
            agents[best_uav].energy -= rx_e
            agents[best_uav].radio_rx_energy += rx_e

            target = agents[best_uav]

            def _insert_new(buf_agent, mid, sid, t_create, qos):
                buf_agent.buffer.append(SprayMessage(
                    msg_id=mid, source_id=sid, creation_time=t_create,
                    hop_count=0, tokens=INITIAL_TOKENS, qos=qos,
                    payload_bytes=PhyConst.SENSOR_PAYLOAD_BYTES
                ))
                buf_agent.seen_msgs.add(mid)

            if len(target.buffer) < target.MAX_BUFFER:
                sensor_queues[s].pop(0)
                _insert_new(target, msg_id, s, t0, qos_val)
            elif BUFFER_POLICY == "SMART":
                evicted = False
                for vi, vm in enumerate(target.buffer):
                    if vm.hop_count > 0:
                        target.buffer.pop(vi)
                        sensor_queues[s].pop(0)
                        _insert_new(target, msg_id, s, t0, qos_val)
                        evicted = True
                        break
                if not evicted and qos_val == 0:
                    sensor_queues[s].pop(0)
            elif BUFFER_POLICY == "FIFO":
                target.buffer.pop(0)
                sensor_queues[s].pop(0)
                _insert_new(target, msg_id, s, t0, qos_val)
            else:
                if qos_val == 0:
                    sensor_queues[s].pop(0)

    # Results
    total_uav_tx = sum(a.radio_tx_energy for a in agents.values())
    total_uav_rx = sum(a.radio_rx_energy for a in agents.values())
    total_uav_radio = total_uav_tx + total_uav_rx
    total_control_energy = control_tx_energy + control_rx_energy
    total_data_wifi_energy = dwt[0] + dwr[0]

    overhead_factor = (
        (uav_relay_events + total_delivered) / float(total_delivered)
        if total_delivered > 0 else 1.0
    )
    relayed_hop_counts = [h for h in hop_counts if h > 0]

    results = {
        "pdr": 100.0 * total_delivered / max(1, total_generated),
        "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency": float(np.median(latencies)) if latencies else 0.0,
        "avg_hops": float(np.mean(hop_counts)) if hop_counts else 0.0,
        "avg_hops_relayed": float(np.mean(relayed_hop_counts)) if relayed_hop_counts else 0.0,
        "direct_deliveries": direct_deliveries,
        "relayed_deliveries": relayed_deliveries,
        "direct_delivery_ratio": 100.0 * direct_deliveries / max(1, total_delivered),
        "overhead_factor": overhead_factor,
        "total_generated": total_generated,
        "total_delivered": total_delivered,
        "uav_relay_events": uav_relay_events,
        "spray_events": spray_events,
        "focus_events": focus_events,
        "sink_delivery_events": sink_delivery_events,
        "control_messages_sent": control_messages_sent,
        "control_energy": total_control_energy,
        "data_wifi_energy": total_data_wifi_energy,
        "data_zigbee_energy": sensor_tx_energy,
        "total_uav_radio_energy": total_uav_radio,
        "energy_per_msg_mJ": (total_uav_radio / max(1, total_delivered)) * 1000,
        "control_bytes": control_bytes_sent,
        "data_bytes": data_bytes_sent,
        "sink_batches_sent": sink_batches_sent,
        "sink_batch_samples": sink_batch_samples,
        "spray_batches_sent": spray_batches_sent,
        "spray_batch_samples": spray_batch_samples,
        "single_msg_sends": single_msg_sends,
        "avg_sink_batch_size": sink_batch_samples / max(1, sink_batches_sent),
        "avg_spray_batch_size": spray_batch_samples / max(1, spray_batches_sent),
        "sink_flight_energy_kJ": (
            initial_energy[SINK_ID] - agents[SINK_ID].energy
            - agents[SINK_ID].radio_tx_energy - agents[SINK_ID].radio_rx_energy
        ) / 1000.0,
        "total_system_energy_kJ": sum(
            initial_energy[uid] - agents[uid].energy for uid in agents
        ) / 1000.0,
    }

    if verbose:
        print(f"  [SPRAY&FOCUS DDS] PDR: {results['pdr']:.1f}% | "
              f"Latency: {results['avg_latency']:.2f}s | "
              f"Spray: {spray_events} | Focus: {focus_events} | "
              f"Batches: {sink_batches_sent + spray_batches_sent} "
              f"(sink:{sink_batches_sent}, spray:{spray_batches_sent})")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("DDS SPRAY & FOCUS SIMULATION")
    print("=" * 60)

    config = {"NUM_UAVS": 8, "NUM_SENSORS": 6, "GLOBAL_QOS": 1, "INITIAL_TOKENS": 10}
    result = run_spray_focus_dds_simulation(config, verbose=True)

    print(f"\nSpray & Focus DDS Results:")
    print(f"  PDR: {result['pdr']:.2f}%")
    print(f"  Avg Latency: {result['avg_latency']:.2f}s")
    print(f"  Avg Hops: {result['avg_hops']:.2f}")
    print(f"  Overhead Factor: {result['overhead_factor']:.2f}x")
    print(f"  Control Messages: {result['control_messages_sent']}")
    print(f"  Energy/Msg: {result['energy_per_msg_mJ']:.2f} mJ")
    print("=" * 60)
