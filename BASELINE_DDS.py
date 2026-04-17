"""
Vanilla DDS Simulation (No DTN Routing)

Baseline simulation using simulated DDS middleware without Spray & Focus routing.
Direct sensor → UAV → sink delivery only.
"""

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ==========================================
# 1) GLOBAL CONFIG
# ==========================================

GLOBAL_QOS = 1  # 0 = Best Effort, 1 = Reliable
NUM_UAVS = 6
SINK_ID = 0
NUM_SENSORS = 10
SINK_MOBILITY_FRACTION = 0.40

ZIGBEE_MAX_FRAME = 127
ZIGBEE_MAX_PAYLOAD = 100

_SIMULATED_DDS_QUEUE: List = []


# ==========================================
# 2) PHYSICS / RADIO PROFILES
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
# 3) DDS OVERHEAD MODEL
# ==========================================

RTPS_MESSAGE_HEADER = 20
RTPS_DATA_SUBMSG_HEADER = 24

WIFI_UDP_HEADER = 8
WIFI_IP_HEADER = 20
WIFI_L2_OVERHEAD = 30
WIFI_TRANSPORT_OVERHEAD = WIFI_UDP_HEADER + WIFI_IP_HEADER + WIFI_L2_OVERHEAD

ZIGBEE_L2_OVERHEAD = 15
ZIGBEE_TRANSPORT_OVERHEAD = ZIGBEE_L2_OVERHEAD


def dds_frame_size_zigbee(payload_bytes: int) -> int:
    return ZIGBEE_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER + RTPS_DATA_SUBMSG_HEADER + payload_bytes


def dds_frame_size_wifi(payload_bytes: int) -> int:
    return WIFI_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER + RTPS_DATA_SUBMSG_HEADER + payload_bytes


def dds_frame_size(payload_bytes: int) -> int:
    return dds_frame_size_wifi(payload_bytes)


# ==========================================
# 4) DDS MESSAGE TYPE
# ==========================================

@dataclass
class SensorMessage:
    msg_id: int = 0
    source_id: int = 0
    creation_time_ms: int = 0
    hop_count: int = 0
    qos_level: int = 0


# ==========================================
# 5) SIMULATED DDS INTERFACE
# ==========================================

class SimulatedWriter:
    def write(self, msg):
        _SIMULATED_DDS_QUEUE.append(msg)


class SimulatedReader:
    def take_data(self):
        global _SIMULATED_DDS_QUEUE
        msgs = _SIMULATED_DDS_QUEUE.copy()
        _SIMULATED_DDS_QUEUE.clear()
        return msgs


class DDSInterface:
    def __init__(self, node_id: int, reliable: bool = True):
        self.node_id = node_id
        self.reliable = reliable
        self.writer = SimulatedWriter()
        self.reader = SimulatedReader()


# ==========================================
# 6) UAV AGENT
# ==========================================

@dataclass
class BufferedMessage:
    msg_id: int
    source_id: int
    creation_time: float
    hop_count: int
    qos: int


class VanillaDDSAgent:
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
        self.buffer: List[BufferedMessage] = []
        self.seen_msgs = set()
        self.dds = DDSInterface(uid, reliable=reliable)
        self.waypoints: List[np.ndarray] = []

        if is_sink:
            self.delivered_ids = set()

    def move(self, dt: float):
        if not self.waypoints:
            center = self.area_size / 2
            if self.id == SINK_ID:
                radius = self.area_size * SINK_MOBILITY_FRACTION / 2
                self.waypoints = [
                    np.array([
                        center + random.uniform(-radius, radius),
                        center + random.uniform(-radius, radius),
                        PhyConst.H
                    ]) for _ in range(5)
                ]
            else:
                self.waypoints = [
                    np.array([
                        random.uniform(100, self.area_size - 100),
                        random.uniform(100, self.area_size - 100),
                        PhyConst.H
                    ]) for _ in range(5)
                ]

        speed = 20.0
        target = self.waypoints[0]
        direction = target - self.pos
        dist = np.linalg.norm(direction)
        step = speed * dt

        if dist <= step:
            self.pos = target.copy()
            self.waypoints.pop(0)
        else:
            self.pos += (direction / dist) * step

        self.energy -= self._flight_power(speed) * dt

    def _flight_power(self, velocity: float) -> float:
        term1 = PhyConst.P_C * (1 + (3 * velocity ** 2) / (PhyConst.U_TIP ** 2))
        term2 = PhyConst.WEIGHT * (
            math.sqrt(1 + (velocity ** 4) / (4 * PhyConst.V_0 ** 4)) -
            (velocity ** 2) / (2 * PhyConst.V_0 ** 2)
        )
        term3 = 0.5 * PhyConst.D_0 * PhyConst.RHO * PhyConst.S_SOL * PhyConst.AREA * (velocity ** 3)
        return term1 + term2 + term3

    def publish_to_sink(self, msg: BufferedMessage) -> Tuple[bool, float, float]:
        dds_msg = SensorMessage(
            msg_id=msg.msg_id,
            source_id=msg.source_id,
            creation_time_ms=int(msg.creation_time * 1000),
            hop_count=msg.hop_count,
            qos_level=msg.qos
        )
        try:
            self.dds.writer.write(dds_msg)
            frame_bytes = dds_frame_size(PhyConst.WIFI_DATA_PAYLOAD_BYTES)
            tx_energy = frame_bytes * 8 * WIFI.E_tx_per_bit
            return True, tx_energy, frame_bytes
        except Exception:
            return False, 0.0, 0


# ==========================================
# 7) SIMULATION
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


def run_vanilla_dds_simulation(config: dict, verbose: bool = False) -> dict:
    global NUM_UAVS, NUM_SENSORS, GLOBAL_QOS, SINK_ID, _SIMULATED_DDS_QUEUE
    _SIMULATED_DDS_QUEUE = []

    NUM_UAVS = config.get("NUM_UAVS", 8)
    NUM_SENSORS = config.get("NUM_SENSORS", 6)
    GLOBAL_QOS = config.get("GLOBAL_QOS", 1)
    SINK_ID = 0

    AREA_SIZE = config.get("AREA_SIZE", 500)
    SINK_MOBILE = config.get("SINK_MOBILE", True)
    RELIABLE = GLOBAL_QOS == 1
    duration = config.get("DURATION", 1500.0)
    dt = 0.1

    if "WIFI_PAYLOAD_BYTES" in config:
        PhyConst.WIFI_DATA_PAYLOAD_BYTES = config["WIFI_PAYLOAD_BYTES"]

    if "MAX_BUFFER" in config:
        VanillaDDSAgent.MAX_BUFFER = config["MAX_BUFFER"]

    agents: Dict[int, VanillaDDSAgent] = {}

    sink_pos = (
        [random.uniform(100, AREA_SIZE - 100), random.uniform(100, AREA_SIZE - 100), PhyConst.H]
        if SINK_MOBILE else [AREA_SIZE / 2, AREA_SIZE / 2, PhyConst.H]
    )
    agents[SINK_ID] = VanillaDDSAgent(SINK_ID, sink_pos, is_sink=not SINK_MOBILE,
                                      reliable=RELIABLE, area_size=AREA_SIZE)
    agents[SINK_ID].delivered_ids = set()

    for i in range(1, NUM_UAVS):
        agents[i] = VanillaDDSAgent(
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
    sink_delivery_events = 0
    latencies: List[float] = []
    hop_counts: List[int] = []

    sensor_tx_energy = 0.0
    data_wifi_tx_energy = 0.0
    data_wifi_rx_energy = 0.0
    data_bytes_sent = 0

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

        # UAV → Sink delivery via DDS
        sink = agents[SINK_ID]
        for i in range(1, NUM_UAVS):
            ai = agents[i]
            if not ai.buffer:
                continue

            rate_bps, d, prof = link_rate(ai.pos, sink.pos, is_ground_to_uav=False)
            if rate_bps < prof.B:
                continue

            max_bytes_this_step = (rate_bps * dt) / 8.0
            bytes_sent = 0
            msgs_to_remove = []

            for msg in ai.buffer:
                if msg.msg_id in sink.delivered_ids:
                    msgs_to_remove.append(msg)
                    continue

                frame_bytes = dds_frame_size(PhyConst.WIFI_DATA_PAYLOAD_BYTES)
                if bytes_sent + frame_bytes > max_bytes_this_step:
                    break

                success, tx_e, sent_bytes = ai.publish_to_sink(msg)
                if not success:
                    continue

                ai.radio_tx_energy += tx_e
                data_wifi_tx_energy += tx_e

                rx_e = sent_bytes * 8 * prof.E_rx_per_bit
                sink.radio_rx_energy += rx_e
                data_wifi_rx_energy += rx_e

                data_bytes_sent += sent_bytes
                bytes_sent += sent_bytes
                msgs_to_remove.append(msg)

            for m in msgs_to_remove:
                if m in ai.buffer:
                    ai.buffer.remove(m)

        # Sink reads DDS messages
        try:
            for data in sink.dds.reader.take_data():
                if data.msg_id in sink.delivered_ids:
                    continue
                sink.delivered_ids.add(data.msg_id)
                total_delivered += 1
                sink_delivery_events += 1
                hop_counts.append(data.hop_count)
                latencies.append(sim_time - (data.creation_time_ms / 1000.0))
        except Exception:
            pass

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
                msg_id, qos_val, t0 = sensor_queues[s][0]
                if qos_val == 0:
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

            if len(agents[best_uav].buffer) < agents[best_uav].MAX_BUFFER:
                sensor_queues[s].pop(0)
                agents[best_uav].buffer.append(
                    BufferedMessage(msg_id=msg_id, source_id=s, creation_time=t0,
                                   hop_count=0, qos=qos_val)
                )
                agents[best_uav].seen_msgs.add(msg_id)
            elif qos_val == 0:
                sensor_queues[s].pop(0)

    total_uav_tx = sum(a.radio_tx_energy for a in agents.values())
    total_uav_rx = sum(a.radio_rx_energy for a in agents.values())
    total_uav_radio = total_uav_tx + total_uav_rx

    results = {
        "pdr": 100.0 * total_delivered / max(1, total_generated),
        "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency": float(np.median(latencies)) if latencies else 0.0,
        "avg_hops": float(np.mean(hop_counts)) if hop_counts else 0.0,
        "overhead_factor": 1.0,
        "total_generated": total_generated,
        "total_delivered": total_delivered,
        "uav_relay_events": 0,
        "sink_delivery_events": sink_delivery_events,
        "control_messages_sent": 0,
        "control_energy": 0.0,
        "data_wifi_energy": data_wifi_tx_energy + data_wifi_rx_energy,
        "data_zigbee_energy": sensor_tx_energy,
        "total_uav_radio_energy": total_uav_radio,
        "energy_per_msg_mJ": (total_uav_radio / max(1, total_delivered)) * 1000,
        "sink_flight_energy_kJ": (
            initial_energy[SINK_ID] - agents[SINK_ID].energy
            - agents[SINK_ID].radio_tx_energy - agents[SINK_ID].radio_rx_energy
        ) / 1000.0,
        "total_system_energy_kJ": sum(
            initial_energy[uid] - agents[uid].energy for uid in agents
        ) / 1000.0,
    }

    if verbose:
        print(f"  [VANILLA DDS] PDR: {results['pdr']:.1f}% | Latency: {results['avg_latency']:.2f}s")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("VANILLA DDS SIMULATION (No DTN Routing)")
    print("=" * 60)

    config = {"NUM_UAVS": 8, "NUM_SENSORS": 6, "GLOBAL_QOS": 1}
    result = run_vanilla_dds_simulation(config, verbose=True)

    print(f"\nVanilla DDS Results:")
    print(f"  PDR: {result['pdr']:.2f}%")
    print(f"  Avg Latency: {result['avg_latency']:.2f}s")
    print(f"  Energy/Msg: {result['energy_per_msg_mJ']:.2f} mJ")
    print("=" * 60)
