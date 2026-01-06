"""
BASELINE MQTT Simulation (No DTN Routing)

This is the baseline comparison for the EdgeDrone enhanced MQTT with Spray & Focus.
In this simulation:
- UAVs collect data from sensors
- UAVs can ONLY deliver to sink directly (no UAV-to-UAV relay)
- Messages wait in buffer until UAV reaches sink
- No spray, no focus, no DTN routing

This simulates standard MQTT over mobile ad-hoc network without DTN enhancement.
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# ==========================================
# GLOBAL CONFIG
# ==========================================

GLOBAL_QOS = 1
INITIAL_TOKENS = 10  # Not used in baseline, kept for compatibility
NUM_UAVS = 8
SINK_ID = 0
NUM_SENSORS = 6

MQTT_TOPIC_LEN = 16
TCP_IP_OVERHEAD = 40
L2_OVERHEAD = 30

# ==========================================
# PHYSICS / RADIO PROFILES
# ==========================================

# ZigBee MTU constraint (IEEE 802.15.4 max frame = 127 bytes)
ZIGBEE_MAX_FRAME = 127
ZIGBEE_MAX_PAYLOAD = 100  # ~27 bytes MAC overhead

class PhyConst:
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
    
    # ZigBee (IoT → UAV): Fixed small sensor payload
    SENSOR_PAYLOAD_BYTES = 64
    
    # WiFi (UAV → Sink): Variable data payload  
    WIFI_DATA_PAYLOAD_BYTES = 256
    
    # Control messages
    CONTROL_PAYLOAD_BYTES = 80


class PHYProfile:
    def __init__(self, name: str, B: float, P_tx: float, N0: float, E_tx_per_bit: float, E_rx_per_bit: float):
        self.name = name
        self.B = B
        self.P_tx = P_tx
        self.N0 = N0
        self.beta0 = None
        self.E_tx_per_bit = E_tx_per_bit
        self.E_rx_per_bit = E_rx_per_bit


ZIGBEE = PHYProfile("zigbee", B=250_000.0, P_tx=0.001, N0=1e-13, E_tx_per_bit=3e-6, E_rx_per_bit=2.1e-6)
WIFI = PHYProfile("wifi", B=20_000_000.0, P_tx=0.1, N0=1e-13, E_tx_per_bit=5e-7, E_rx_per_bit=3.5e-7)

REF_DIST = 100.0


def calibrate_beta0(prof: PHYProfile, target_snr_dB: float = 0.0, ref_dist: float = REF_DIST):
    # snr_linear = 10 ** (target_snr_dB / 10.0)
    prof.beta0 = prof.N0 * (REF_DIST**2) / prof.P_tx

def shannon_rate_3d(dist_3d: float, profile: PHYProfile) -> float:
    if dist_3d <= 0.0:
        dist_3d = 1e-3
    snr = (profile.beta0 * profile.P_tx) / (profile.N0 * (dist_3d**2))
    return profile.B * math.log2(1.0 + snr)

def link_rate(pos1, pos2, is_ground_to_uav: bool):
    d = float(np.linalg.norm(np.array(pos1) - np.array(pos2)))
    prof = ZIGBEE if is_ground_to_uav else WIFI
    return shannon_rate_3d(d, prof), d, prof

calibrate_beta0(ZIGBEE, target_snr_dB=0.0)
calibrate_beta0(WIFI, target_snr_dB=0.0)


# ==========================================
# MQTT FRAME SIZES
# ==========================================

def mqtt_frame_size(payload_bytes: int, qos: int) -> int:
    mqtt_header = 2 + 2 + MQTT_TOPIC_LEN + (2 if qos > 0 else 0)
    return TCP_IP_OVERHEAD + L2_OVERHEAD + mqtt_header + payload_bytes


def mqtt_puback_size() -> int:
    return TCP_IP_OVERHEAD + L2_OVERHEAD + 4


# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class BaselineMessage:
    """Simple MQTT message (no tokens, no DTN)"""
    msg_id: int
    source_id: int
    creation_time: float
    hop_count: int  # Will always be 0 or 1 in baseline
    payload: bytes
    qos: int

    def payload_size(self) -> int:
        return PhyConst.DATA_PAYLOAD_BYTES


class MQTTTransmission:
    @staticmethod
    def transmit_data(rate_bps: float, prof: PHYProfile, msg):
        if rate_bps < prof.B:
            return False, 0.0, 0.0, 0

        frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
        bits = frame_bytes * 8
        tx_energy = bits * prof.E_tx_per_bit
        rx_energy = bits * prof.E_rx_per_bit
        return True, tx_energy, rx_energy, frame_bytes

    @staticmethod
    def transmit_puback(rate_bps: float, prof: PHYProfile):
        if rate_bps < prof.B:
            return False, 0.0, 0.0, 0

        ack_bytes = mqtt_puback_size()
        bits = ack_bytes * 8
        tx_energy = bits * prof.E_tx_per_bit
        rx_energy = bits * prof.E_rx_per_bit
        return True, tx_energy, rx_energy, ack_bytes



# ==========================================
# UAV AGENT
# ==========================================

class BaselineUAVAgent:
    MAX_BUFFER = 50

    def __init__(self, uid: int, pos: List[float], is_sink: bool = False, area_size: float = 500):
        self.id = uid
        self.pos = np.array(pos, dtype=float)
        self.energy = 300000.0  # 300 kJ - ~28 min flight time
        self.is_sink = is_sink
        self.area_size = area_size
        self.buffer: List[BaselineMessage] = []
        self.seen_msgs = set()
        self.radio_tx_energy = 0.0
        self.radio_rx_energy = 0.0
        self.sink_received_count = 0
        self.vel = np.array([0.0, 0.0, 0.0])
        self._waypoint_timer = 0.0

    def move(self, dt: float):
        if self.is_sink:
            return
        self._waypoint_timer -= dt
        if self._waypoint_timer <= 0:
            self._waypoint_timer = random.uniform(5.0, 15.0)
            target = np.array([random.uniform(100, self.area_size-100), 
                              random.uniform(100, self.area_size-100), PhyConst.H])
            direction = target - self.pos
            norm = np.linalg.norm(direction[:2])
            speed = random.uniform(5.0, 15.0)
            if norm > 1.0:
                self.vel[:2] = (direction[:2] / norm) * speed

        self.pos += self.vel * dt
        self.pos[0] = np.clip(self.pos[0], 0, self.area_size)
        self.pos[1] = np.clip(self.pos[1], 0, self.area_size)

        flight_power = self.flight_power(np.linalg.norm(self.vel[:2]))
        self.energy -= flight_power * dt

    def flight_power(self, velocity: float) -> float:
        term1 = PhyConst.P_C * (
            1 + (3 * velocity ** 2) / (PhyConst.U_TIP ** 2)
        )
        term2 = PhyConst.WEIGHT * (
            math.sqrt(
                1 + (velocity ** 4) / (4 * PhyConst.V_0 ** 4)
            ) -
            (velocity ** 2) / (2 * PhyConst.V_0 ** 2)
        )
        term3 = 0.5 * PhyConst.D_0 * PhyConst.RHO * PhyConst.S_SOL * PhyConst.AREA * (velocity ** 3)
        return term1 + term2 + term3


# ==========================================
# BASELINE SIMULATION (Direct Delivery Only)
# ==========================================

def run_baseline_simulation(config: dict, verbose: bool = False) -> dict:
    """
    Run BASELINE simulation with direct-delivery only (no DTN routing).
    
    Messages are held in UAV buffer until the UAV can deliver directly to sink.
    No UAV-to-UAV relay is performed.
    """
    global NUM_UAVS, NUM_SENSORS, GLOBAL_QOS, SINK_ID
    
    NUM_UAVS = config.get("NUM_UAVS", 8)
    NUM_SENSORS = config.get("NUM_SENSORS", 6)
    GLOBAL_QOS = config.get("GLOBAL_QOS", 1)
    SINK_ID = 0
    
    # Area and sink configuration - match enhanced simulation (500x500m)
    AREA_SIZE = config.get("AREA_SIZE", 500)  # 500x500m to match enhanced
    SINK_MOBILE = config.get("SINK_MOBILE", True)  # Mobile sink by default
    
    if "DATA_PAYLOAD_BYTES" in config:
        PhyConst.DATA_PAYLOAD_BYTES = config["DATA_PAYLOAD_BYTES"]
    
    duration = config.get("DURATION", 1500.0)
    dt = 0.1
    
    # Initialize UAVs with larger area
    agents: Dict[int, BaselineUAVAgent] = {}
    
    # Sink: mobile or static at center
    if SINK_MOBILE:
        sink_pos = [random.uniform(100, AREA_SIZE-100), random.uniform(100, AREA_SIZE-100), PhyConst.H]
    else:
        sink_pos = [AREA_SIZE/2, AREA_SIZE/2, PhyConst.H]
    agents[SINK_ID] = BaselineUAVAgent(SINK_ID, sink_pos, is_sink=not SINK_MOBILE, area_size=AREA_SIZE)
    
    for i in range(1, NUM_UAVS):
        agents[i] = BaselineUAVAgent(i, [random.uniform(100, AREA_SIZE-100), random.uniform(100, AREA_SIZE-100), PhyConst.H], area_size=AREA_SIZE)

    # Initialize sensors spread across larger area
    iot_nodes = [
        np.array([100 + (i % 7) * (AREA_SIZE-200)/6, 100 + (i // 7) * (AREA_SIZE-200)/6, 10.0])
        for i in range(NUM_SENSORS)
    ]

    sim_time = 0.0
    SENSOR_RATE = 0.5
    SENSOR_BUF_MAX = 50
    sensor_queues: List[List[Tuple[int, int, float]]] = [[] for _ in range(NUM_SENSORS)]
    MSG_COUNTER = 0

    # Statistics
    total_generated = 0
    total_delivered = 0
    sink_delivery_events = 0
    latencies = []
    hop_counts = []
    
    sensor_tx_energy = 0.0
    sensor_rx_energy = 0.0
    data_wifi_tx_energy = 0.0
    data_wifi_rx_energy = 0.0
    data_bytes_sent = 0

    # Main loop
    while sim_time < duration:
        sim_time += dt

        # 0) SENSOR DATA GENERATION
        for s in range(NUM_SENSORS):
            if random.random() < SENSOR_RATE * dt:
                MSG_COUNTER += 1
                sensor_queues[s].append((MSG_COUNTER, GLOBAL_QOS, sim_time))
                if len(sensor_queues[s]) > SENSOR_BUF_MAX:
                    sensor_queues[s].pop(0)
                total_generated += 1

        # 1) UAV MOVEMENT
        for agent in agents.values():
            agent.move(dt)

        # 2) DIRECT DELIVERY TO SINK ONLY (NO UAV-TO-UAV RELAY)
        sink = agents[SINK_ID]
        for i in range(1, NUM_UAVS):
            ai = agents[i]
            if not ai.buffer:
                continue
            
            rate_bps, d, prof = link_rate(ai.pos, sink.pos, is_ground_to_uav=False)
            if rate_bps < prof.B:
                continue  # Not in range of sink
            
            max_bytes_this_step = (rate_bps * dt) / 8.0
            bytes_sent = 0
            msgs_to_remove = []

            for msg in ai.buffer:
                if msg.msg_id in sink.seen_msgs:
                    msgs_to_remove.append(msg)
                    continue
                
                frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                if bytes_sent + frame_bytes > max_bytes_this_step:
                    break
                
                success, tx_e, rx_e, _ = MQTTTransmission.transmit_data(rate_bps, prof, msg)
                if not success:
                    continue
                
                ai.energy -= tx_e
                ai.radio_tx_energy += tx_e
                data_wifi_tx_energy += tx_e
                sink.energy -= rx_e
                sink.radio_rx_energy += rx_e
                data_wifi_rx_energy += rx_e

                if msg.qos == 1:
                    succ_ack, tx_ack, rx_ack, _ = MQTTTransmission.transmit_puback(rate_bps, prof)
                    if succ_ack:
                        sink.energy -= tx_ack
                        sink.radio_tx_energy += tx_ack
                        data_wifi_tx_energy += tx_ack
                        ai.energy -= rx_ack
                        ai.radio_rx_energy += rx_ack
                        data_wifi_rx_energy += rx_ack

                sink.seen_msgs.add(msg.msg_id)
                total_delivered += 1
                sink_delivery_events += 1
                hop_counts.append(msg.hop_count)
                latencies.append(sim_time - msg.creation_time)
                bytes_sent += frame_bytes
                msgs_to_remove.append(msg)

            for m in msgs_to_remove:
                if m in ai.buffer:
                    ai.buffer.remove(m)

        # 3) SENSOR → UAV UPLOAD
        for s, src_pos in enumerate(iot_nodes):
            if not sensor_queues[s]:
                continue
            
            best_uav = min(range(1, NUM_UAVS), key=lambda k: np.linalg.norm(agents[k].pos - src_pos))
            rate, d, prof = link_rate(src_pos, agents[best_uav].pos, is_ground_to_uav=True)
            
            if rate < prof.B:
                msg_id, qos_val, t0 = sensor_queues[s][0]
                if qos_val == 0:
                    sensor_queues[s].pop(0)
                continue

            msg_id, qos_val, t0 = sensor_queues[s][0]
            temp_msg = BaselineMessage(
                msg_id=msg_id, source_id=s, creation_time=t0,
                hop_count=0, payload=b"SENSOR_DATA", qos=qos_val
            )
            
            max_bytes_this_step = (rate * dt) / 8.0
            frame_bytes = mqtt_frame_size(temp_msg.payload_size(), temp_msg.qos)
            if frame_bytes > max_bytes_this_step:
                if qos_val == 0:
                    sensor_queues[s].pop(0)
                continue

            success, tx_e, rx_e, _ = MQTTTransmission.transmit_data(rate, prof, temp_msg)
            if success and len(agents[best_uav].buffer) < agents[best_uav].MAX_BUFFER:
                sensor_queues[s].pop(0)
                agents[best_uav].buffer.append(temp_msg)
                agents[best_uav].seen_msgs.add(msg_id)
                sensor_tx_energy += tx_e
                agents[best_uav].energy -= rx_e
                agents[best_uav].radio_rx_energy += rx_e

                if qos_val == 1:
                    succ_ack, tx_ack, rx_ack, _ = MQTTTransmission.transmit_puback(rate, prof)
                    if succ_ack:
                        agents[best_uav].energy -= tx_ack
                        agents[best_uav].radio_tx_energy += tx_ack
                        sensor_rx_energy += rx_ack
            else:
                if qos_val == 0:
                    sensor_queues[s].pop(0)

    # Compute results
    total_uav_tx = sum(a.radio_tx_energy for a in agents.values())
    total_uav_rx = sum(a.radio_rx_energy for a in agents.values())
    total_uav_radio = total_uav_tx + total_uav_rx
    
    total_data_wifi_energy = data_wifi_tx_energy + data_wifi_rx_energy
    total_data_zigbee = sensor_tx_energy + sensor_rx_energy

    results = {
        "pdr": 100.0 * total_delivered / max(1, total_generated),
        "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency": float(np.median(latencies)) if latencies else 0.0,
        "avg_hops": float(np.mean(hop_counts)) if hop_counts else 0.0,
        "overhead_factor": 1.0,  # No relay overhead in baseline
        "total_generated": total_generated,
        "total_delivered": total_delivered,
        "uav_relay_events": 0,  # No relay in baseline
        "sink_delivery_events": sink_delivery_events,
        "control_messages_sent": 0,  # No control messages in baseline
        "control_energy": 0.0,
        "data_wifi_energy": total_data_wifi_energy,
        "data_zigbee_energy": total_data_zigbee,
        "total_uav_radio_energy": total_uav_radio,
        "energy_per_msg_mJ": (total_uav_radio / max(1, total_delivered)) * 1000
    }
    
    if verbose:
        print(f"  [BASELINE] PDR: {results['pdr']:.1f}% | Latency: {results['avg_latency']:.2f}s")
    
    return results


if __name__ == "__main__":
    # Quick test
    config = {"NUM_UAVS": 8, "NUM_SENSORS": 6, "GLOBAL_QOS": 1}
    # random.seed(42)
    # np.random.seed(42)
    result = run_baseline_simulation(config, verbose=True)
    print(f"\nBaseline MQTT Results:")
    print(f"  PDR: {result['pdr']:.2f}%")
    print(f"  Avg Latency: {result['avg_latency']:.2f}s")
    print(f"  Energy/Msg: {result['energy_per_msg_mJ']:.2f} mJ")

