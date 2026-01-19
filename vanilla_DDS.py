"""
Vanilla DDS Simulation (No DTN Routing)

Baseline simulation using RTI Connext DDS middleware without Spray & Focus routing.
Uses the same physics and modeling as the MQTT baseline for fair comparison.

Key characteristics:
- Same radio model as MQTT (calibrate_beta0, shannon_rate_3d, 1/r² pathloss)
- Same UAV mobility and energy model
- RTI Connext DDS for message transport (BEST_EFFORT or RELIABLE)
- No spray and focus routing - direct delivery to sink only
- DDS handles message transport semantics
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# --- RTI CONNEXT DDS IMPORTS (with fallback) ---
try:
    import rti.connextdds as dds
    from rti.types import struct
    RTI_AVAILABLE = True
except ImportError:
    RTI_AVAILABLE = False
    print("[WARNING] RTI Connext DDS not available - using simulated DDS mode")
    
    # Simulated struct decorator
    def struct(cls):
        return dataclass(cls)

# ==========================================
# 1) GLOBAL CONFIG
# ==========================================

GLOBAL_QOS = 1  # 0 = Best Effort, 1 = Reliable
NUM_UAVS = 8
SINK_ID = 0
NUM_SENSORS = 6

# DDS-specific overhead
RTPS_HEADER_BYTES = 64  # RTPS protocol overhead
DDS_BATCH_SIZE = 5  # RTI Connext batches ~5 messages per MTU

# ==========================================
# 2) PHYSICS / RADIO PROFILES (MATCHED TO MQTT)
# ==========================================

# ZigBee MTU constraint (IEEE 802.15.4 max frame = 127 bytes)
ZIGBEE_MAX_FRAME = 127
ZIGBEE_MAX_PAYLOAD = 100  # ~27 bytes MAC overhead

class PhyConst:
    H = 90.0  # UAV altitude (m)
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


# ZigBee 802.15.4 (2.4 GHz) - System-level energy from Siekkinen et al. (IEEE WCNC 2012)
# Measured: ~1 µJ/bit end-to-end. Using sender-only charging (E_rx=0).
ZIGBEE = PHYProfile("zigbee", B=250_000.0, P_tx=0.0774, N0=1e-13, 
                    E_tx_per_bit=1e-6, E_rx_per_bit=0)

# WiFi 802.11 (20 MHz) - System-level energy from Liu & Choi (ACM SIGMETRICS 2023)
# Measured: ~200 nJ/bit. Using sender-only charging (E_rx=0).
WIFI = PHYProfile("wifi", B=20_000_000.0, P_tx=1.5, N0=1e-13, 
                  E_tx_per_bit=2e-7, E_rx_per_bit=0)

REF_DIST = 100.0


def calibrate_beta0(prof: PHYProfile):
    """Calibrate path loss so SNR(100m) = 1 (0 dB)"""
    prof.beta0 = prof.N0 * (REF_DIST**2) / prof.P_tx


for _p in (ZIGBEE, WIFI):
    calibrate_beta0(_p)


def shannon_rate_3d(dist_3d: float, profile: PHYProfile) -> float:
    """Shannon rate (bps) with 1/r² pathloss"""
    if dist_3d <= 0.0:
        dist_3d = 1e-3
    snr = (profile.beta0 * profile.P_tx) / (profile.N0 * (dist_3d**2))
    return profile.B * math.log2(1.0 + snr)


def link_rate(pos1, pos2, is_ground_to_uav: bool) -> Tuple[float, float, PHYProfile]:
    """Return (rate bps, distance m, profile)"""
    d = float(np.linalg.norm(np.array(pos1) - np.array(pos2)))
    prof = ZIGBEE if is_ground_to_uav else WIFI
    return shannon_rate_3d(d, prof), d, prof


# ==========================================
# 3) DDS OVERHEAD MODEL (ZigBee vs WiFi)
# ==========================================

# --- RTPS Protocol Overhead ---
RTPS_MESSAGE_HEADER = 20      # RTPS Header
RTPS_DATA_SUBMSG_HEADER = 24  # DATA Submessage header

# --- WiFi (802.11) Transport Layer - for UAV↔UAV and UAV→Sink ---
WIFI_TCP_IP_OVERHEAD = 40     # IP (20) + UDP (8) + RTP/other (12)
WIFI_L2_OVERHEAD = 30         # WiFi 802.11 MAC/PHY overhead
WIFI_TRANSPORT_OVERHEAD = WIFI_TCP_IP_OVERHEAD + WIFI_L2_OVERHEAD  # = 70 bytes

# --- ZigBee (802.15.4) Transport Layer - for IoT/Sensor→UAV ---
ZIGBEE_L2_OVERHEAD = 15       # IEEE 802.15.4 MAC overhead
ZIGBEE_TRANSPORT_OVERHEAD = ZIGBEE_L2_OVERHEAD  # No TCP/IP for sensors


def dds_frame_size_zigbee(payload_bytes: int) -> int:
    """DDS/RTPS frame size for ZigBee links (IoT/Sensor→UAV)"""
    return ZIGBEE_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER + RTPS_DATA_SUBMSG_HEADER + payload_bytes
    # = 15 + 20 + 24 + payload = 59 + payload bytes


def dds_frame_size_wifi(payload_bytes: int) -> int:
    """DDS/RTPS frame size for WiFi links (UAV↔UAV, UAV→Sink)"""
    return WIFI_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER + RTPS_DATA_SUBMSG_HEADER + payload_bytes
    # = 70 + 20 + 24 + payload = 114 + payload bytes


# Legacy alias (for backward compatibility - defaults to WiFi)
def dds_frame_size(payload_bytes: int, batched: bool = False) -> int:
    """Legacy function - uses WiFi overhead for UAV links"""
    return dds_frame_size_wifi(payload_bytes)


# ==========================================
# 4) DDS MESSAGE TYPE
# ==========================================

@struct
class SensorMessage:
    """RTI Connext DDS topic type for sensor data"""
    msg_id: int = 0
    source_id: int = 0
    creation_time_ms: int = 0
    hop_count: int = 0
    qos_level: int = 0


# ==========================================
# 5) DDS INTERFACE (RTI or Simulated)
# ==========================================

# Shared message queue for simulated DDS
_SIMULATED_DDS_QUEUE: List = []


class SimulatedDDSInterface:
    """Simulated DDS interface for testing without RTI installed"""
    
    def __init__(self, node_id: int, reliable: bool = True):
        self.node_id = node_id
        self.reliable = reliable
        self.writer = SimulatedWriter(node_id)
        self.reader = SimulatedReader(node_id)


class SimulatedWriter:
    def __init__(self, node_id: int):
        self.node_id = node_id
    
    def write(self, msg):
        """Write message to shared queue"""
        _SIMULATED_DDS_QUEUE.append(msg)


class SimulatedReader:
    def __init__(self, node_id: int):
        self.node_id = node_id
    
    def take_data(self):
        """Take all messages from shared queue"""
        global _SIMULATED_DDS_QUEUE
        msgs = _SIMULATED_DDS_QUEUE.copy()
        _SIMULATED_DDS_QUEUE.clear()
        return msgs


class RTIDDSInterface:
    """RTI Connext DDS wrapper for UAV communication"""
    
    def __init__(self, node_id: int, reliable: bool = True):
        self.node_id = node_id
        self.reliable = reliable
        
        # Create participant
        pqos = dds.DomainParticipantQos()
        self.participant = dds.DomainParticipant(domain_id=0, qos=pqos)
        self.topic = dds.Topic(self.participant, "VanillaDDS", SensorMessage)
        
        self.publisher = dds.Publisher(self.participant)
        self.subscriber = dds.Subscriber(self.participant)
        
        # Writer QoS
        wqos = dds.DataWriterQos(self.publisher.default_datawriter_qos)
        if reliable:
            wqos.reliability.kind = dds.ReliabilityKind.RELIABLE
            wqos.reliability.max_blocking_time = dds.Duration(seconds=0, nanoseconds=100_000_000)
        else:
            wqos.reliability.kind = dds.ReliabilityKind.BEST_EFFORT
        wqos.durability.kind = dds.DurabilityKind.VOLATILE
        wqos.history.kind = dds.HistoryKind.KEEP_LAST
        wqos.history.depth = 100
        
        self.writer = dds.DataWriter(self.publisher, self.topic, wqos)
        
        # Reader QoS
        rqos = dds.DataReaderQos(self.subscriber.default_datareader_qos)
        if reliable:
            rqos.reliability.kind = dds.ReliabilityKind.RELIABLE
        else:
            rqos.reliability.kind = dds.ReliabilityKind.BEST_EFFORT
        rqos.durability.kind = dds.DurabilityKind.VOLATILE
        rqos.history.kind = dds.HistoryKind.KEEP_LAST
        rqos.history.depth = 100
        
        self.reader = dds.DataReader(self.subscriber, self.topic, rqos)


# Select appropriate interface based on RTI availability
def DDSInterface(node_id: int, reliable: bool = True):
    """Factory function to create DDS interface"""
    global RTI_AVAILABLE
    if RTI_AVAILABLE:
        try:
            return RTIDDSInterface(node_id, reliable)
        except Exception as e:
            print(f"[WARNING] RTI DDS failed: {e}")
            print("[WARNING] Falling back to simulated DDS mode")
            RTI_AVAILABLE = False
            return SimulatedDDSInterface(node_id, reliable)
    else:
        return SimulatedDDSInterface(node_id, reliable)


# ==========================================
# 6) UAV AGENT (VANILLA - NO DTN)
# ==========================================

@dataclass
class BufferedMessage:
    """Message held in UAV buffer waiting for sink delivery"""
    msg_id: int
    source_id: int
    creation_time: float
    hop_count: int
    qos: int


class VanillaDDSAgent:
    MAX_BUFFER = 50
    
    def __init__(self, uid: int, pos: List[float], is_sink: bool = False, 
                 reliable: bool = True, area_size: float = 500):
        self.id = uid
        self.pos = np.array(pos, dtype=float)
        self.is_sink = is_sink
        self.area_size = area_size
        self.energy = 300000.0  # 300 kJ - ~28 min flight time
        self.radio_tx_energy = 0.0
        self.radio_rx_energy = 0.0
        
        # Message buffer (no DTN - just hold until sink contact)
        self.buffer: List[BufferedMessage] = []
        self.seen_msgs = set()
        
        # DDS interface
        self.dds = DDSInterface(uid, reliable=reliable)
        
        # For sink
        if is_sink:
            self.delivered_ids = set()
        
        # Mobility
        self.vel = np.array([0.0, 0.0, 0.0])
        self._waypoint_timer = 0.0
    
    def move(self, dt: float):
        if self.is_sink:
            return
        
        # Random waypoint mobility (matching spray_focus_DDS.py)
        if not hasattr(self, 'waypoints') or not self.waypoints:
            self.waypoints = [np.array([random.uniform(100, self.area_size-100), 
                                        random.uniform(100, self.area_size-100), 
                                        PhyConst.H]) for _ in range(5)]
        
        speed = 20.0  # Fixed 20 m/s (matching S&F simulations)
        target = self.waypoints[0]
        direction = target - self.pos
        dist = np.linalg.norm(direction)
        step = speed * dt
        
        if dist <= step:
            self.pos = target.copy()
            self.waypoints.pop(0)
        else:
            self.pos += (direction / dist) * step
        
        flight_power = self.flight_power(speed)
        self.energy -= flight_power * dt
    
    def flight_power(self, velocity: float) -> float:
        term1 = PhyConst.P_C * (1 + (3 * velocity**2) / (PhyConst.U_TIP**2))
        term2 = PhyConst.WEIGHT * (
            math.sqrt(1 + (velocity**4) / (4 * PhyConst.V_0**4)) -
            (velocity**2) / (2 * PhyConst.V_0**2)
        )
        term3 = 0.5 * PhyConst.D_0 * PhyConst.RHO * PhyConst.S_SOL * PhyConst.AREA * (velocity**3)
        return term1 + term2 + term3
    
    def publish_to_sink(self, msg: BufferedMessage) -> Tuple[bool, float, float]:
        """Publish message via DDS. Returns (success, tx_energy, bytes_sent)"""
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
            bits = frame_bytes * 8
            tx_energy = bits * WIFI.E_tx_per_bit
            return True, tx_energy, frame_bytes
        except Exception:
            return False, 0.0, 0


# ==========================================
# 7) VANILLA DDS SIMULATION
# ==========================================

def run_vanilla_dds_simulation(config: dict, verbose: bool = False) -> dict:
    """
    Run vanilla DDS simulation with direct-delivery only (no DTN routing).
    Same physics as MQTT baseline but using RTI Connext DDS.
    """
    global NUM_UAVS, NUM_SENSORS, GLOBAL_QOS, SINK_ID
    global _SIMULATED_DDS_QUEUE
    _SIMULATED_DDS_QUEUE = []
    
    NUM_UAVS = config.get("NUM_UAVS", 8)
    NUM_SENSORS = config.get("NUM_SENSORS", 6)
    GLOBAL_QOS = config.get("GLOBAL_QOS", 1)
    SINK_ID = 0
    
    AREA_SIZE = config.get("AREA_SIZE", 500)
    SINK_MOBILE = config.get("SINK_MOBILE", True)
    RELIABLE = GLOBAL_QOS == 1
    
    # Set WiFi payload size from config (for payload sweep - sensor payload stays fixed)
    if "WIFI_PAYLOAD_BYTES" in config:
        PhyConst.WIFI_DATA_PAYLOAD_BYTES = config["WIFI_PAYLOAD_BYTES"]
    
    duration = config.get("DURATION", 1500.0)
    dt = 0.1
    
    # Initialize UAVs
    agents: Dict[int, VanillaDDSAgent] = {}
    
    # Sink - is_sink=True always, but can be mobile or static for movement
    if SINK_MOBILE:
        sink_pos = [random.uniform(100, AREA_SIZE-100), random.uniform(100, AREA_SIZE-100), PhyConst.H]
    else:
        sink_pos = [AREA_SIZE/2, AREA_SIZE/2, PhyConst.H]
    agents[SINK_ID] = VanillaDDSAgent(SINK_ID, sink_pos, is_sink=True, 
                                       reliable=RELIABLE, area_size=AREA_SIZE)
    # For mobile sink, we override the movement behavior
    if SINK_MOBILE:
        agents[SINK_ID].is_sink = False  # Allow movement
    agents[SINK_ID].delivered_ids = set()  # Always ensure this exists
    
    for i in range(1, NUM_UAVS):
        agents[i] = VanillaDDSAgent(i, [random.uniform(100, AREA_SIZE-100), 
                                        random.uniform(100, AREA_SIZE-100), PhyConst.H],
                                    reliable=RELIABLE, area_size=AREA_SIZE)
    
    # Sensors
    iot_nodes = [
        np.array([100 + (i % 7) * (AREA_SIZE-200)/6, 
                  100 + (i // 7) * (AREA_SIZE-200)/6, 10.0])
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
        for uid, agent in agents.items():
            if uid in [SINK_ID] and not SINK_MOBILE:
                continue
            agent.move(dt)
        
        # 2) DIRECT DELIVERY TO SINK VIA DDS (NO UAV-TO-UAV RELAY)
        sink = agents[SINK_ID]
        
        for i in range(1, NUM_UAVS):
            ai = agents[i]
            if not ai.buffer:
                continue
            
            rate_bps, d, prof = link_rate(ai.pos, sink.pos, is_ground_to_uav=False)
            
            if rate_bps < prof.B:
                continue  # Link down
            
            max_bytes_this_step = (rate_bps * dt) / 8.0
            bytes_sent = 0
            msgs_to_remove = []
            
            for msg in ai.buffer:
                if hasattr(sink, 'delivered_ids') and msg.msg_id in sink.delivered_ids:
                    msgs_to_remove.append(msg)
                    continue
                
                frame_bytes = dds_frame_size(PhyConst.WIFI_DATA_PAYLOAD_BYTES)
                if bytes_sent + frame_bytes > max_bytes_this_step:
                    break
                
                # Transmit via DDS
                success, tx_e, sent_bytes = ai.publish_to_sink(msg)
                if not success:
                    continue
                
                ai.radio_tx_energy += tx_e
                data_wifi_tx_energy += tx_e
                
                # RX energy at sink
                rx_e = sent_bytes * 8 * prof.E_rx_per_bit
                sink.radio_rx_energy += rx_e
                data_wifi_rx_energy += rx_e
                
                data_bytes_sent += sent_bytes
                bytes_sent += sent_bytes
                msgs_to_remove.append(msg)
            
            for m in msgs_to_remove:
                if m in ai.buffer:
                    ai.buffer.remove(m)
        
        # 3) SINK READS DDS MESSAGES
        try:
            samples = sink.dds.reader.take_data()
            for data in samples:
                mid = data.msg_id
                if mid in sink.delivered_ids:
                    continue
                sink.delivered_ids.add(mid)
                
                total_delivered += 1
                sink_delivery_events += 1
                hop_counts.append(data.hop_count)
                latencies.append(sim_time - (data.creation_time_ms / 1000.0))
        except Exception:
            pass
        
        # 4) SENSOR → UAV UPLOAD (ZigBee)
        for s, src_pos in enumerate(iot_nodes):
            if not sensor_queues[s]:
                continue
            
            best_uav = min(range(NUM_UAVS), 
                          key=lambda k: np.linalg.norm(agents[k].pos - src_pos))
            rate, d, prof = link_rate(src_pos, agents[best_uav].pos, is_ground_to_uav=True)
            
            if rate < prof.B:
                msg_id, qos_val, t0 = sensor_queues[s][0]
                if qos_val == 0:
                    sensor_queues[s].pop(0)
                continue
            
            msg_id, qos_val, t0 = sensor_queues[s][0]
            
            # Sensor TX energy (ZigBee)
            frame_bytes = dds_frame_size_zigbee(PhyConst.SENSOR_PAYLOAD_BYTES)
            
            # ZigBee max frame constraint
            if frame_bytes > ZIGBEE_MAX_FRAME:
                continue
            
            # Energy accounting
            sensor_tx_energy += frame_bytes * 8 * prof.E_tx_per_bit
            rx_e = frame_bytes * 8 * prof.E_rx_per_bit
            agents[best_uav].energy -= rx_e
            agents[best_uav].radio_rx_energy += rx_e
            
            # CRITICAL FIX: If sink collects directly, count as instant delivery
            if best_uav == SINK_ID:
                sensor_queues[s].pop(0)  # Pop AFTER successful handling
                if msg_id not in sink.delivered_ids:
                    sink.delivered_ids.add(msg_id)
                    total_delivered += 1
                    sink_delivery_events += 1
                    hop_counts.append(0)  # Direct collection = 0 hops
                    latencies.append(sim_time - t0)
            else:
                # Regular UAV: put in buffer for later delivery to sink
                if len(agents[best_uav].buffer) < agents[best_uav].MAX_BUFFER:
                    sensor_queues[s].pop(0)  # Pop AFTER successful buffer insertion
                    new_msg = BufferedMessage(
                        msg_id=msg_id, source_id=s, creation_time=t0,
                        hop_count=0, qos=qos_val
                    )
                    agents[best_uav].buffer.append(new_msg)
                    agents[best_uav].seen_msgs.add(msg_id)
                else:
                    # Buffer full - BE drops, Reliable stays for retry
                    if qos_val == 0:
                        sensor_queues[s].pop(0)
    
    # Compute results
    total_uav_tx = sum(a.radio_tx_energy for a in agents.values())
    total_uav_rx = sum(a.radio_rx_energy for a in agents.values())
    total_uav_radio = total_uav_tx + total_uav_rx
    
    total_data_wifi_energy = data_wifi_tx_energy + data_wifi_rx_energy
    total_data_zigbee = sensor_tx_energy
    
    results = {
        "pdr": 100.0 * total_delivered / max(1, total_generated),
        "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency": float(np.median(latencies)) if latencies else 0.0,
        "avg_hops": float(np.mean(hop_counts)) if hop_counts else 0.0,
        "overhead_factor": 1.0,  # No relay overhead in vanilla DDS
        "total_generated": total_generated,
        "total_delivered": total_delivered,
        "uav_relay_events": 0,
        "sink_delivery_events": sink_delivery_events,
        "control_messages_sent": 0,  # DDS handles discovery internally
        "control_energy": 0.0,
        "data_wifi_energy": total_data_wifi_energy,
        "data_zigbee_energy": total_data_zigbee,
        "total_uav_radio_energy": total_uav_radio,
        "energy_per_msg_mJ": (total_uav_radio / max(1, total_delivered)) * 1000
    }
    
    if verbose:
        print(f"  [VANILLA DDS] PDR: {results['pdr']:.1f}% | Latency: {results['avg_latency']:.2f}s")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("VANILLA DDS SIMULATION (No DTN Routing)")
    print("="*60)
    
    config = {"NUM_UAVS": 8, "NUM_SENSORS": 6, "GLOBAL_QOS": 1}
    result = run_vanilla_dds_simulation(config, verbose=True)
    
    print(f"\nVanilla DDS Results:")
    print(f"  PDR: {result['pdr']:.2f}%")
    print(f"  Avg Latency: {result['avg_latency']:.2f}s")
    print(f"  Energy/Msg: {result['energy_per_msg_mJ']:.2f} mJ")
    print("="*60)

