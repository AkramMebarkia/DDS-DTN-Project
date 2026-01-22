"""
DDS Spray & Focus Simulation

Enhanced DDS simulation using RTI Connext DDS with Spray & Focus DTN routing.
Combines DDS middleware from vanilla_DDS with routing algorithm from EdgeDrone.

Key characteristics:
- Same radio model as MQTT/vanilla DDS (calibrate_beta0, shannon_rate_3d, 1/r² pathloss)
- Same UAV mobility and energy model
- RTI Connext DDS for message transport (BEST_EFFORT or RELIABLE)
- Spray & Focus DTN routing for message delivery
- No batching - each message has full RTPS overhead
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

# Spray & Focus initial token budget
INITIAL_TOKENS = 10

# ==========================================
# DDS/RTPS PROTOCOL OVERHEAD (per OMG RTPS Spec)
# ==========================================

# --- WiFi (802.11) Transport Layer - for UAV↔UAV and UAV→Sink ---
# DDS uses UDP/IP over WiFi
WIFI_UDP_HEADER = 8       # UDP header
WIFI_IP_HEADER = 20       # IPv4 header  
WIFI_L2_OVERHEAD = 30     # WiFi 802.11 MAC/PHY overhead
WIFI_TRANSPORT_OVERHEAD = WIFI_UDP_HEADER + WIFI_IP_HEADER + WIFI_L2_OVERHEAD  # = 58 bytes

# --- ZigBee (802.15.4) Transport Layer - for IoT/Sensor→UAV ---
# Sensors typically use direct ZigBee frames (no IP/UDP stack)
# IEEE 802.15.4 frame structure:
# - Frame Control: 2 bytes
# - Sequence Number: 1 byte
# - Addressing (short): 4-8 bytes (PAN ID + addresses)
# - FCS: 2 bytes
# Total MAC overhead: ~11-15 bytes (using short addressing)
ZIGBEE_L2_OVERHEAD = 15   # IEEE 802.15.4 MAC overhead (short addressing)
# No IP/UDP for simple sensor → UAV link (direct ZigBee)
ZIGBEE_TRANSPORT_OVERHEAD = ZIGBEE_L2_OVERHEAD  # = 15 bytes

# RTPS Message Header (fixed, per OMG RTPS 2.3 spec Section 8.3.3)
# - Protocol 'RTPS': 4 bytes
# - Version: 2 bytes
# - Vendor ID: 2 bytes
# - GUID Prefix: 12 bytes
RTPS_MESSAGE_HEADER = 20  # bytes

# DATA Submessage Header (per OMG RTPS 2.3 spec Section 8.3.7)
# - Submessage Header: 4 bytes (ID + flags + length)
# - Extra Flags: 2 bytes
# - Octets to Inline QoS: 2 bytes  
# - Reader Entity ID: 4 bytes
# - Writer Entity ID: 4 bytes
# - Sequence Number: 8 bytes
RTPS_DATA_SUBMSG_HEADER = 24  # bytes (fixed portion)

# ACKNACK Submessage (per OMG RTPS 2.3 spec Section 8.3.7.1)
# - Submessage Header: 4 bytes
# - Reader Entity ID: 4 bytes
# - Writer Entity ID: 4 bytes
# - First Seq Num: 8 bytes
# - Num Bits: 4 bytes
# - Count: 4 bytes
# - Bitmap: 0+ bytes (variable, minimum empty)
RTPS_ACKNACK_SUBMSG = 28  # bytes (minimum, no bitmap)

# ==========================================
# 2) PHYSICS / RADIO PROFILES (MATCHED TO MQTT/VANILLA)
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
    
    # ZigBee (IoT → UAV): Fixed small sensor payload (must fit in single frame)
    SENSOR_PAYLOAD_BYTES = 64  # Safe for ZigBee (≤ ZIGBEE_MAX_PAYLOAD)
    
    # WiFi (UAV ↔ UAV, UAV → Sink): Variable data payload
    WIFI_DATA_PAYLOAD_BYTES = 256  # Default for WiFi links (no MTU issue)
    
    # Control messages (small, works on both ZigBee and WiFi)
    CONTROL_PAYLOAD_BYTES = 12  # Utility value (8) + overhead (4)


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
# 3) DDS FRAME SIZE FUNCTIONS
# ==========================================

def dds_frame_size_wifi(payload_bytes: int) -> int:
    """
    DDS/RTPS frame size for WiFi links (UAV↔UAV, UAV→Sink).
    Includes: WiFi L2 + IP + UDP + RTPS Header + DATA Submsg + Payload
    """
    return WIFI_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER + RTPS_DATA_SUBMSG_HEADER + payload_bytes
    # = 58 + 20 + 24 + payload = 102 + payload bytes


def dds_frame_size_zigbee(payload_bytes: int) -> int:
    """
    DDS/RTPS frame size for ZigBee links (IoT/Sensor→UAV).
    Includes: ZigBee 802.15.4 L2 + RTPS Header + DATA Submsg + Payload
    Note: Sensors use direct ZigBee (no IP/UDP stack)
    """
    return ZIGBEE_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER + RTPS_DATA_SUBMSG_HEADER + payload_bytes
    # = 15 + 20 + 24 + payload = 59 + payload bytes


def dds_acknack_size_wifi() -> int:
    """
    DDS/RTPS ACKNACK frame size for WiFi links.
    Includes: WiFi L2 + IP + UDP + RTPS Header + ACKNACK Submessage
    """
    return WIFI_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER + RTPS_ACKNACK_SUBMSG
    # = 58 + 20 + 28 = 106 bytes


def dds_acknack_size_zigbee() -> int:
    """
    DDS/RTPS ACKNACK frame size for ZigBee links (rarely used, but for completeness).
    Includes: ZigBee 802.15.4 L2 + RTPS Header + ACKNACK Submessage
    """
    return ZIGBEE_TRANSPORT_OVERHEAD + RTPS_MESSAGE_HEADER + RTPS_ACKNACK_SUBMSG
    # = 15 + 20 + 28 = 63 bytes


# Legacy alias for backward compatibility (defaults to WiFi)
def dds_frame_size(payload_bytes: int) -> int:
    """Legacy function - uses WiFi overhead (for UAV↔UAV links)"""
    return dds_frame_size_wifi(payload_bytes)


def dds_acknack_size() -> int:
    """Legacy function - uses WiFi overhead"""
    return dds_acknack_size_wifi()


# ==========================================
# 4) DDS MESSAGE TYPES
# ==========================================

@struct
class SensorMessage:
    """RTI Connext DDS topic type for sensor data"""
    msg_id: int = 0
    source_id: int = 0
    creation_time_ms: int = 0
    hop_count: int = 0
    qos_level: int = 0
    tokens: int = 0


@struct
class ControlMessage:
    """RTI Connext DDS topic type for routing control (utility exchange)"""
    sender_id: int = 0
    utility_to_sink: float = 0.0


# ==========================================
# 5) DDS INTERFACE (RTI or Simulated)
# ==========================================

# Shared message queues for simulated DDS
_SIMULATED_DATA_QUEUE: List = []
_SIMULATED_CONTROL_QUEUE: List = []


class SimulatedDDSInterface:
    """Simulated DDS interface for testing without RTI installed"""
    
    def __init__(self, node_id: int, reliable: bool = True):
        self.node_id = node_id
        self.reliable = reliable
        self.data_writer = SimulatedDataWriter(node_id)
        self.data_reader = SimulatedDataReader(node_id)
        self.control_writer = SimulatedControlWriter(node_id)
        self.control_reader = SimulatedControlReader(node_id)


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
        msgs = [msg for sender, msg in _SIMULATED_DATA_QUEUE]
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
        msgs = [(sender, msg) for sender, msg in _SIMULATED_CONTROL_QUEUE]
        _SIMULATED_CONTROL_QUEUE.clear()
        return msgs


class RTIDDSInterface:
    """RTI Connext DDS wrapper for UAV communication"""
    
    def __init__(self, node_id: int, reliable: bool = True):
        self.node_id = node_id
        self.reliable = reliable
        
        # Create participant
        pqos = dds.DomainParticipantQos()
        self.participant = dds.DomainParticipant(domain_id=0, qos=pqos)
        
        # Data topic
        self.data_topic = dds.Topic(self.participant, "SprayFocusData", SensorMessage)
        self.control_topic = dds.Topic(self.participant, "SprayFocusControl", ControlMessage)
        
        self.publisher = dds.Publisher(self.participant)
        self.subscriber = dds.Subscriber(self.participant)
        
        # Data Writer QoS
        wqos = dds.DataWriterQos(self.publisher.default_datawriter_qos)
        if reliable:
            wqos.reliability.kind = dds.ReliabilityKind.RELIABLE
            wqos.reliability.max_blocking_time = dds.Duration(seconds=0, nanoseconds=100_000_000)
        else:
            wqos.reliability.kind = dds.ReliabilityKind.BEST_EFFORT
        wqos.durability.kind = dds.DurabilityKind.VOLATILE
        wqos.history.kind = dds.HistoryKind.KEEP_LAST
        wqos.history.depth = 100
        
        self.data_writer = dds.DataWriter(self.publisher, self.data_topic, wqos)
        self.control_writer = dds.DataWriter(self.publisher, self.control_topic, wqos)
        
        # Reader QoS
        rqos = dds.DataReaderQos(self.subscriber.default_datareader_qos)
        if reliable:
            rqos.reliability.kind = dds.ReliabilityKind.RELIABLE
        else:
            rqos.reliability.kind = dds.ReliabilityKind.BEST_EFFORT
        rqos.durability.kind = dds.DurabilityKind.VOLATILE
        rqos.history.kind = dds.HistoryKind.KEEP_LAST
        rqos.history.depth = 100
        
        self.data_reader = dds.DataReader(self.subscriber, self.data_topic, rqos)
        self.control_reader = dds.DataReader(self.subscriber, self.control_topic, rqos)


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
# 6) DATA STRUCTURES
# ==========================================

@dataclass
class SprayMessage:
    """DTN message in UAV buffer with tokens"""
    msg_id: int
    source_id: int
    creation_time: float
    hop_count: int
    tokens: int
    qos: int
    payload_bytes: int = PhyConst.WIFI_DATA_PAYLOAD_BYTES  # 64B for sensors, 256B for UAV data
    
    def payload_size(self) -> int:
        """Return configured payload size"""
        return self.payload_bytes


# ==========================================
# 7) SPRAY & FOCUS DDS AGENT
# ==========================================

class SprayFocusDDSAgent:
    MAX_BUFFER = 250
    
    def __init__(self, uid: int, pos: List[float], is_sink: bool = False, 
                 reliable: bool = True, area_size: float = 500):
        self.id = uid
        self.pos = np.array(pos, dtype=float)
        self.is_sink = is_sink
        self.area_size = area_size
        self.energy = 300000.0  # 300 kJ - ~28 min flight time
        self.radio_tx_energy = 0.0
        self.radio_rx_energy = 0.0
        
        # DTN state
        self.buffer: List[SprayMessage] = []
        self.seen_msgs = set()
        self.encounter_timers: Dict[int, float] = {}
        
        # Initialize encounter timers
        for i in range(NUM_UAVS):
            self.encounter_timers[i] = 9999.0
        self.encounter_timers[uid] = 0.0
        
        # DDS interface
        self.dds = DDSInterface(uid, reliable=reliable)
        
        # For sink
        if is_sink:
            self.delivered_ids = set()
            self.sink_received_count = 0
        
        # Mobility
        self.vel = np.array([0.0, 0.0, 0.0])
        self._waypoint_timer = 0.0
        self.waypoints = []
    
    def move(self, dt: float):
        """Update position and encounter timers"""
        if self.is_sink:
            return
        
        # Age encounter timers
        for k in self.encounter_timers.keys():
            self.encounter_timers[k] += dt
        self.encounter_timers[self.id] = 0.0
        
        # Random waypoint mobility
        if not self.waypoints:
            self.waypoints = [np.array([random.uniform(100, self.area_size-100), 
                                        random.uniform(100, self.area_size-100), 
                                        PhyConst.H]) for _ in range(5)]
        
        speed = 20.0  # m/s
        target = self.waypoints[0]
        direction = target - self.pos
        dist = np.linalg.norm(direction)
        step = speed * dt
        
        if dist <= step:
            self.pos = target.copy()
            self.waypoints.pop(0)
        else:
            self.pos += (direction / dist) * step
        
        # Flight energy
        flight_power = self.flight_power(speed)
        self.energy -= flight_power * dt
    
    def flight_power(self, velocity: float) -> float:
        """Calculate flight power consumption (Watts)"""
        P0 = (PhyConst.DELTA / 8.0) * PhyConst.RHO * PhyConst.S_SOL * PhyConst.AREA \
             * (PhyConst.OMEGA ** 3) * (PhyConst.R_RAD ** 3)
        P_ind = 1.1 * (PhyConst.WEIGHT ** 1.5) / math.sqrt(2 * PhyConst.RHO * PhyConst.AREA)
        
        if velocity < 0.1:
            return P0 + P_ind
        
        term1 = P0 * (1.0 + 3.0 * velocity ** 2 / (PhyConst.U_TIP ** 2))
        term2 = P_ind * math.sqrt(
            math.sqrt(1.0 + (velocity ** 4) / (4 * PhyConst.V_0 ** 4)) - 
            (velocity ** 2) / (2 * PhyConst.V_0 ** 2)
        )
        term3 = 0.5 * PhyConst.D_0 * PhyConst.RHO * PhyConst.S_SOL * PhyConst.AREA * (velocity ** 3)
        
        return term1 + term2 + term3
    
    def get_utility(self) -> float:
        """Get utility to sink (lower is better)"""
        return self.encounter_timers[SINK_ID]


# ==========================================
# 8) SPRAY & FOCUS DDS SIMULATION
# ==========================================

def run_spray_focus_dds_simulation(config: dict, verbose: bool = False) -> dict:
    """
    Run DDS simulation with Spray & Focus DTN routing.
    Same physics as MQTT/vanilla but using RTI Connext DDS with DTN.
    """
    global NUM_UAVS, NUM_SENSORS, GLOBAL_QOS, SINK_ID, INITIAL_TOKENS
    
    NUM_UAVS = config.get("NUM_UAVS", 8)
    NUM_SENSORS = config.get("NUM_SENSORS", 6)
    GLOBAL_QOS = config.get("GLOBAL_QOS", 1)
    INITIAL_TOKENS = config.get("INITIAL_TOKENS", 10)
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
    agents: Dict[int, SprayFocusDDSAgent] = {}
    
    # Sink
    if SINK_MOBILE:
        sink_pos = [random.uniform(100, AREA_SIZE-100), random.uniform(100, AREA_SIZE-100), PhyConst.H]
    else:
        sink_pos = [AREA_SIZE/2, AREA_SIZE/2, PhyConst.H]
    agents[SINK_ID] = SprayFocusDDSAgent(SINK_ID, sink_pos, is_sink=True, 
                                          reliable=RELIABLE, area_size=AREA_SIZE)
    # Mobile sink needs to move
    if SINK_MOBILE:
        agents[SINK_ID].is_sink = False
    agents[SINK_ID].delivered_ids = set()
    agents[SINK_ID].sink_received_count = 0
    
    for i in range(1, NUM_UAVS):
        agents[i] = SprayFocusDDSAgent(i, [random.uniform(100, AREA_SIZE-100), 
                                            random.uniform(100, AREA_SIZE-100), PhyConst.H],
                                        reliable=RELIABLE, area_size=AREA_SIZE)
    
    # Initialize sensors with well-spaced random positions (reproducible)
    def generate_spread_sensors(num_sensors, area_size, seed=42):
        """Generate well-spaced sensor positions using seeded random with minimum distance."""
        rng = np.random.RandomState(seed)
        margin = 100  # Keep sensors away from edges
        min_dist = (area_size - 2 * margin) / (num_sensors ** 0.5 + 1)  # Minimum spacing
        
        positions = []
        max_attempts = 1000
        
        for _ in range(num_sensors):
            for attempt in range(max_attempts):
                x = rng.uniform(margin, area_size - margin)
                y = rng.uniform(margin, area_size - margin)
                
                # Check distance from all existing sensors
                valid = True
                for px, py, _ in positions:
                    if np.sqrt((x - px)**2 + (y - py)**2) < min_dist:
                        valid = False
                        break
                
                if valid or attempt == max_attempts - 1:
                    positions.append([x, y, 10.0])  # z=10m ground level
                    break
        
        return [np.array(pos) for pos in positions]
    
    iot_nodes = generate_spread_sensors(NUM_SENSORS, AREA_SIZE, seed=42)
    
    sim_time = 0.0
    SENSOR_RATE = 0.5
    SENSOR_BUF_MAX = 50
    sensor_queues: List[List[Tuple[int, int, float]]] = [[] for _ in range(NUM_SENSORS)]
    MSG_COUNTER = 0
    
    # Statistics
    total_generated = 0
    total_delivered = 0
    uav_relay_events = 0
    spray_events = 0
    focus_events = 0
    sink_delivery_events = 0
    control_messages_sent = 0
    latencies = []
    hop_counts = []
    
    # Energy tracking
    sensor_tx_energy = 0.0
    control_tx_energy = 0.0
    control_rx_energy = 0.0
    data_wifi_tx_energy = 0.0
    data_wifi_rx_energy = 0.0
    control_bytes_sent = 0
    data_bytes_sent = 0
    
    # Main loop
    while sim_time < duration:
        sim_time += dt
        
        # ================================================
        # 0) SENSOR DATA GENERATION
        # ================================================
        for s in range(NUM_SENSORS):
            if random.random() < SENSOR_RATE * dt:
                MSG_COUNTER += 1
                qos_val = GLOBAL_QOS
                sensor_queues[s].append((MSG_COUNTER, qos_val, sim_time))
                if len(sensor_queues[s]) > SENSOR_BUF_MAX:
                    sensor_queues[s].pop(0)
                total_generated += 1
        
        # ================================================
        # 1) UAV MOVEMENT
        # ================================================
        for uid, agent in agents.items():
            if uid == SINK_ID and not SINK_MOBILE:
                continue
            agent.move(dt)
        
        # ================================================
        # 2) ENCOUNTER DETECTION & DV UPDATES
        # ================================================
        for i in range(NUM_UAVS):
            for j in range(i + 1, NUM_UAVS):
                rate, d, _prof = link_rate(agents[i].pos, agents[j].pos, is_ground_to_uav=False)
                
                if rate >= WIFI.B:
                    # Direct encounter
                    agents[i].encounter_timers[j] = 0.0
                    agents[j].encounter_timers[i] = 0.0
                    
                    # Distance Vector update
                    t_meet = d / 20.0
                    t_i_sink = agents[i].encounter_timers[SINK_ID]
                    t_j_sink = agents[j].encounter_timers[SINK_ID]
                    
                    if t_j_sink + t_meet < t_i_sink:
                        agents[i].encounter_timers[SINK_ID] = t_j_sink + t_meet
                    if t_i_sink + t_meet < t_j_sink:
                        agents[j].encounter_timers[SINK_ID] = t_i_sink + t_meet
        
        # ================================================
        # 3) UAV → SINK DELIVERY (PRIORITIZED - run before relay)
        # ================================================
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
                
                frame_bytes = dds_frame_size(msg.payload_size())
                if bytes_sent + frame_bytes > max_bytes_this_step:
                    break
                
                # Transmit via DDS
                bits = frame_bytes * 8
                tx_e = bits * prof.E_tx_per_bit
                rx_e = bits * prof.E_rx_per_bit
                
                ai.energy -= tx_e
                ai.radio_tx_energy += tx_e
                data_wifi_tx_energy += tx_e
                sink.energy -= rx_e
                sink.radio_rx_energy += rx_e
                data_wifi_rx_energy += rx_e
                
                # ACKNACK energy for RELIABLE QoS (sink sends ACKNACK)
                if GLOBAL_QOS == 1:
                    ack_bytes = dds_acknack_size()
                    ack_bits = ack_bytes * 8
                    ack_tx_e = ack_bits * prof.E_tx_per_bit
                    ack_rx_e = ack_bits * prof.E_rx_per_bit
                    sink.energy -= ack_tx_e
                    sink.radio_tx_energy += ack_tx_e
                    data_wifi_tx_energy += ack_tx_e
                    ai.energy -= ack_rx_e
                    ai.radio_rx_energy += ack_rx_e
                    data_wifi_rx_energy += ack_rx_e
                
                data_bytes_sent += frame_bytes
                bytes_sent += frame_bytes
                
                # Mark as delivered
                sink.delivered_ids.add(msg.msg_id)
                sink.sink_received_count += 1
                
                total_delivered += 1
                sink_delivery_events += 1
                hop_counts.append(msg.hop_count)
                latencies.append(sim_time - msg.creation_time)
                
                msgs_to_remove.append(msg)
            
            for m in msgs_to_remove:
                if m in ai.buffer:
                    ai.buffer.remove(m)
        
        # ================================================
        # 3b) GLOBAL DUPLICATE PURGE
        # ================================================
        if total_delivered > 0 and int(sim_time * 10) % 10 == 0:
            delivered_ids = sink.delivered_ids.copy()
            for uid, a in agents.items():
                if uid == SINK_ID or not a.buffer:
                    continue
                a.buffer = [m for m in a.buffer if m.msg_id not in delivered_ids]
        
        # ================================================
        # 4) SPRAY & FOCUS ROUTING (UAV ↔ UAV)
        # ================================================
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
                    continue  # Link down
                
                # Separate spray and focus messages
                spray_messages = [m for m in ai.buffer if m.tokens > 1]
                focus_messages = [m for m in ai.buffer if m.tokens == 1]
                
                # Bandwidth limit for this timestep
                max_bytes_this_step = (rate * dt) / 8.0
                bytes_sent_this_step = 0
                
                # ========================================
                # SPRAY PHASE (tokens > 1) - NO CONTROL
                # ========================================
                for msg in spray_messages:
                    if msg.msg_id in aj.seen_msgs:
                        continue
                    if len(aj.buffer) >= aj.MAX_BUFFER:
                        continue
                    
                    send_tokens = msg.tokens // 2
                    if send_tokens <= 0:
                        continue
                    
                    # Check bandwidth capacity
                    frame_bytes = dds_frame_size(msg.payload_size())
                    if bytes_sent_this_step + frame_bytes > max_bytes_this_step:
                        break
                    
                    # Send DATA via DDS
                    bits = frame_bytes * 8
                    tx_e = bits * prof.E_tx_per_bit
                    rx_e = bits * prof.E_rx_per_bit
                    
                    # Energy accounting
                    ai.energy -= tx_e
                    ai.radio_tx_energy += tx_e
                    data_wifi_tx_energy += tx_e
                    aj.energy -= rx_e
                    aj.radio_rx_energy += rx_e
                    data_wifi_rx_energy += rx_e
                    
                    # ACKNACK energy for RELIABLE QoS (receiver sends ACKNACK)
                    if GLOBAL_QOS == 1:
                        ack_bytes = dds_acknack_size()
                        ack_bits = ack_bytes * 8
                        ack_tx_e = ack_bits * prof.E_tx_per_bit
                        ack_rx_e = ack_bits * prof.E_rx_per_bit
                        aj.energy -= ack_tx_e
                        aj.radio_tx_energy += ack_tx_e
                        data_wifi_tx_energy += ack_tx_e
                        ai.energy -= ack_rx_e
                        ai.radio_rx_energy += ack_rx_e
                        data_wifi_rx_energy += ack_rx_e
                    
                    data_bytes_sent += frame_bytes
                    bytes_sent_this_step += frame_bytes
                    
                    # Create copy in neighbor
                    new_msg = SprayMessage(
                        msg_id=msg.msg_id,
                        source_id=msg.source_id,
                        creation_time=msg.creation_time,
                        hop_count=msg.hop_count + 1,
                        tokens=send_tokens,
                        qos=msg.qos,
                        payload_bytes=msg.payload_bytes,
                        
                    )
                    aj.buffer.append(new_msg)
                    aj.seen_msgs.add(msg.msg_id)
                    
                    # Update sender
                    msg.tokens -= send_tokens
                    
                    uav_relay_events += 1
                    spray_events += 1
                
                # ========================================
                # FOCUS PHASE (tokens == 1) - WITH CONTROL
                # ========================================
                if focus_messages:
                    # Estimate control exchange bytes (including ACKNACKs)
                    ctrl_frame_size = dds_frame_size(PhyConst.CONTROL_PAYLOAD_BYTES)
                    ack_frame_size = dds_acknack_size()
                    total_control_bytes = 2 * ctrl_frame_size + 2 * ack_frame_size  # Inquiry + ACK + Response + ACK
                    
                    if bytes_sent_this_step + total_control_bytes > max_bytes_this_step:
                        continue
                    
                    # STEP 1: UAV_i sends INQUIRY
                    ctrl_bits = ctrl_frame_size * 8
                    tx_e = ctrl_bits * prof.E_tx_per_bit
                    rx_e = ctrl_bits * prof.E_rx_per_bit
                    
                    ai.energy -= tx_e
                    ai.radio_tx_energy += tx_e
                    control_tx_energy += tx_e
                    aj.energy -= rx_e
                    aj.radio_rx_energy += rx_e
                    control_rx_energy += rx_e
                    
                    control_messages_sent += 1
                    control_bytes_sent += ctrl_frame_size
                    bytes_sent_this_step += ctrl_frame_size
                    
                    # ACKNACK for INQUIRY (UAV_j sends ACKNACK to UAV_i)
                    ack_bits = ack_frame_size * 8
                    ack_tx_e = ack_bits * prof.E_tx_per_bit
                    ack_rx_e = ack_bits * prof.E_rx_per_bit
                    aj.energy -= ack_tx_e
                    aj.radio_tx_energy += ack_tx_e
                    control_tx_energy += ack_tx_e
                    ai.energy -= ack_rx_e
                    ai.radio_rx_energy += ack_rx_e
                    control_rx_energy += ack_rx_e
                    bytes_sent_this_step += ack_frame_size
                    
                    # STEP 2: UAV_j sends RESPONSE
                    aj.energy -= tx_e
                    aj.radio_tx_energy += tx_e
                    control_tx_energy += tx_e
                    ai.energy -= rx_e
                    ai.radio_rx_energy += rx_e
                    control_rx_energy += rx_e
                    
                    control_messages_sent += 1
                    control_bytes_sent += ctrl_frame_size
                    bytes_sent_this_step += ctrl_frame_size
                    
                    # ACKNACK for RESPONSE (UAV_i sends ACKNACK to UAV_j)
                    ai.energy -= ack_tx_e
                    ai.radio_tx_energy += ack_tx_e
                    control_tx_energy += ack_tx_e
                    aj.energy -= ack_rx_e
                    aj.radio_rx_energy += ack_rx_e
                    control_rx_energy += ack_rx_e
                    bytes_sent_this_step += ack_frame_size
                    
                    # STEP 3: Use received utility for decision
                    my_utility = ai.get_utility()
                    nb_utility = aj.get_utility()
                    t_meet = d / 20.0
                    
                    # STEP 4: Forward if neighbor is better
                    for msg in focus_messages:
                        if (nb_utility - t_meet) < my_utility:
                            if msg.msg_id in aj.seen_msgs:
                                continue
                            if len(aj.buffer) >= aj.MAX_BUFFER:
                                continue
                            
                            frame_bytes = dds_frame_size(msg.payload_size())
                            if bytes_sent_this_step + frame_bytes > max_bytes_this_step:
                                break
                            
                            # Send DATA via DDS
                            bits = frame_bytes * 8
                            tx_e = bits * prof.E_tx_per_bit
                            rx_e = bits * prof.E_rx_per_bit
                            
                            ai.energy -= tx_e
                            ai.radio_tx_energy += tx_e
                            data_wifi_tx_energy += tx_e
                            aj.energy -= rx_e
                            aj.radio_rx_energy += rx_e
                            data_wifi_rx_energy += rx_e
                            
                            # ACKNACK energy for RELIABLE QoS (receiver sends ACKNACK)
                            if GLOBAL_QOS == 1:
                                ack_bytes = dds_acknack_size()
                                ack_bits = ack_bytes * 8
                                ack_tx_e = ack_bits * prof.E_tx_per_bit
                                ack_rx_e = ack_bits * prof.E_rx_per_bit
                                aj.energy -= ack_tx_e
                                aj.radio_tx_energy += ack_tx_e
                                data_wifi_tx_energy += ack_tx_e
                                ai.energy -= ack_rx_e
                                ai.radio_rx_energy += ack_rx_e
                                data_wifi_rx_energy += ack_rx_e
                            
                            data_bytes_sent += frame_bytes
                            bytes_sent_this_step += frame_bytes
                            
                            # Create copy (FOCUS: move, not copy)
                            new_msg = SprayMessage(
                                msg_id=msg.msg_id,
                                source_id=msg.source_id,
                                creation_time=msg.creation_time,
                                hop_count=msg.hop_count + 1,
                                tokens=1,
                                qos=msg.qos,
                                payload_bytes=msg.payload_bytes,
                                
                            )
                            aj.buffer.append(new_msg)
                            aj.seen_msgs.add(msg.msg_id)
                            
                            # FOCUS: Remove from sender
                            if msg in ai.buffer:
                                ai.buffer.remove(msg)
                            
                            uav_relay_events += 1
                            focus_events += 1
        
        # ================================================
        # 5) SENSOR → UAV UPLOAD
        # ================================================
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
            
            # Sensor TX energy (ZigBee 802.15.4 frame)
            frame_bytes = dds_frame_size_zigbee(PhyConst.SENSOR_PAYLOAD_BYTES)
            
            # ZigBee max frame constraint
            if frame_bytes > ZIGBEE_MAX_FRAME:
                continue
            
            # Energy accounting
            sensor_tx_energy += frame_bytes * 8 * prof.E_tx_per_bit
            rx_e = frame_bytes * 8 * prof.E_rx_per_bit
            agents[best_uav].energy -= rx_e
            agents[best_uav].radio_rx_energy += rx_e
            
            # CRITICAL FIX: If sink collects directly, count as instant delivery (like vanilla DDS)
            if best_uav == SINK_ID:
                sensor_queues[s].pop(0)  # Pop AFTER successful handling
                if msg_id not in agents[SINK_ID].delivered_ids:
                    agents[SINK_ID].delivered_ids.add(msg_id)
                    agents[SINK_ID].sink_received_count += 1
                    total_delivered += 1
                    sink_delivery_events += 1
                    hop_counts.append(0)  # Direct collection = 0 hops
                    latencies.append(sim_time - t0)
            else:
                # Regular UAV: put in buffer for S&F routing
                if len(agents[best_uav].buffer) < agents[best_uav].MAX_BUFFER:
                    sensor_queues[s].pop(0)  # Pop AFTER successful buffer insertion
                    new_msg = SprayMessage(
                        msg_id=msg_id, source_id=s, creation_time=t0,
                        hop_count=0, tokens=INITIAL_TOKENS, qos=qos_val,
                        payload_bytes=PhyConst.SENSOR_PAYLOAD_BYTES
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
    
    total_control_energy = control_tx_energy + control_rx_energy
    total_data_wifi_energy = data_wifi_tx_energy + data_wifi_rx_energy
    
    if total_delivered > 0:
        overhead_factor = (uav_relay_events + total_delivered) / float(total_delivered)
    else:
        overhead_factor = 1.0
    
    results = {
        "pdr": 100.0 * total_delivered / max(1, total_generated),
        "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency": float(np.median(latencies)) if latencies else 0.0,
        "avg_hops": float(np.mean(hop_counts)) if hop_counts else 0.0,
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
        "data_bytes": data_bytes_sent
    }
    
    if verbose:
        print(f"  [SPRAY&FOCUS DDS] PDR: {results['pdr']:.1f}% | Latency: {results['avg_latency']:.2f}s | "
              f"Spray: {spray_events} | Focus: {focus_events}")
    
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

