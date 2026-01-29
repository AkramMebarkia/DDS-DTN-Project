import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# ==========================================
# GLOBAL CONFIG
# ==========================================

# Choose QoS level for ALL messages: 0 or 1
GLOBAL_QOS = 1          # 0 = fire-and-forget, 1 = at-least-once with retry

# Spray & Focus initial token budget per new DTN message
INITIAL_TOKENS = 8
# Message Time-To-Live (seconds)

# Topology sizes
NUM_UAVS = 6           # UAVs 0..99
SINK_ID = 0              # UAV 0 is the sink/base station
NUM_SENSORS = 10         # Ground IoT nodes

# Sink mobility: moves within 30% of area_size around center (saves energy, UAVs come to it)
SINK_MOBILITY_FRACTION = 0.40

# MQTT overhead approximations (bytes)
MQTT_TOPIC_LEN = 16      # e.g. "edge/data/NN"

# ==========================================
# 1) PHYSICS / RADIO PROFILES
# ==========================================

# ZigBee MTU constraint (IEEE 802.15.4 max frame = 127 bytes)
ZIGBEE_MAX_FRAME = 127
ZIGBEE_MAX_PAYLOAD = 100  # ~27 bytes MAC overhead

class PhyConst:
    # Geometry
    H = 90.0  # UAV altitude (m)

    # Flight/rotor parameters (multirotor energy model)
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
    CONTROL_PAYLOAD_BYTES = 80    # DTN routing control payload


class PHYProfile:
    """Radio profile with Shannon capacity model"""
    def __init__(self, name: str, B: float, P_tx: float, N0: float, E_tx_per_bit: float, E_rx_per_bit: float):
        self.name = name      # "zigbee" or "wifi"
        self.B = B            # Bandwidth (Hz)
        self.P_tx = P_tx      # Tx power (W)
        self.N0 = N0          # Noise power (W)
        self.beta0 = None     # Calibrated path loss factor
        self.E_tx_per_bit = E_tx_per_bit  # J/bit for transmission
        self.E_rx_per_bit = E_rx_per_bit  # J/bit for reception


# ZigBee 802.15.4 (2.4 GHz) - System-level energy from Siekkinen et al. (IEEE WCNC 2012)
# Measured: ~1 µJ/bit end-to-end (includes TX, RX, CSMA overhead)
# Using sender-only charging: E_rx=0 since E_tx includes aggregate system cost
ZIGBEE = PHYProfile("zigbee", B=250_000.0, P_tx=0.0774, N0=1e-13, E_tx_per_bit=1e-6, E_rx_per_bit=0)

# WiFi 802.11 (20 MHz) - System-level energy from Liu & Choi (ACM SIGMETRICS 2023)
# Measured: ~200 nJ/bit for 20 MHz channel (scaled from 80 MHz baseline)
# Using sender-only charging: E_rx=0 since E_tx includes aggregate system cost
WIFI   = PHYProfile("wifi",   B=20_000_000.0, P_tx=1.5, N0=1e-13, E_tx_per_bit=2e-7, E_rx_per_bit=0)

REF_DIST = 100.0  # meters


def calibrate_beta0(profile: PHYProfile):
    """Calibrate path loss so SNR(100m) = 1 (0 dB)"""
    profile.beta0 = profile.N0 * (REF_DIST**2) / profile.P_tx


for _p in (ZIGBEE, WIFI):
    calibrate_beta0(_p)


def shannon_rate_3d(dist_3d: float, profile: PHYProfile) -> float:
    """Shannon rate (bps) with 1/r^2 pathloss"""
    if dist_3d <= 0.0:
        dist_3d = 1e-3
    snr = (profile.beta0 * profile.P_tx) / (profile.N0 * (dist_3d**2))
    return profile.B * math.log2(1.0 + snr)


def link_rate(u_pos: np.ndarray, v_pos: np.ndarray, is_ground_to_uav: bool) -> Tuple[float, float, PHYProfile]:
    """Return (rate bps, distance m, profile) for correct PHY"""
    d = float(np.linalg.norm(u_pos - v_pos))
    prof = ZIGBEE if is_ground_to_uav else WIFI
    return shannon_rate_3d(d, prof), d, prof


# ==========================================
# 2) MQTT OVERHEAD MODEL (WiFi vs ZigBee)
# ==========================================

# --- WiFi (802.11) Transport Layer - for UAV↔UAV and UAV→Sink ---
# MQTT over TCP/IP over WiFi
WIFI_TCP_IP_OVERHEAD = 40     # IP (20) + TCP (20) header
WIFI_L2_OVERHEAD = 30         # WiFi 802.11 MAC/PHY overhead
WIFI_TRANSPORT_OVERHEAD = WIFI_TCP_IP_OVERHEAD + WIFI_L2_OVERHEAD  # = 70 bytes

# --- ZigBee (802.15.4) Transport Layer - for IoT/Sensor→UAV ---
# Sensors use lightweight protocol (no TCP/IP stack)
# IEEE 802.15.4 frame: Frame Control(2) + Seq(1) + Addressing(~8) + FCS(2) = ~15 bytes
ZIGBEE_L2_OVERHEAD = 15       # IEEE 802.15.4 MAC overhead
ZIGBEE_TRANSPORT_OVERHEAD = ZIGBEE_L2_OVERHEAD  # No TCP/IP for sensors

# MQTT header components
MQTT_FIXED_HDR = 2            # Fixed header (type + remaining length)
MQTT_TOPIC_HDR = 2 + MQTT_TOPIC_LEN  # Topic length (2) + topic string
MQTT_PKT_ID = 2               # Packet ID for QoS 1

# Legacy aliases for backward compatibility
TCP_IP_OVERHEAD = WIFI_TCP_IP_OVERHEAD
L2_OVERHEAD = WIFI_L2_OVERHEAD
BASE_WIRE_OVERHEAD = WIFI_TRANSPORT_OVERHEAD


def mqtt_frame_size_wifi(payload_bytes: int, qos: int) -> int:
    """MQTT frame size for WiFi links (UAV↔UAV, UAV→Sink)"""
    mqtt_overhead = MQTT_FIXED_HDR + MQTT_TOPIC_HDR
    if qos == 1:
        mqtt_overhead += MQTT_PKT_ID
    return payload_bytes + mqtt_overhead + WIFI_TRANSPORT_OVERHEAD


def mqtt_frame_size_zigbee(payload_bytes: int, qos: int) -> int:
    """MQTT frame size for ZigBee links (IoT/Sensor→UAV)"""
    mqtt_overhead = MQTT_FIXED_HDR + MQTT_TOPIC_HDR
    if qos == 1:
        mqtt_overhead += MQTT_PKT_ID
    return payload_bytes + mqtt_overhead + ZIGBEE_TRANSPORT_OVERHEAD


def mqtt_puback_size_wifi() -> int:
    """PUBACK frame size for WiFi links"""
    return MQTT_FIXED_HDR + MQTT_PKT_ID + WIFI_TRANSPORT_OVERHEAD
    # = 2 + 2 + 70 = 74 bytes


def mqtt_puback_size_zigbee() -> int:
    """PUBACK frame size for ZigBee links (if sensors use QoS 1)"""
    return MQTT_FIXED_HDR + MQTT_PKT_ID + ZIGBEE_TRANSPORT_OVERHEAD
    # = 2 + 2 + 15 = 19 bytes


# Legacy aliases (default to WiFi for UAV↔UAV links)
def mqtt_frame_size(payload_bytes: int, qos: int) -> int:
    """Legacy function - uses WiFi overhead"""
    return mqtt_frame_size_wifi(payload_bytes, qos)


def mqtt_puback_size() -> int:
    """Legacy function - uses WiFi overhead"""
    return mqtt_puback_size_wifi()


# ==========================================
# 3) DATA STRUCTURES
# ==========================================

@dataclass
class RoutingControlMessage:
    """DTN routing control - utility exchange only"""
    utility_to_sink: float
    
    def payload_size(self) -> int:
        """Control message payload: just the utility value (8 bytes float + 4 bytes overhead)"""
        return 12  # Minimal control payload


@dataclass
class SprayMessage:
    """DTN message in UAV buffer"""
    msg_id: int
    source_id: int
    creation_time: float
    hop_count: int
    tokens: int
    payload: bytes
    qos: int
    payload_bytes: int = PhyConst.WIFI_DATA_PAYLOAD_BYTES  # 64B for sensors, 256B for UAV data
    
    def payload_size(self) -> int:
        """Return configured payload size"""
        return self.payload_bytes


# ==========================================
# 4) MQTT TRANSMISSION SIMULATOR
# ==========================================

class MQTTTransmission:
    """Simulates MQTT message transmission with proper overhead"""
    
    @staticmethod
    def transmit_control(rate_bps: float, profile: PHYProfile, 
                         control_msg: RoutingControlMessage) -> Tuple[bool, float, float, int]:
        """
        Transmit DTN control message via MQTT.
        
        Returns:
            success: True if link is active (rate >= B)
            tx_energy: Transmission energy (Joules)
            rx_energy: Reception energy (Joules)
            frame_bytes: Total bytes on wire
        """
        if rate_bps < profile.B:
            return False, 0.0, 0.0, 0
        
        payload_bytes = control_msg.payload_size()
        # Use correct frame size based on link type (ZigBee vs WiFi)
        if profile.name == "zigbee":
            frame_bytes = mqtt_frame_size_zigbee(payload_bytes, qos=1)
            # ZigBee max frame constraint
            if frame_bytes > ZIGBEE_MAX_FRAME:
                return False, 0.0, 0.0, 0
        else:
            frame_bytes = mqtt_frame_size_wifi(payload_bytes, qos=1)
        bits = frame_bytes * 8
        
        tx_energy = bits * profile.E_tx_per_bit
        rx_energy = bits * profile.E_rx_per_bit
        
        return True, tx_energy, rx_energy, frame_bytes
    
    @staticmethod
    def transmit_data(rate_bps: float, profile: PHYProfile,
                      data_msg: SprayMessage) -> Tuple[bool, float, float, int]:
        """
        Transmit actual data message via MQTT.
        
        Returns:
            success: True if link is active
            tx_energy: Transmission energy (Joules)
            rx_energy: Reception energy (Joules)
            frame_bytes: Total bytes on wire
        """
        if rate_bps < profile.B:
            return False, 0.0, 0.0, 0
        
        payload_bytes = data_msg.payload_size()
        # Use correct frame size based on link type (ZigBee vs WiFi)
        if profile.name == "zigbee":
            frame_bytes = mqtt_frame_size_zigbee(payload_bytes, data_msg.qos)
            # ZigBee max frame constraint
            if frame_bytes > ZIGBEE_MAX_FRAME:
                return False, 0.0, 0.0, 0
        else:
            frame_bytes = mqtt_frame_size_wifi(payload_bytes, data_msg.qos)
        bits = frame_bytes * 8
        
        tx_energy = bits * profile.E_tx_per_bit
        rx_energy = bits * profile.E_rx_per_bit
        
        return True, tx_energy, rx_energy, frame_bytes
    
    @staticmethod
    def transmit_puback(rate_bps: float, profile: PHYProfile) -> Tuple[bool, float, float, int]:
        """
        Transmit PUBACK (QoS 1 acknowledgment).
        
        Returns:
            success: True if link is active
            tx_energy: Transmission energy (Joules)
            rx_energy: Reception energy (Joules)
            frame_bytes: Total bytes on wire
        """
        if rate_bps < profile.B:
            return False, 0.0, 0.0, 0
        
        # Use correct PUBACK size based on link type (ZigBee vs WiFi)
        if profile.name == "zigbee":
            frame_bytes = mqtt_puback_size_zigbee()
        else:
            frame_bytes = mqtt_puback_size_wifi()
        bits = frame_bytes * 8
        
        tx_energy = bits * profile.E_tx_per_bit
        rx_energy = bits * profile.E_rx_per_bit
        
        return True, tx_energy, rx_energy, frame_bytes


# ==========================================
# 5) UAV AGENT
# ==========================================

class MqttUAVAgent:
    def __init__(self, uid: int, start_pos, is_sink: bool = False, area_size: float = 500):
        self.id = uid
        self.pos = np.array(start_pos, dtype=float)
        self.is_sink = is_sink
        self.area_size = area_size
        self.energy = 300000.0  # 300 kJ - ~28 min flight time
        self.radio_tx_energy = 0.0  # J
        self.radio_rx_energy = 0.0  # J
        self.waypoints = []

        # DTN state
        self.buffer: List[SprayMessage] = []
        self.seen_msgs = set()
        self.encounter_timers: Dict[int, float] = {}
        self.MAX_BUFFER = 250

        for i in range(NUM_UAVS):
            self.encounter_timers[i] = 9999.0
        self.encounter_timers[uid] = 0.0

        if self.is_sink:
            self.sink_received_count = 0

    def move(self, dt: float):
        """Update position and encounter timers"""
        # Age encounter timers
        for k in self.encounter_timers.keys():
            self.encounter_timers[k] += dt
        self.encounter_timers[self.id] = 0.0

        # Random waypoint mobility (using configurable area size)
        if not self.waypoints:
            if self.id == SINK_ID:
                # Sink: Limited mobility within 30% radius around center (saves energy)
                center = self.area_size / 2
                radius = self.area_size * SINK_MOBILITY_FRACTION / 2
                self.waypoints = [np.array([
                    center + random.uniform(-radius, radius),
                    center + random.uniform(-radius, radius),
                    PhyConst.H]) for _ in range(5)]
            else:
                # Regular UAVs: Full area coverage for sensor collection
                self.waypoints = [np.array([random.uniform(100, self.area_size-100), 
                                            random.uniform(100, self.area_size-100), 
                                            PhyConst.H]) for _ in range(5)]

        speed = 20.0  # m/s
        target = self.waypoints[0]
        direction = target - self.pos
        dist = np.linalg.norm(direction)
        step = speed * dt

        if dist <= step:
            self.pos = target
            self.waypoints.pop(0)
        else:
            self.pos += (direction / dist) * step

        # Flight energy drain
        p = self.calc_flight_power(speed)
        self.energy -= p * dt

    @staticmethod
    def calc_flight_power(velocity: float) -> float:
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

    def get_utility_message(self) -> RoutingControlMessage:
        """Create routing control message with utility only"""
        return RoutingControlMessage(
            utility_to_sink=self.encounter_timers[SINK_ID]
        )


# ==========================================
# 6) SIMULATION LOOP
# ==========================================

def run_mqtt_simulation():
    print("=" * 70)
    print("FIXED MQTT-DTN UAV SIMULATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - UAVs: {NUM_UAVS} (Sink: UAV {SINK_ID})")
    print(f"  - Sensors: {NUM_SENSORS}")
    print(f"  - QoS Level: {GLOBAL_QOS}")
    print(f"  - Initial Tokens: {INITIAL_TOKENS}")
    print(f"")
    print(f"Architecture:")
    print(f"  - Separate control (80B) and data (256B) messages")
    print(f"  - Control messages: Always QoS 1 (reliable)")
    print(f"  - Data messages: QoS {GLOBAL_QOS}")
    print(f"  - Both TX and RX energy modeled")
    print(f"  - PUBACK energy for QoS 1")
    print(f"")
    print(f"Radio Profiles:")
    print(f"  - ZigBee (Sensor↔UAV): 250 kbps, 1 mW Tx")
    print(f"  - WiFi (UAV↔UAV): 20 Mbps, 100 mW Tx")
    print(f"  - SNR @ 100m: 0 dB (calibrated)")
    print("=" * 70)
    print()

    # ---- Initialize UAVs ----
    agents: Dict[int, MqttUAVAgent] = {}
    agents[SINK_ID] = MqttUAVAgent(SINK_ID, [250.0, 250.0, PhyConst.H], is_sink=True)
    for i in range(1, NUM_UAVS):
        agents[i] = MqttUAVAgent(i, [random.uniform(50, 450), random.uniform(50, 450), PhyConst.H])

    # ---- Initialize Ground IoT nodes with well-spaced random positions ----
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
    
    iot_nodes = generate_spread_sensors(NUM_SENSORS, 500, seed=42)  # Fixed 500m area for standalone run

    # ---- Simulation parameters ----
    sim_time = 0.0
    duration = 1500.0
    dt = 0.1

    # ---- Sensor queues ----
    SENSOR_RATE = 3.0  # msgs/s per sensor
    SENSOR_BUF_MAX = 50
    sensor_queues: List[List[Tuple[int, int, float]]] = [[] for _ in range(NUM_SENSORS)]
    MSG_COUNTER = 0

    # ---- Statistics ----
    total_generated = 0
    total_generated_qos0 = 0
    total_generated_qos1 = 0
    total_delivered = 0
    total_delivered_qos0 = 0
    total_delivered_qos1 = 0
    
    uav_relay_events = 0
    spray_events = 0
    focus_events = 0
    sink_delivery_events = 0
    control_messages_sent = 0
    
    latencies = []
    latencies_qos0 = []
    latencies_qos1 = []
    hop_counts = []
    
    # Energy tracking by message type and PHY
    sensor_tx_energy = 0.0    # ZigBee: Sensor→UAV TX
    sensor_rx_energy = 0.0    # ZigBee: Sensor→UAV RX (PUBACK)
    
    control_tx_energy = 0.0   # WiFi: UAV↔UAV control TX
    control_rx_energy = 0.0   # WiFi: UAV↔UAV control RX
    data_wifi_tx_energy = 0.0 # WiFi: UAV↔UAV and UAV→Sink data TX
    data_wifi_rx_energy = 0.0 # WiFi: UAV↔UAV and UAV→Sink data RX
    
    control_bytes_sent = 0
    data_bytes_sent = 0

    print(f"Starting simulation (duration: {duration}s)...\n")

    try:
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
                    if qos_val == 0:
                        total_generated_qos0 += 1
                    else:
                        total_generated_qos1 += 1

            # ================================================
            # 1) UAV MOVEMENT
            # ================================================
            for agent in agents.values():
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
            # 3) SPRAY & FOCUS ROUTING (UAV ↔ UAV)
            # ================================================
            # 3) Routing: Spray & Focus (UAV<->UAV, WIFI)
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
                    
                    # ========================================
                    # SPRAY PHASE (tokens > 1) - NO CONTROL
                    # Bandwidth-limited per timestep
                    # ========================================
                    max_bytes_this_step = (rate * dt) / 8.0
                    bytes_sent_this_step = 0
                    
                    for msg in spray_messages:
                        if msg.msg_id in aj.seen_msgs:
                            continue
                        if len(aj.buffer) >= aj.MAX_BUFFER:
                            continue
                        
                        send_tokens = msg.tokens // 2
                        if send_tokens <= 0:
                            continue
                        
                        # Check bandwidth capacity
                        frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                        if bytes_sent_this_step + frame_bytes > max_bytes_this_step:
                            break  # No more capacity this timestep
                        
                        # Send DATA directly (no control exchange needed)
                        success, tx_e, rx_e, data_bytes = MQTTTransmission.transmit_data(
                            rate, prof, msg
                        )
                        
                        if not success:
                            continue
                        
                        # Energy accounting (data WiFi)
                        ai.energy -= tx_e
                        ai.radio_tx_energy += tx_e
                        data_wifi_tx_energy += tx_e
                        aj.energy -= rx_e
                        aj.radio_rx_energy += rx_e
                        data_wifi_rx_energy += rx_e
                        
                        data_bytes_sent += data_bytes
                        bytes_sent_this_step += data_bytes
                        
                        # PUBACK if QoS 1
                        if msg.qos == 1:
                            succ_ack, tx_ack, rx_ack, _ = MQTTTransmission.transmit_puback(rate, prof)
                            if succ_ack:
                                aj.energy -= tx_ack
                                aj.radio_tx_energy += tx_ack
                                data_wifi_tx_energy += tx_ack
                                ai.energy -= rx_ack
                                ai.radio_rx_energy += rx_ack
                                data_wifi_rx_energy += rx_ack
                        
                        # Create copy in neighbor
                        new_msg = SprayMessage(
                            msg_id=msg.msg_id,
                            source_id=msg.source_id,
                            creation_time=msg.creation_time,
                            hop_count=msg.hop_count + 1,
                            tokens=send_tokens,
                            payload=msg.payload,
                            qos=msg.qos,
                            payload_bytes=msg.payload_bytes  # CRITICAL: Preserve original payload size
                        )
                        aj.buffer.append(new_msg)
                        aj.seen_msgs.add(msg.msg_id)
                        
                        # Update sender
                        msg.tokens -= send_tokens
                        
                        uav_relay_events += 1
                        spray_events += 1
                    
                    # ========================================
                    # FOCUS PHASE (tokens == 1) - WITH CONTROL
                    # Request-Response pattern (capacity-limited):
                    #   1. UAV_i sends INQUIRY (control msg)
                    #   2. UAV_j sends RESPONSE with utility
                    #   3. UAV_i uses RECEIVED utility for decision
                    # ========================================
                    if focus_messages:
                        # Check capacity for control exchange before starting
                        # Estimate: inquiry + PUBACK + response + PUBACK
                        inquiry_msg = ai.get_utility_message()
                        ctrl_frame_size = mqtt_frame_size(inquiry_msg.payload_size(), 1)
                        puback_size = mqtt_puback_size()
                        total_control_bytes = 2 * ctrl_frame_size + 2 * puback_size
                        
                        if bytes_sent_this_step + total_control_bytes > max_bytes_this_step:
                            continue  # Defer focus exchange to next timestep
                        
                        # STEP 1: UAV_i sends INQUIRY to UAV_j
                        success, tx_e, rx_e, ctrl_bytes = MQTTTransmission.transmit_control(
                            rate, prof, inquiry_msg
                        )
                        
                        if not success:
                            continue
                        
                        # Energy: UAV_i transmits inquiry (control WiFi)
                        ai.energy -= tx_e
                        ai.radio_tx_energy += tx_e
                        control_tx_energy += tx_e
                        aj.energy -= rx_e
                        aj.radio_rx_energy += rx_e
                        control_rx_energy += rx_e
                        
                        control_messages_sent += 1
                        control_bytes_sent += ctrl_bytes
                        bytes_sent_this_step += ctrl_bytes
                        
                        # PUBACK for inquiry (QoS 1)
                        succ_ack, tx_ack, rx_ack, ack_bytes = MQTTTransmission.transmit_puback(rate, prof)
                        if succ_ack:
                            aj.energy -= tx_ack
                            aj.radio_tx_energy += tx_ack
                            control_tx_energy += tx_ack
                            ai.energy -= rx_ack
                            ai.radio_rx_energy += rx_ack
                            control_rx_energy += rx_ack
                            bytes_sent_this_step += ack_bytes
                        
                        # STEP 2: UAV_j sends RESPONSE with its utility
                        response_msg = aj.get_utility_message()
                        success2, tx_e2, rx_e2, ctrl_bytes2 = MQTTTransmission.transmit_control(
                            rate, prof, response_msg
                        )
                        
                        if not success2:
                            continue
                        
                        # Energy: UAV_j transmits response (control WiFi)
                        aj.energy -= tx_e2
                        aj.radio_tx_energy += tx_e2
                        control_tx_energy += tx_e2
                        ai.energy -= rx_e2
                        ai.radio_rx_energy += rx_e2
                        control_rx_energy += rx_e2
                        
                        control_messages_sent += 1
                        control_bytes_sent += ctrl_bytes2
                        bytes_sent_this_step += ctrl_bytes2
                        
                        # PUBACK for response (QoS 1)
                        succ_ack2, tx_ack2, rx_ack2, ack_bytes2 = MQTTTransmission.transmit_puback(rate, prof)
                        if succ_ack2:
                            ai.energy -= tx_ack2  # UAV_i sends PUBACK
                            ai.radio_tx_energy += tx_ack2
                            control_tx_energy += tx_ack2
                            aj.energy -= rx_ack2  # UAV_j receives PUBACK
                            aj.radio_rx_energy += rx_ack2
                            control_rx_energy += rx_ack2
                            bytes_sent_this_step += ack_bytes2
                        
                        # STEP 3: Use RECEIVED utility (from response_msg)
                        my_utility = ai.encounter_timers[SINK_ID]
                        nb_utility = response_msg.utility_to_sink  # USE RECEIVED VALUE
                        t_meet = d / 20.0
                        
                        # STEP 4: Forward data if neighbor is better (bandwidth-limited)
                        for msg in focus_messages:
                            if (nb_utility - t_meet) < my_utility:
                                if msg.msg_id in aj.seen_msgs:
                                    continue
                                if len(aj.buffer) >= aj.MAX_BUFFER:
                                    continue
                                
                                # Check bandwidth capacity (shared with spray phase)
                                frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                                if bytes_sent_this_step + frame_bytes > max_bytes_this_step:
                                    break  # No more capacity this timestep
                                
                                # Send DATA
                                success, tx_e, rx_e, data_bytes = MQTTTransmission.transmit_data(
                                    rate, prof, msg
                                )
                                
                                if not success:
                                    continue
                                
                                # Energy accounting (data WiFi)
                                ai.energy -= tx_e
                                ai.radio_tx_energy += tx_e
                                data_wifi_tx_energy += tx_e
                                aj.energy -= rx_e
                                aj.radio_rx_energy += rx_e
                                data_wifi_rx_energy += rx_e
                                
                                data_bytes_sent += data_bytes
                                bytes_sent_this_step += data_bytes
                                
                                # PUBACK if QoS 1
                                if msg.qos == 1:
                                    succ_ack, tx_ack, rx_ack, _ = MQTTTransmission.transmit_puback(rate, prof)
                                    if succ_ack:
                                        aj.energy -= tx_ack
                                        aj.radio_tx_energy += tx_ack
                                        data_wifi_tx_energy += tx_ack
                                        ai.energy -= rx_ack
                                        ai.radio_rx_energy += rx_ack
                                        data_wifi_rx_energy += rx_ack
                                
                                # Create copy in neighbor
                                new_msg = SprayMessage(
                                    msg_id=msg.msg_id,
                                    source_id=msg.source_id,
                                    creation_time=msg.creation_time,
                                    hop_count=msg.hop_count + 1,
                                    tokens=1,
                                    payload=msg.payload,
                                    qos=msg.qos,
                                    payload_bytes=msg.payload_bytes  # CRITICAL: Preserve original payload size
                                )
                                aj.buffer.append(new_msg)
                                aj.seen_msgs.add(msg.msg_id)
                                
                                # FOCUS: Remove from sender
                                if msg in ai.buffer:
                                    ai.buffer.remove(msg)
                                
                                uav_relay_events += 1
                                focus_events += 1

            # ================================================
            # 4) UAV → SINK DELIVERY
            # ================================================
            sink = agents[SINK_ID]
            
            for i in range(1, NUM_UAVS):
                ai = agents[i]
                if not ai.buffer:
                    continue
                
                rate_bps, d, prof = link_rate(ai.pos, sink.pos, is_ground_to_uav=False)
                
                if rate_bps < prof.B:
                    continue  # Link down
                
                # Capacity for this time step
                max_bytes_this_step = (rate_bps * dt) / 8.0
                bytes_sent = 0
                msgs_to_remove = []
                
                for msg in ai.buffer:
                    if msg.msg_id in sink.seen_msgs:
                        msgs_to_remove.append(msg)
                        continue
                    
                    frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                    
                    if bytes_sent + frame_bytes > max_bytes_this_step:
                        break  # No more capacity
                    
                    success, tx_e, rx_e, _ = MQTTTransmission.transmit_data(rate_bps, prof, msg)
                    
                    if not success:
                        continue
                    
                    # Energy accounting (data WiFi)
                    ai.energy -= tx_e
                    ai.radio_tx_energy += tx_e
                    data_wifi_tx_energy += tx_e
                    sink.energy -= rx_e
                    sink.radio_rx_energy += rx_e
                    data_wifi_rx_energy += rx_e
                    
                    # PUBACK if QoS 1
                    if msg.qos == 1:
                        succ_ack, tx_ack, rx_ack, _ = MQTTTransmission.transmit_puback(rate_bps, prof)
                        if succ_ack:
                            sink.energy -= tx_ack
                            sink.radio_tx_energy += tx_ack
                            data_wifi_tx_energy += tx_ack
                            ai.energy -= rx_ack
                            ai.radio_rx_energy += rx_ack
                            data_wifi_rx_energy += rx_ack
                    
                    # Mark as delivered
                    sink.seen_msgs.add(msg.msg_id)
                    sink.sink_received_count += 1
                    
                    total_delivered += 1
                    sink_delivery_events += 1
                    hop_counts.append(msg.hop_count)
                    
                    lat = sim_time - msg.creation_time
                    latencies.append(lat)
                    
                    if msg.qos == 0:
                        total_delivered_qos0 += 1
                        latencies_qos0.append(lat)
                    else:
                        total_delivered_qos1 += 1
                        latencies_qos1.append(lat)
                    
                    bytes_sent += frame_bytes
                    msgs_to_remove.append(msg)
                    
                    if total_delivered % 50 == 0:
                        print(f"[t={sim_time:.1f}s] Delivered: {total_delivered} | "
                              f"PDR: {100*total_delivered/max(1,total_generated):.1f}%")
                
                # Clean buffer
                for m in msgs_to_remove:
                    if m in ai.buffer:
                        ai.buffer.remove(m)

            # ================================================
            # 4b) GLOBAL DUPLICATE PURGE
            # ================================================
            if total_delivered > 0 and int(sim_time * 10) % 10 == 0:
                delivered_ids = sink.seen_msgs.copy()
                for uid, a in agents.items():
                    if uid == SINK_ID or not a.buffer:
                        continue
                    a.buffer = [m for m in a.buffer if m.msg_id not in delivered_ids]

            # ================================================
            # 5) SENSOR → UAV UPLOAD (bandwidth-limited)
            # ================================================
            for s, src_pos in enumerate(iot_nodes):
                if not sensor_queues[s]:
                    continue
                
                # Find nearest UAV (EXCLUDING sink - sink only receives from UAVs, not sensors)
                best_uav = min(
                    [k for k in range(NUM_UAVS) if k != SINK_ID],
                    key=lambda k: np.linalg.norm(agents[k].pos - src_pos)
                )
                
                rate, d, prof = link_rate(src_pos, agents[best_uav].pos, is_ground_to_uav=True)
                
                # Check if link is up
                if rate < prof.B:
                    # Link down - QoS 0 drops, QoS 1 retries
                    msg_id, qos_val, t0 = sensor_queues[s][0]
                    if qos_val == 0:
                        sensor_queues[s].pop(0)
                    continue
                
                msg_id, qos_val, t0 = sensor_queues[s][0]  # Peek head
                
                # Check bandwidth capacity for this timestep
                max_bytes_this_step = (rate * dt) / 8.0
                frame_bytes = mqtt_frame_size_zigbee(PhyConst.SENSOR_PAYLOAD_BYTES, qos_val)
                
                if frame_bytes > max_bytes_this_step:
                    # Not enough capacity this timestep
                    if qos_val == 0:
                        sensor_queues[s].pop(0)  # QoS 0: drop
                    # QoS 1: retry next timestep
                    continue
                
                # Energy accounting: Sensor TX (ZigBee)
                bits = frame_bytes * 8
                tx_e = bits * prof.E_tx_per_bit
                rx_e = bits * prof.E_rx_per_bit
                sensor_tx_energy += tx_e
                agents[best_uav].energy -= rx_e
                agents[best_uav].radio_rx_energy += rx_e
                
                # PUBACK energy for QoS 1
                if qos_val == 1:
                    puback_bytes = mqtt_puback_size_zigbee()
                    puback_bits = puback_bytes * 8
                    agents[best_uav].energy -= puback_bits * prof.E_tx_per_bit
                    agents[best_uav].radio_tx_energy += puback_bits * prof.E_tx_per_bit
                    sensor_rx_energy += puback_bits * prof.E_rx_per_bit
                
                # UAV buffers sensor data for S&F routing to sink
                # SMART BUFFER POLICY: Prioritize own sensor data over relayed replicas
                if len(agents[best_uav].buffer) < agents[best_uav].MAX_BUFFER:
                    sensor_queues[s].pop(0)  # Pop AFTER successful buffer insertion
                    temp_msg = SprayMessage(
                        msg_id=msg_id,
                        source_id=s,
                        creation_time=t0,
                        hop_count=0,
                        tokens=INITIAL_TOKENS,
                        payload=b"SENSOR_DATA",
                        qos=qos_val,
                        payload_bytes=PhyConst.SENSOR_PAYLOAD_BYTES  # 64B for ZigBee sensors
                    )
                    agents[best_uav].buffer.append(temp_msg)
                    agents[best_uav].seen_msgs.add(msg_id)
                else:
                    # Buffer full - Try to evict a relayed replica (hop_count > 0)
                    evicted = False
                    for victim_idx, victim_msg in enumerate(agents[best_uav].buffer):
                        if victim_msg.hop_count > 0:
                            # Evict this replica to make space for own data
                            agents[best_uav].buffer.pop(victim_idx)
                            evicted = True
                            
                            # Insert new sensor message
                            sensor_queues[s].pop(0)
                            temp_msg = SprayMessage(
                                msg_id=msg_id,
                                source_id=s,
                                creation_time=t0,
                                hop_count=0,
                                tokens=INITIAL_TOKENS,
                                payload=b"SENSOR_DATA",
                                qos=qos_val,
                                payload_bytes=PhyConst.SENSOR_PAYLOAD_BYTES
                            )
                            agents[best_uav].buffer.append(temp_msg)
                            agents[best_uav].seen_msgs.add(msg_id)
                            break
                    
                    if not evicted:
                        # Buffer full of own data - QoS0 drops, QoS1 stays for retry
                        if qos_val == 0:
                            sensor_queues[s].pop(0)

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")

    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 TRAFFIC STATISTICS:")
    print(f"  Configuration: QoS {GLOBAL_QOS}")
    print(f"  Total Generated:          {total_generated}")
    print(f"    ├─ QoS 0:               {total_generated_qos0}")
    print(f"    └─ QoS 1:               {total_generated_qos1}")
    print(f"  Total Delivered:          {total_delivered}")
    print(f"    ├─ QoS 0:               {total_delivered_qos0}")
    print(f"    └─ QoS 1:               {total_delivered_qos1}")
    
    if total_generated > 0:
        pdr = 100.0 * total_delivered / total_generated
        print(f"\n  📈 PDR (Packet Delivery Ratio):")
        print(f"    Overall:                {pdr:.2f}%")
        if total_generated_qos0 > 0:
            pdr0 = 100.0 * total_delivered_qos0 / total_generated_qos0
            print(f"    QoS 0:                  {pdr0:.2f}%")
        if total_generated_qos1 > 0:
            pdr1 = 100.0 * total_delivered_qos1 / total_generated_qos1
            print(f"    QoS 1:                  {pdr1:.2f}%")
    else:
        print(f"\n  PDR: N/A (no traffic generated)")
    
    if latencies:
        print(f"\n  ⏱️  LATENCY (End-to-End):")
        print(f"    Mean:                   {np.mean(latencies):.3f} s")
        print(f"    Median:                 {np.median(latencies):.3f} s")
        print(f"    Std Dev:                {np.std(latencies):.3f} s")
        print(f"    Min:                    {np.min(latencies):.3f} s")
        print(f"    Max:                    {np.max(latencies):.3f} s")
        
        if latencies_qos0:
            print(f"    Mean (QoS 0):           {np.mean(latencies_qos0):.3f} s")
        if latencies_qos1:
            print(f"    Mean (QoS 1):           {np.mean(latencies_qos1):.3f} s")
    
    if hop_counts:
        print(f"\n  🔄 HOP COUNT:")
        print(f"    Mean:                   {np.mean(hop_counts):.2f}")
        print(f"    Median:                 {np.median(hop_counts):.0f}")
        print(f"    Max:                    {np.max(hop_counts)}")
    
    print(f"\n📡 ROUTING OVERHEAD:")
    total_data_transfers = uav_relay_events + sink_delivery_events
    print(f"  UAV↔UAV Relay Events:     {uav_relay_events}")
    print(f"    ├─ Spray events:        {spray_events}")
    print(f"    └─ Focus events:        {focus_events}")
    print(f"  UAV→Sink Deliveries:      {sink_delivery_events}")
    print(f"  Total Data Transfers:     {total_data_transfers}")
    print(f"  Control Messages Sent:    {control_messages_sent}")
    
    if total_delivered > 0:
        overhead_factor = (uav_relay_events + total_delivered) / float(total_delivered)
        print(f"  Message Overhead Factor:  {overhead_factor:.2f}x")
        
        if control_messages_sent > 0:
            ctrl_data_ratio = control_messages_sent / uav_relay_events if uav_relay_events > 0 else 0
            print(f"  Control/Relay Ratio:      {ctrl_data_ratio:.2f}")
    
    print(f"\n📶 BANDWIDTH USAGE:")
    total_bytes = control_bytes_sent + data_bytes_sent
    print(f"  Control Bytes Sent:       {control_bytes_sent:,} B ({control_bytes_sent/1024:.1f} KB)")
    print(f"  Data Bytes Sent:          {data_bytes_sent:,} B ({data_bytes_sent/1024:.1f} KB)")
    print(f"  Total Bytes Sent:         {total_bytes:,} B ({total_bytes/1024:.1f} KB)")
    if total_bytes > 0:
        ctrl_overhead_pct = 100 * control_bytes_sent / total_bytes
        print(f"  Control Overhead:         {ctrl_overhead_pct:.1f}%")
    
    print(f"\n⚡ ENERGY ANALYSIS:")
    
    # UAV energy breakdown
    total_uav_tx = sum(a.radio_tx_energy for a in agents.values())
    total_uav_rx = sum(a.radio_rx_energy for a in agents.values())
    total_uav_radio = total_uav_tx + total_uav_rx
    
    print(f"  UAV Radio Energy (Total):")
    print(f"    TX Energy:              {total_uav_tx:.4f} J")
    print(f"    RX Energy:              {total_uav_rx:.4f} J")
    print(f"    Total Radio:            {total_uav_radio:.4f} J")
    
    # Energy by message type (ACTUAL, not estimated)
    total_control_energy = control_tx_energy + control_rx_energy
    total_data_wifi_energy = data_wifi_tx_energy + data_wifi_rx_energy
    total_sensor_zigbee = sensor_tx_energy + sensor_rx_energy
    
    print(f"\n  Energy by Message Type (Actual):")
    print(f"    Control (WiFi UAV↔UAV):")
    print(f"      TX: {control_tx_energy:.4f} J  |  RX: {control_rx_energy:.4f} J")
    print(f"      Total: {total_control_energy:.4f} J")
    print(f"    Data (WiFi UAV↔UAV/Sink):")
    print(f"      TX: {data_wifi_tx_energy:.4f} J  |  RX: {data_wifi_rx_energy:.4f} J")
    print(f"      Total: {total_data_wifi_energy:.4f} J")
    print(f"    Data (ZigBee Sensor→UAV):")
    print(f"      TX: {sensor_tx_energy:.4f} J  |  RX: {sensor_rx_energy:.4f} J")
    print(f"      Total: {total_sensor_zigbee:.4f} J")
    
    # Percentage breakdown
    grand_total = total_control_energy + total_data_wifi_energy + total_sensor_zigbee
    if grand_total > 0:
        print(f"\n  Energy Distribution:")
        print(f"    Control (WiFi):         {100*total_control_energy/grand_total:.1f}%")
        print(f"    Data (WiFi):            {100*total_data_wifi_energy/grand_total:.1f}%")
        print(f"    Data (ZigBee):          {100*total_sensor_zigbee/grand_total:.1f}%")
    
    # Per-message energy
    if total_delivered > 0:
        avg_energy_per_msg = (total_uav_radio / total_delivered) * 1000
        print(f"\n  Avg Energy per Delivered Msg:")
        print(f"    UAV Radio:              {avg_energy_per_msg:.4f} mJ")
    
    # UAV remaining energy
    print(f"\n  UAV Energy Status (Remaining):")
    min_energy = min(a.energy for a in agents.values())
    max_energy = max(a.energy for a in agents.values())
    avg_energy = np.mean([a.energy for a in agents.values()])
    
    print(f"    Min:                    {min_energy:.2f} J")
    print(f"    Max:                    {max_energy:.2f} J")
    print(f"    Average:                {avg_energy:.2f} J")
    
    # Show individual UAVs with lowest energy
    energy_list = [(uid, a.energy) for uid, a in agents.items()]
    energy_list.sort(key=lambda x: x[1])
    print(f"    Lowest 5 UAVs:")
    for uid, energy in energy_list[:5]:
        role = " (SINK)" if uid == SINK_ID else ""
        print(f"      UAV {uid:3d}{role}: {energy:8.2f} J")
    
    print(f"\n" + "=" * 70)
    print("✅ Simulation Complete")
    print("=" * 70)
    
    # Summary for journal paper
    print(f"\n📄 KEY METRICS SUMMARY (for paper):")
    if total_generated > 0:
        print(f"  • PDR: {100*total_delivered/total_generated:.2f}%")
    if latencies:
        print(f"  • Avg Latency: {np.mean(latencies):.2f}s")
    if hop_counts:
        print(f"  • Avg Hops: {np.mean(hop_counts):.2f}")
    if total_delivered > 0:
        print(f"  • Overhead Factor: {(uav_relay_events + total_delivered) / total_delivered:.2f}x")
        print(f"  • Energy/Msg: {(total_uav_radio/total_delivered)*1000:.2f} mJ")
    print(f"  • Control Messages: {control_messages_sent}")
    print(f"  • UAV↔UAV Relays: {uav_relay_events}")
    print(f"  • UAV→Sink Deliveries: {sink_delivery_events}")
    print()


def run_simulation(config: dict, verbose: bool = False) -> dict:
    """
    Run simulation with the given configuration and return results.
    
    Args:
        config: Dictionary with keys:
            - NUM_UAVS: Number of mobile UAVs
            - NUM_SINKS: Number of sink nodes (default 1)
            - SINK_MOBILE: Whether sinks are mobile (default False)
            - NUM_SENSORS: Number of sensors
            - INITIAL_TOKENS: Initial token budget
            - GLOBAL_QOS: QoS level (0 or 1)
            - DATA_PAYLOAD_BYTES: Payload size in bytes
            - DURATION: Simulation duration (optional, default 560)
        verbose: If True, print progress
    
    Returns:
        Dictionary with all metrics
    """
    # Apply configuration to globals
    global NUM_UAVS, NUM_SENSORS, INITIAL_TOKENS, GLOBAL_QOS, SINK_ID
    
    NUM_UAVS = config.get("NUM_UAVS", 8)
    NUM_SINKS = config.get("NUM_SINKS", 1)
    SINK_MOBILE = config.get("SINK_MOBILE", True)  # Mobile by default for fair comparison
    NUM_SENSORS = config.get("NUM_SENSORS", 6)
    INITIAL_TOKENS = config.get("INITIAL_TOKENS", 10)
    GLOBAL_QOS = config.get("GLOBAL_QOS", 1)
    
    # Area size
    AREA_SIZE = config.get("AREA_SIZE", 750)  # Default to 750m (matching sweep baseline)
    
    # Apply buffer size if present
    if "MAX_BUFFER" in config:
        MqttUAVAgent.MAX_BUFFER = config["MAX_BUFFER"]
    
    # Apply payload size
    if "WIFI_PAYLOAD_BYTES" in config:
        PhyConst.WIFI_DATA_PAYLOAD_BYTES = config["WIFI_PAYLOAD_BYTES"]
    
    duration = config.get("DURATION", 1500.0)  # Match sweep duration
    dt = 0.1
    
    # ---- Single-sink mode (default, matching sweep behavior) ----
    # SINK_ID = 0, Mobile UAVs = 1 to NUM_UAVS-1
    SINK_ID = 0
    SINK_IDS = [SINK_ID]
    
    agents: Dict[int, MqttUAVAgent] = {}
    
    # Initialize sink (UAV 0)
    if SINK_MOBILE:
        sink_pos = [random.uniform(100, AREA_SIZE-100), random.uniform(100, AREA_SIZE-100), PhyConst.H]
        agents[SINK_ID] = MqttUAVAgent(SINK_ID, sink_pos, is_sink=False, area_size=AREA_SIZE)
    else:
        sink_pos = [AREA_SIZE/2, AREA_SIZE/2, PhyConst.H]
        agents[SINK_ID] = MqttUAVAgent(SINK_ID, sink_pos, is_sink=True, area_size=AREA_SIZE)
    
    # Initialize mobile UAVs (1 to NUM_UAVS-1)
    for i in range(1, NUM_UAVS):
        agents[i] = MqttUAVAgent(i, [random.uniform(50, AREA_SIZE-50), 
                                     random.uniform(50, AREA_SIZE-50), PhyConst.H], area_size=AREA_SIZE)
    
    # ---- Initialize Ground IoT nodes (well-spaced positions) ----
    def generate_spread_sensors(num_sensors, area_size, seed=42):
        """Generate well-spaced sensor positions."""
        rng = np.random.RandomState(seed)
        margin = 100
        min_dist = (area_size - 2 * margin) / (num_sensors ** 0.5 + 1)
        positions = []
        for _ in range(num_sensors):
            for attempt in range(1000):
                x = rng.uniform(margin, area_size - margin)
                y = rng.uniform(margin, area_size - margin)
                valid = True
                for px, py, _ in positions:
                    if np.sqrt((x - px)**2 + (y - py)**2) < min_dist:
                        valid = False
                        break
                if valid or attempt == 999:
                    positions.append([x, y, 10.0])
                    break
        return [np.array(pos) for pos in positions]
    
    iot_nodes = generate_spread_sensors(NUM_SENSORS, AREA_SIZE, seed=42)

    # ---- Simulation parameters ----
    sim_time = 0.0

    # ---- Sensor queues ----
    SENSOR_RATE = 3.0  # Must match all other simulation files (DDS, vanilla, baseline)
    SENSOR_BUF_MAX = 50
    sensor_queues: List[List[Tuple[int, int, float]]] = [[] for _ in range(NUM_SENSORS)]
    MSG_COUNTER = 0

    # ---- Statistics ----
    total_generated = 0
    total_delivered = 0
    
    uav_relay_events = 0
    spray_events = 0
    focus_events = 0
    sink_delivery_events = 0
    control_messages_sent = 0
    
    latencies = []
    hop_counts = []
    
    # Direct vs Relayed delivery tracking
    direct_deliveries = 0    # IoT → Sink directly (hop_count=0)
    relayed_deliveries = 0   # Via UAV relay (hop_count≥1)
    
    # Energy tracking
    sensor_tx_energy = 0.0
    sensor_rx_energy = 0.0
    control_tx_energy = 0.0
    control_rx_energy = 0.0
    data_wifi_tx_energy = 0.0
    data_wifi_rx_energy = 0.0
    
    control_bytes_sent = 0
    data_bytes_sent = 0

    # ---- Simulation loop (simplified for batch runs) ----
    while sim_time < duration:
        sim_time += dt

        # 0) SENSOR DATA GENERATION
        for s in range(NUM_SENSORS):
            if random.random() < SENSOR_RATE * dt:
                MSG_COUNTER += 1
                qos_val = GLOBAL_QOS
                sensor_queues[s].append((MSG_COUNTER, qos_val, sim_time))
                if len(sensor_queues[s]) > SENSOR_BUF_MAX:
                    sensor_queues[s].pop(0)
                total_generated += 1

        # 1) UAV MOVEMENT (skip static sinks)
        for uid, agent in agents.items():
            if uid in SINK_IDS and not SINK_MOBILE:
                continue  # Static sinks don't move
            agent.move(dt)

        # 2) ENCOUNTER DETECTION & DV UPDATES (with multiple sinks)
        # Track encounters between all mobile UAVs
        for i in range(NUM_UAVS):
            for j in range(i + 1, NUM_UAVS):
                rate, d, _prof = link_rate(agents[i].pos, agents[j].pos, is_ground_to_uav=False)
                if rate >= WIFI.B:
                    agents[i].encounter_timers[j] = 0.0
                    agents[j].encounter_timers[i] = 0.0
                    t_meet = d / 20.0
                    # Use minimum utility to any sink
                    t_i_sink = min(agents[i].encounter_timers.get(sid, 9999.0) for sid in SINK_IDS)
                    t_j_sink = min(agents[j].encounter_timers.get(sid, 9999.0) for sid in SINK_IDS)
                    # Update closest sink utility
                    for sid in SINK_IDS:
                        if t_j_sink + t_meet < agents[i].encounter_timers.get(sid, 9999.0):
                            agents[i].encounter_timers[sid] = t_j_sink + t_meet
                        if t_i_sink + t_meet < agents[j].encounter_timers.get(sid, 9999.0):
                            agents[j].encounter_timers[sid] = t_i_sink + t_meet
        
        # Track encounters with sinks
        for i in range(NUM_UAVS):
            for sid in SINK_IDS:
                rate, d, _prof = link_rate(agents[i].pos, agents[sid].pos, is_ground_to_uav=False)
                if rate >= WIFI.B:
                    agents[i].encounter_timers[sid] = 0.0





        # 3) SPRAY & FOCUS ROUTING (between mobile UAVs only - run BEFORE sink delivery)
        for i in range(NUM_UAVS):
            if i in SINK_IDS:
                continue  # Sink doesn't spray
            ai = agents[i]
            if not ai.buffer:
                continue
            
            for j in range(NUM_UAVS):
                if j == i or j in SINK_IDS:
                    continue  # Don't spray to self or sink (sink handled in Step 4)
                if j == i:
                    continue
                aj = agents[j]
                rate, d, prof = link_rate(ai.pos, aj.pos, is_ground_to_uav=False)
                if rate < prof.B:
                    continue

                max_bytes_this_step = (rate * dt) / 8.0
                bytes_sent_this_step = 0

                spray_messages = [m for m in ai.buffer if m.tokens > 1]
                focus_messages = [m for m in ai.buffer if m.tokens == 1]

                # SPRAY PHASE
                for msg in spray_messages:
                    if msg.msg_id in aj.seen_msgs or len(aj.buffer) >= aj.MAX_BUFFER:
                        continue
                    send_tokens = msg.tokens // 2
                    if send_tokens <= 0:
                        continue
                    frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                    if bytes_sent_this_step + frame_bytes > max_bytes_this_step:
                        break
                    
                    success, tx_e, rx_e, data_bytes = MQTTTransmission.transmit_data(rate, prof, msg)
                    if not success:
                        continue
                    
                    ai.energy -= tx_e
                    ai.radio_tx_energy += tx_e
                    data_wifi_tx_energy += tx_e
                    aj.energy -= rx_e
                    aj.radio_rx_energy += rx_e
                    data_wifi_rx_energy += rx_e
                    data_bytes_sent += data_bytes
                    bytes_sent_this_step += data_bytes

                    if msg.qos == 1:
                        succ_ack, tx_ack, rx_ack, ack_bytes = MQTTTransmission.transmit_puback(rate, prof)
                        if succ_ack:
                            aj.energy -= tx_ack
                            aj.radio_tx_energy += tx_ack
                            data_wifi_tx_energy += tx_ack
                            ai.energy -= rx_ack
                            ai.radio_rx_energy += rx_ack
                            data_wifi_rx_energy += rx_ack

                    new_msg = SprayMessage(
                        msg_id=msg.msg_id, source_id=msg.source_id,
                        creation_time=msg.creation_time, hop_count=msg.hop_count + 1,
                        tokens=send_tokens, payload=msg.payload, qos=msg.qos,
                        payload_bytes=msg.payload_bytes  # CRITICAL: Preserve original payload size
                    )
                    aj.buffer.append(new_msg)
                    aj.seen_msgs.add(msg.msg_id)
                    msg.tokens -= send_tokens
                    uav_relay_events += 1
                    spray_events += 1

                # FOCUS PHASE
                if focus_messages:
                    inquiry_msg = ai.get_utility_message()
                    ctrl_frame_size = mqtt_frame_size(inquiry_msg.payload_size(), 1)
                    puback_size = mqtt_puback_size()
                    total_control_bytes = 2 * ctrl_frame_size + 2 * puback_size
                    
                    if bytes_sent_this_step + total_control_bytes > max_bytes_this_step:
                        continue
                    
                    success, tx_e, rx_e, ctrl_bytes = MQTTTransmission.transmit_control(rate, prof, inquiry_msg)
                    if not success:
                        continue
                    
                    ai.energy -= tx_e
                    ai.radio_tx_energy += tx_e
                    control_tx_energy += tx_e
                    aj.energy -= rx_e
                    aj.radio_rx_energy += rx_e
                    control_rx_energy += rx_e
                    control_messages_sent += 1
                    control_bytes_sent += ctrl_bytes
                    bytes_sent_this_step += ctrl_bytes

                    succ_ack, tx_ack, rx_ack, ack_bytes = MQTTTransmission.transmit_puback(rate, prof)
                    if succ_ack:
                        aj.energy -= tx_ack
                        aj.radio_tx_energy += tx_ack
                        control_tx_energy += tx_ack
                        ai.energy -= rx_ack
                        ai.radio_rx_energy += rx_ack
                        control_rx_energy += rx_ack
                        bytes_sent_this_step += ack_bytes

                    response_msg = aj.get_utility_message()
                    success2, tx_e2, rx_e2, ctrl_bytes2 = MQTTTransmission.transmit_control(rate, prof, response_msg)
                    if not success2:
                        continue
                    
                    aj.energy -= tx_e2
                    aj.radio_tx_energy += tx_e2
                    control_tx_energy += tx_e2
                    ai.energy -= rx_e2
                    ai.radio_rx_energy += rx_e2
                    control_rx_energy += rx_e2
                    control_messages_sent += 1
                    control_bytes_sent += ctrl_bytes2
                    bytes_sent_this_step += ctrl_bytes2

                    succ_ack2, tx_ack2, rx_ack2, ack_bytes2 = MQTTTransmission.transmit_puback(rate, prof)
                    if succ_ack2:
                        ai.energy -= tx_ack2
                        ai.radio_tx_energy += tx_ack2
                        control_tx_energy += tx_ack2
                        aj.energy -= rx_ack2
                        aj.radio_rx_energy += rx_ack2
                        control_rx_energy += rx_ack2
                        bytes_sent_this_step += ack_bytes2

                    # Use minimum utility to any sink
                    my_utility = min(ai.encounter_timers.get(sid, 9999.0) for sid in SINK_IDS)
                    nb_utility = response_msg.utility_to_sink
                    t_meet = d / 20.0

                    for msg in focus_messages:
                        if (nb_utility - t_meet) < my_utility:
                            if msg.msg_id in aj.seen_msgs or len(aj.buffer) >= aj.MAX_BUFFER:
                                continue
                            frame_bytes = mqtt_frame_size(msg.payload_size(), msg.qos)
                            if bytes_sent_this_step + frame_bytes > max_bytes_this_step:
                                break
                            
                            success, tx_e, rx_e, data_bytes = MQTTTransmission.transmit_data(rate, prof, msg)
                            if not success:
                                continue
                            
                            ai.energy -= tx_e
                            ai.radio_tx_energy += tx_e
                            data_wifi_tx_energy += tx_e
                            aj.energy -= rx_e
                            aj.radio_rx_energy += rx_e
                            data_wifi_rx_energy += rx_e
                            data_bytes_sent += data_bytes
                            bytes_sent_this_step += data_bytes

                            if msg.qos == 1:
                                succ_ack, tx_ack, rx_ack, _ = MQTTTransmission.transmit_puback(rate, prof)
                                if succ_ack:
                                    aj.energy -= tx_ack
                                    aj.radio_tx_energy += tx_ack
                                    data_wifi_tx_energy += tx_ack
                                    ai.energy -= rx_ack
                                    ai.radio_rx_energy += rx_ack
                                    data_wifi_rx_energy += rx_ack

                            new_msg = SprayMessage(
                                msg_id=msg.msg_id, source_id=msg.source_id,
                                creation_time=msg.creation_time, hop_count=msg.hop_count + 1,
                                tokens=1, payload=msg.payload, qos=msg.qos,
                                payload_bytes=msg.payload_bytes  # CRITICAL: Preserve original payload size
                            )
                            aj.buffer.append(new_msg)
                            aj.seen_msgs.add(msg.msg_id)
                            if msg in ai.buffer:
                                ai.buffer.remove(msg)
                            uav_relay_events += 1
                            focus_events += 1

        # 4) UAV → SINK DELIVERY (run AFTER S&F routing)
        # Collect all delivered message IDs across all sinks for purging
        all_delivered_ids = set()
        for sid in SINK_IDS:
            all_delivered_ids.update(agents[sid].seen_msgs)
        
        for i in range(NUM_UAVS):
            if i in SINK_IDS:
                continue
            ai = agents[i]
            if not ai.buffer:
                continue
            
            # Try each sink
            for sid in SINK_IDS:
                sink = agents[sid]
                rate_bps, d, prof = link_rate(ai.pos, sink.pos, is_ground_to_uav=False)
                if rate_bps < prof.B:
                    continue
                
                max_bytes_this_step = (rate_bps * dt) / 8.0
                bytes_sent = 0
                msgs_to_remove = []

                for msg in ai.buffer:
                    # Check if already delivered to ANY sink
                    if msg.msg_id in all_delivered_ids:
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
                    all_delivered_ids.add(msg.msg_id)  # Update for other sinks
                    total_delivered += 1
                    sink_delivery_events += 1
                    hop_counts.append(msg.hop_count)
                    latencies.append(sim_time - msg.creation_time)
                    bytes_sent += frame_bytes
                    msgs_to_remove.append(msg)
                    # Track direct vs relayed
                    if msg.hop_count > 0:
                        relayed_deliveries += 1
                    else:
                        direct_deliveries += 1

                for m in msgs_to_remove:
                    if m in ai.buffer:
                        ai.buffer.remove(m)

        # 4b) DUPLICATE PURGE (using all sinks' delivered messages)
        if total_delivered > 0 and int(sim_time * 10) % 10 == 0:
            for uid, a in agents.items():
                if uid in SINK_IDS or not a.buffer:
                    continue
                a.buffer = [m for m in a.buffer if m.msg_id not in all_delivered_ids]

        # 5) SENSOR → UAV UPLOAD (EXCLUDING SINK - sink only receives from UAVs)
        for s, src_pos in enumerate(iot_nodes):
            if not sensor_queues[s]:
                continue
            # Exclude sink from sensor collection
            best_uav = min([k for k in range(NUM_UAVS) if k not in SINK_IDS], 
                          key=lambda k: np.linalg.norm(agents[k].pos - src_pos))
            rate, d, prof = link_rate(src_pos, agents[best_uav].pos, is_ground_to_uav=True)
            
            if rate < prof.B:
                msg_id, qos_val, t0 = sensor_queues[s][0]
                if qos_val == 0:
                    sensor_queues[s].pop(0)
                continue

            msg_id, qos_val, t0 = sensor_queues[s][0]
            temp_msg = SprayMessage(
                msg_id=msg_id, source_id=s, creation_time=t0,
                hop_count=0, tokens=INITIAL_TOKENS, payload=b"SENSOR_DATA", qos=qos_val,
                payload_bytes=PhyConst.SENSOR_PAYLOAD_BYTES  # 64B for ZigBee sensors
            )
            
            max_bytes_this_step = (rate * dt) / 8.0
            frame_bytes = mqtt_frame_size_zigbee(temp_msg.payload_size(), temp_msg.qos)
            if frame_bytes > max_bytes_this_step:
                if qos_val == 0:
                    sensor_queues[s].pop(0)
                continue

            success, tx_e, rx_e, _ = MQTTTransmission.transmit_data(rate, prof, temp_msg)
            if success:
                # Energy update
                sensor_tx_energy += tx_e
                agents[best_uav].energy -= rx_e
                agents[best_uav].radio_rx_energy += rx_e
                
                if qos_val == 1:
                    succ_ack, tx_ack, rx_ack, _ = MQTTTransmission.transmit_puback(rate, prof)
                    if succ_ack:
                        agents[best_uav].energy -= tx_ack
                        agents[best_uav].radio_tx_energy += tx_ack
                        sensor_rx_energy += rx_ack

                sensor_queues[s].pop(0)
                
                # UAV buffers sensor data for S&F routing to sink
                if len(agents[best_uav].buffer) < agents[best_uav].MAX_BUFFER:
                    agents[best_uav].buffer.append(temp_msg)
                    agents[best_uav].seen_msgs.add(msg_id)
            else:
                if qos_val == 0:
                    sensor_queues[s].pop(0)

    # ---- Compute results ----
    total_uav_tx = sum(a.radio_tx_energy for a in agents.values())
    total_uav_rx = sum(a.radio_rx_energy for a in agents.values())
    total_uav_radio = total_uav_tx + total_uav_rx
    
    total_control_energy = control_tx_energy + control_rx_energy
    total_data_wifi_energy = data_wifi_tx_energy + data_wifi_rx_energy
    total_data_zigbee = sensor_tx_energy + sensor_rx_energy

    # Compute avg_hops_relayed (only messages with hop_count > 0)
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
        "overhead_factor": (uav_relay_events + total_delivered) / max(1, total_delivered),
        "total_generated": total_generated,
        "total_delivered": total_delivered,
        "uav_relay_events": uav_relay_events,
        "spray_events": spray_events,
        "focus_events": focus_events,
        "sink_delivery_events": sink_delivery_events,
        "control_messages_sent": control_messages_sent,
        "control_energy": total_control_energy,
        "data_wifi_energy": total_data_wifi_energy,
        "data_zigbee_energy": total_data_zigbee,
        "total_uav_radio_energy": total_uav_radio,
        "energy_per_msg_mJ": (total_uav_radio / max(1, total_delivered)) * 1000
    }
    
    if verbose:
        print(f"  PDR: {results['pdr']:.1f}% | Latency: {results['avg_latency']:.2f}s | Hops: {results['avg_hops']:.2f}")
    
    return results


if __name__ == "__main__":
    run_mqtt_simulation()
