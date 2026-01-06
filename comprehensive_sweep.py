"""
Comprehensive Parameter Sweep Benchmark for Q1 Journal Publication

Experiments:
1. Scalability: Vary NUM_UAVS [4, 6, 8, 10, 12]
2. Traffic Load: Vary NUM_SENSORS [4, 6, 8, 10, 12]
3. Message Size: Vary DATA_PAYLOAD_BYTES [64, 128, 256, 512]
4. Sink Mobility: SINK_MOBILE [True, False]

Each experiment runs all 6 protocols 10 times with 95% CI.
"""

import sys
import csv
import math
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, 'c:/Users/akrem/OneDrive - KFUPM/Desktop/Fastdds_Python')

# ==========================================
# CONFIGURATION
# ==========================================

NUM_RUNS = 10  # Per configuration for 95% CI

# Default baseline configuration
BASELINE = {
    "NUM_UAVS": 8,
    "NUM_SENSORS": 6,
    "DURATION": 1500.0,
    "INITIAL_TOKENS": 10,
    "AREA_SIZE": 500,
    "SINK_MOBILE": True,
    "WIFI_PAYLOAD_BYTES": 256,  # Only affects WiFi (UAV↔UAV, UAV→Sink)
    "GLOBAL_QOS": 1,
    "NUM_SINKS": 1
}

# Parameter variations
PARAM_SWEEPS = {
    "NUM_UAVS": [4, 6, 8, 10, 12],
    "NUM_SENSORS": [4, 6, 8, 10, 12],
    "WIFI_PAYLOAD_BYTES": [64, 128, 256, 512],  # Only WiFi links (ZigBee sensor payload is fixed at 64)
    "SINK_MOBILE": [True, False]
}

METRICS = [
    "pdr",
    "avg_latency",
    "median_latency",
    "energy_per_msg_mJ",
    "avg_hops",
    "total_delivered",
    "spray_events",
    "focus_events",
    "overhead_factor"
]


# ==========================================
# SIMULATION WRAPPERS
# ==========================================

def run_mqtt_baseline(config: dict) -> dict:
    from BASELINE_MQTT import run_baseline_simulation
    return run_baseline_simulation(config, verbose=False)


def run_mqtt_enhanced(config: dict) -> dict:
    """MQTT Enhanced S&F simulation"""
    import NEW_REPLICATION_EDGESRONE as m
    
    m.GLOBAL_QOS = config.get("GLOBAL_QOS", 1)
    m.NUM_UAVS = config.get("NUM_UAVS", 8)
    m.NUM_SENSORS = config.get("NUM_SENSORS", 6)
    m.INITIAL_TOKENS = config.get("INITIAL_TOKENS", 10)
    
    # Set WiFi payload size from config (for payload sweep - sensor payload stays fixed)
    if "WIFI_PAYLOAD_BYTES" in config:
        m.PhyConst.WIFI_DATA_PAYLOAD_BYTES = config["WIFI_PAYLOAD_BYTES"]
    
    AREA_SIZE = config.get("AREA_SIZE", 500)
    duration = config.get("DURATION", 1500.0)
    dt = 0.1
    
    agents = {}
    agents[m.SINK_ID] = m.MqttUAVAgent(m.SINK_ID, [250.0, 250.0, m.PhyConst.H], is_sink=True, area_size=AREA_SIZE)
    for i in range(1, m.NUM_UAVS):
        agents[i] = m.MqttUAVAgent(i, [random.uniform(50, AREA_SIZE-50), 
                                       random.uniform(50, AREA_SIZE-50), m.PhyConst.H], area_size=AREA_SIZE)
    
    iot_nodes = [
        np.array([100 + (i % 7) * 60, 100 + (i // 7) * 60, 10.0])
        for i in range(m.NUM_SENSORS)
    ]
    
    sim_time = 0.0
    SENSOR_RATE = 0.5
    sensor_queues = [[] for _ in range(m.NUM_SENSORS)]
    MSG_COUNTER = 0
    
    total_generated = 0
    total_delivered = 0
    spray_events = 0
    focus_events = 0
    latencies = []
    hop_counts = []
    
    while sim_time < duration:
        sim_time += dt
        
        for s in range(m.NUM_SENSORS):
            if random.random() < SENSOR_RATE * dt:
                MSG_COUNTER += 1
                sensor_queues[s].append((MSG_COUNTER, m.GLOBAL_QOS, sim_time))
                if len(sensor_queues[s]) > 50:
                    sensor_queues[s].pop(0)
                total_generated += 1
        
        for agent in agents.values():
            agent.move(dt)
        
        for i in range(m.NUM_UAVS):
            for j in range(i + 1, m.NUM_UAVS):
                rate, d, _ = m.link_rate(agents[i].pos, agents[j].pos, is_ground_to_uav=False)
                if rate >= m.WIFI.B:
                    agents[i].encounter_timers[j] = 0.0
                    agents[j].encounter_timers[i] = 0.0
                    t_meet = d / 20.0
                    t_i_sink = agents[i].encounter_timers[m.SINK_ID]
                    t_j_sink = agents[j].encounter_timers[m.SINK_ID]
                    if t_j_sink + t_meet < t_i_sink:
                        agents[i].encounter_timers[m.SINK_ID] = t_j_sink + t_meet
                    if t_i_sink + t_meet < t_j_sink:
                        agents[j].encounter_timers[m.SINK_ID] = t_i_sink + t_meet
        
        for i in range(m.NUM_UAVS):
            if i == m.SINK_ID:
                continue
            ai = agents[i]
            if not ai.buffer:
                continue
            
            for j in range(m.NUM_UAVS):
                if j == i or j == m.SINK_ID:
                    continue
                aj = agents[j]
                rate, d, prof = m.link_rate(ai.pos, aj.pos, is_ground_to_uav=False)
                if rate < prof.B:
                    continue
                
                spray_msgs = [msg for msg in ai.buffer if msg.tokens > 1]
                focus_msgs = [msg for msg in ai.buffer if msg.tokens == 1]
                
                max_bytes = (rate * dt) / 8.0
                bytes_sent = 0
                
                for msg in spray_msgs:
                    if msg.msg_id in aj.seen_msgs or len(aj.buffer) >= aj.MAX_BUFFER:
                        continue
                    send_tokens = msg.tokens // 2
                    if send_tokens <= 0:
                        continue
                    
                    frame_bytes = m.mqtt_frame_size(msg.payload_size(), msg.qos)
                    if bytes_sent + frame_bytes > max_bytes:
                        break
                    
                    # Energy tracking - WiFi data transmission
                    bits = frame_bytes * 8
                    tx_e = bits * prof.E_tx_per_bit
                    rx_e = bits * prof.E_rx_per_bit
                    ai.radio_tx_energy += tx_e
                    aj.radio_rx_energy += rx_e
                    
                    # PUBACK energy for QoS 1 data messages
                    if msg.qos == 1:
                        puback_size = m.mqtt_puback_size()
                        puback_bits = puback_size * 8
                        aj.radio_tx_energy += puback_bits * prof.E_tx_per_bit
                        ai.radio_rx_energy += puback_bits * prof.E_rx_per_bit
                    
                    new_msg = m.SprayMessage(
                        msg_id=msg.msg_id, source_id=msg.source_id,
                        creation_time=msg.creation_time, hop_count=msg.hop_count + 1,
                        tokens=send_tokens, payload=msg.payload, qos=msg.qos
                    )
                    aj.buffer.append(new_msg)
                    aj.seen_msgs.add(msg.msg_id)
                    msg.tokens -= send_tokens
                    bytes_sent += frame_bytes
                    spray_events += 1
                
                if focus_msgs:
                    # === FOCUS PHASE: Control message exchange (like DDS S&F) ===
                    # Estimate control exchange overhead before starting
                    ctrl_payload_size = 12  # utility value (8) + overhead (4)
                    ctrl_frame_size = m.mqtt_frame_size(ctrl_payload_size, 1)  # QoS 1
                    puback_size = m.mqtt_puback_size()
                    total_control_bytes = 2 * ctrl_frame_size + 2 * puback_size
                    
                    if bytes_sent + total_control_bytes > max_bytes:
                        continue  # Defer focus exchange to next timestep
                    
                    # STEP 1: UAV_i sends INQUIRY control message
                    ctrl_bits = ctrl_frame_size * 8
                    ctrl_tx_e = ctrl_bits * prof.E_tx_per_bit
                    ctrl_rx_e = ctrl_bits * prof.E_rx_per_bit
                    ai.radio_tx_energy += ctrl_tx_e
                    aj.radio_rx_energy += ctrl_rx_e
                    bytes_sent += ctrl_frame_size
                    
                    # PUBACK for inquiry (QoS 1)
                    puback_bits = puback_size * 8
                    puback_tx_e = puback_bits * prof.E_tx_per_bit
                    puback_rx_e = puback_bits * prof.E_rx_per_bit
                    aj.radio_tx_energy += puback_tx_e
                    ai.radio_rx_energy += puback_rx_e
                    bytes_sent += puback_size
                    
                    # STEP 2: UAV_j sends RESPONSE control message
                    aj.radio_tx_energy += ctrl_tx_e
                    ai.radio_rx_energy += ctrl_rx_e
                    bytes_sent += ctrl_frame_size
                    
                    # PUBACK for response (QoS 1)
                    ai.radio_tx_energy += puback_tx_e
                    aj.radio_rx_energy += puback_rx_e
                    bytes_sent += puback_size
                    
                    # STEP 3: Use utility values for routing decision
                    my_utility = ai.encounter_timers[m.SINK_ID]
                    nb_utility = aj.encounter_timers[m.SINK_ID]
                    t_meet = d / 20.0
                    
                    # STEP 4: Forward data if neighbor is better
                    for msg in focus_msgs:
                        if (nb_utility - t_meet) < my_utility:
                            if msg.msg_id in aj.seen_msgs or len(aj.buffer) >= aj.MAX_BUFFER:
                                continue
                            frame_bytes = m.mqtt_frame_size(msg.payload_size(), msg.qos)
                            if bytes_sent + frame_bytes > max_bytes:
                                break
                            
                            # Energy tracking - WiFi data transmission
                            bits = frame_bytes * 8
                            tx_e = bits * prof.E_tx_per_bit
                            rx_e = bits * prof.E_rx_per_bit
                            ai.radio_tx_energy += tx_e
                            aj.radio_rx_energy += rx_e
                            
                            # PUBACK for data if QoS 1
                            if msg.qos == 1:
                                aj.radio_tx_energy += puback_tx_e
                                ai.radio_rx_energy += puback_rx_e
                            
                            new_msg = m.SprayMessage(
                                msg_id=msg.msg_id, source_id=msg.source_id,
                                creation_time=msg.creation_time, hop_count=msg.hop_count + 1,
                                tokens=1, payload=msg.payload, qos=msg.qos
                            )
                            aj.buffer.append(new_msg)
                            aj.seen_msgs.add(msg.msg_id)
                            if msg in ai.buffer:
                                ai.buffer.remove(msg)
                            bytes_sent += frame_bytes
                            focus_events += 1
        
        sink = agents[m.SINK_ID]
        for i in range(1, m.NUM_UAVS):
            ai = agents[i]
            if not ai.buffer:
                continue
            rate, d, prof = m.link_rate(ai.pos, sink.pos, is_ground_to_uav=False)
            if rate < prof.B:
                continue
            
            max_bytes = (rate * dt) / 8.0
            bytes_sent = 0
            msgs_to_remove = []
            
            for msg in ai.buffer:
                if msg.msg_id in sink.seen_msgs:
                    msgs_to_remove.append(msg)
                    continue
                frame_bytes = m.mqtt_frame_size(msg.payload_size(), msg.qos)
                if bytes_sent + frame_bytes > max_bytes:
                    break
                
                # Energy tracking - WiFi data transmission to sink
                bits = frame_bytes * 8
                tx_e = bits * prof.E_tx_per_bit
                rx_e = bits * prof.E_rx_per_bit
                ai.radio_tx_energy += tx_e
                sink.radio_rx_energy += rx_e
                
                # PUBACK energy for QoS 1 data messages
                if msg.qos == 1:
                    puback_size = m.mqtt_puback_size()
                    puback_bits = puback_size * 8
                    sink.radio_tx_energy += puback_bits * prof.E_tx_per_bit
                    ai.radio_rx_energy += puback_bits * prof.E_rx_per_bit
                
                sink.seen_msgs.add(msg.msg_id)
                total_delivered += 1
                hop_counts.append(msg.hop_count)
                latencies.append(sim_time - msg.creation_time)
                bytes_sent += frame_bytes
                msgs_to_remove.append(msg)
            
            for msg in msgs_to_remove:
                if msg in ai.buffer:
                    ai.buffer.remove(msg)
        
        for s, src_pos in enumerate(iot_nodes):
            if not sensor_queues[s]:
                continue
            best_uav = min(range(1, m.NUM_UAVS), 
                          key=lambda k: np.linalg.norm(agents[k].pos - src_pos))
            rate, d, prof = m.link_rate(src_pos, agents[best_uav].pos, is_ground_to_uav=True)
            if rate < prof.B:
                if sensor_queues[s][0][1] == 0:
                    sensor_queues[s].pop(0)
                continue
            
            msg_id, qos_val, t0 = sensor_queues[s][0]
            if len(agents[best_uav].buffer) < agents[best_uav].MAX_BUFFER:
                sensor_queues[s].pop(0)
                new_msg = m.SprayMessage(
                    msg_id=msg_id, source_id=s, creation_time=t0,
                    hop_count=0, tokens=m.INITIAL_TOKENS, 
                    payload=b"DATA", qos=qos_val
                )
                agents[best_uav].buffer.append(new_msg)
                agents[best_uav].seen_msgs.add(msg_id)
    
    total_uav_radio = sum(a.radio_tx_energy + a.radio_rx_energy for a in agents.values())
    overhead = (spray_events + focus_events + total_delivered) / max(1, total_delivered)
    
    return {
        "pdr": 100.0 * total_delivered / max(1, total_generated),
        "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency": float(np.median(latencies)) if latencies else 0.0,
        "energy_per_msg_mJ": (total_uav_radio / max(1, total_delivered)) * 1000,
        "avg_hops": float(np.mean(hop_counts)) if hop_counts else 0.0,
        "total_delivered": total_delivered,
        "spray_events": spray_events,
        "focus_events": focus_events,
        "overhead_factor": overhead
    }


def run_vanilla_dds(config: dict) -> dict:
    from vanilla_DDS import run_vanilla_dds_simulation
    return run_vanilla_dds_simulation(config, verbose=False)


def run_dds_spray_focus(config: dict) -> dict:
    from spray_focus_DDS import run_spray_focus_dds_simulation
    return run_spray_focus_dds_simulation(config, verbose=False)


def run_dds_multisink(config: dict) -> dict:
    """Multi-sink DDS S&F simulation"""
    from spray_focus_DDS_multisink import run_multisink_simulation
    return run_multisink_simulation(config, verbose=False)


# Standard protocols (1 sink)
PROTOCOLS = [
    {"name": "MQTT_Baseline", "qos": 1, "runner": run_mqtt_baseline, "label": "MQTT Baseline"},
    {"name": "MQTT_SF", "qos": 1, "runner": run_mqtt_enhanced, "label": "MQTT S&F"},
    {"name": "Vanilla_DDS", "qos": 1, "runner": run_vanilla_dds, "label": "Vanilla DDS"},
    {"name": "DDS_SF", "qos": 1, "runner": run_dds_spray_focus, "label": "DDS S&F"}
]

# Multi-sink protocol (used only for NUM_SINKS sweep)
MULTISINK_PROTOCOL = {"name": "DDS_SF_MultiSink", "qos": 1, "runner": run_dds_multisink, "label": "DDS S&F"}


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def ci95(values: List[float]) -> tuple:
    if len(values) == 0:
        return 0.0, 0.0
    mean = np.mean(values)
    if len(values) == 1:
        return mean, 0.0
    std = np.std(values, ddof=1)
    return mean, 1.96 * std / math.sqrt(len(values))


def print_progress(current: int, total: int, label: str):
    bar_len = 25
    filled = int(bar_len * current / total)
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f"\r  [{bar}] {current}/{total} - {label}", end='', flush=True)


# ==========================================
# EXPERIMENT RUNNER
# ==========================================

def run_experiment_set(param_name: str, param_values: list, output_prefix: str):
    """Run experiments varying one parameter"""
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT SET: Varying {param_name}")
    print(f"Values: {param_values}")
    print(f"{'='*70}")
    
    all_results = []
    
    for val in param_values:
        print(f"\n📊 {param_name} = {val}")
        
        for proto in PROTOCOLS:
            config = BASELINE.copy()
            config[param_name] = val
            config["GLOBAL_QOS"] = proto["qos"]
            
            metrics_data = {m: [] for m in METRICS}
            
            for run in range(NUM_RUNS):
                print_progress(run + 1, NUM_RUNS, proto["label"])
                
                seed = run * 1000 + hash(str(val)) % 1000
                random.seed(seed)
                np.random.seed(seed)
                
                try:
                    result = proto["runner"](config)
                    for m in METRICS:
                        v = result.get(m, 0)
                        if v is None:
                            v = 0
                        metrics_data[m].append(v)
                except Exception as e:
                    print(f" ERROR: {e}")
            
            print()
            
            row = {
                "param_name": param_name,
                "param_value": val,
                "protocol": proto["label"]
            }
            
            for m in METRICS:
                mean, ci = ci95(metrics_data[m])
                row[f"{m}_mean"] = round(mean, 4)
                row[f"{m}_ci95"] = round(ci, 4)
            
            all_results.append(row)
            
            print(f"    {proto['label']:15} PDR: {row['pdr_mean']:.1f}±{row['pdr_ci95']:.1f}% | "
                  f"Lat: {row['avg_latency_mean']:.1f}s | Energy: {row['energy_per_msg_mJ_mean']:.1f}mJ")
    
    # Save results
    filename = f"{output_prefix}_{param_name}.csv"
    with open(filename, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    
    print(f"\n📁 Saved: {filename}")
    return filename


def run_comprehensive_benchmark():
    """Run all experiment sets"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"sweep_{timestamp}"
    
    print("=" * 70)
    print("COMPREHENSIVE PARAMETER SWEEP BENCHMARK")
    print("=" * 70)
    print(f"Protocols: {len(PROTOCOLS)}")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Parameter sweeps: {list(PARAM_SWEEPS.keys())}")
    print("=" * 70)
    
    total_experiments = sum(len(v) for v in PARAM_SWEEPS.values()) * len(PROTOCOLS) * NUM_RUNS
    print(f"Total experiments: {total_experiments}")
    print(f"Estimated time: {total_experiments * 30 / 3600:.1f} hours")
    print("=" * 70)
    
    output_files = []
    
    for param_name, param_values in PARAM_SWEEPS.items():
        outfile = run_experiment_set(param_name, param_values, output_prefix)
        output_files.append(outfile)
    
    # Create master summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("Output files:")
    for f in output_files:
        print(f"  - {f}")
    
    return output_files


if __name__ == "__main__":
    run_comprehensive_benchmark()
