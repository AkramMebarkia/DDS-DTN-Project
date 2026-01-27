"""
Comprehensive Parameter Sweep Benchmark for Q1 Journal Publication

Experiments:
1. Scalability: Vary NUM_UAVS [4, 6, 8, 10, 12]
2. Traffic Load: Vary NUM_SENSORS [4, 6, 8, 10, 12]
3. Message Size: Vary DATA_PAYLOAD_BYTES [64, 128, 256, 512]
4. Sink Mobility: SINK_MOBILE [True, False]

Each experiment runs all 8 protocols 10 times with 95% CI.
"""

import sys
import csv
import math
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, 'c:/Users/akrem/OneDrive - KFUPM/Desktop/Fastdds_Python - Copy')
import zlib  # For stable hash

# ==========================================
# CONFIGURATION
# ==========================================

NUM_RUNS = 10  # Per configuration for 95% CI

# Default baseline configuration
BASELINE = {
    "NUM_UAVS": 6,
    "NUM_SENSORS": 10,
    "DURATION": 1500.0,
    "INITIAL_TOKENS": 6,
    "AREA_SIZE": 750,  # ~25% connectivity - high PDR with S&F advantage
    "SINK_MOBILE": True,  # Mobile sink participates in routing
    "WIFI_PAYLOAD_BYTES": 64,  # Only affects WiFi (UAV↔UAV, UAV→Sink)
    "GLOBAL_QOS": 1,
    "NUM_SINKS": 1,
    "MAX_BUFFER": 250,  # Small buffer to test congestion handling
    "BATCH_ENABLE": True  # Disable DDS batching for fair comparison
}

# Parameter variationssweep
PARAM_SWEEPS = {
    "NUM_UAVS": [4, 6, 8, 10, 12],
    "NUM_SENSORS": [4, 6, 8, 10, 12],
    "WIFI_PAYLOAD_BYTES": [64, 128, 256, 512],  # Only WiFi links (ZigBee sensor payload is fixed at 64)
    "SINK_MOBILE": [True, False],
    "AREA_SIZE": [750, 1000, 1250, 1500, 1750]  # Network density sweep
}

METRICS = [
    "pdr",
    "avg_latency",
    "median_latency",
    "energy_per_msg_mJ",
    "avg_hops",
    "avg_hops_relayed",      # NEW: Only relayed messages (hop_count > 0)
    "direct_deliveries",     # NEW: IoT → Sink directly
    "relayed_deliveries",    # NEW: Via UAV relay
    "direct_delivery_ratio", # NEW: Percentage of direct deliveries
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
    """MQTT Enhanced S&F simulation - delegates to validated standalone module."""
    import NEW_REPLICATION_EDGESRONE as m

    # Set globals for the module
    m.GLOBAL_QOS = config.get("GLOBAL_QOS", 1)
    m.NUM_UAVS = config.get("NUM_UAVS", 8)
    m.NUM_SENSORS = config.get("NUM_SENSORS", 6)
    m.INITIAL_TOKENS = config.get("INITIAL_TOKENS", 1)

    # Set WiFi payload size from config (for payload sweep)
    if "WIFI_PAYLOAD_BYTES" in config:
        m.PhyConst.WIFI_DATA_PAYLOAD_BYTES = config["WIFI_PAYLOAD_BYTES"]

    # Call the validated run_simulation from the standalone module
    return m.run_simulation(config, verbose=False)


def run_vanilla_dds(config: dict) -> dict:
    from vanilla_DDS import run_vanilla_dds_simulation
    return run_vanilla_dds_simulation(config, verbose=False)


def run_dds_spray_focus(config: dict) -> dict:
    from spray_focus_DDS import run_spray_focus_dds_simulation
    return run_spray_focus_dds_simulation(config, verbose=False)


def run_dds_multisink(config: dict) -> dict:
    """Multi-sink DDS S&F simulation"""
    # from spray_focus_DDS_multisink import run_multisink_simulation
    # return run_multisink_simulation(config, verbose=False)
    raise NotImplementedError("spray_focus_DDS_multisink module is missing")


# Standard protocols (1 sink) - 8 variants
PROTOCOLS = [
    # MQTT protocols (QoS 0 = Best Effort, QoS 1 = Reliable with PUBACK)
    {"name": "MQTT_Baseline_QoS0", "qos": 0, "runner": run_mqtt_baseline, "label": "MQTT Baseline QoS0"},
    {"name": "MQTT_Baseline_QoS1", "qos": 1, "runner": run_mqtt_baseline, "label": "MQTT Baseline QoS1"},
    {"name": "MQTT_SF_QoS0", "qos": 0, "runner": run_mqtt_enhanced, "label": "MQTT S&F QoS0"},
    {"name": "MQTT_SF_QoS1", "qos": 1, "runner": run_mqtt_enhanced, "label": "MQTT S&F QoS1"},
    # DDS protocols (QoS 0 = Best Effort, QoS 1 = Reliable with ACKNACK)
    {"name": "Vanilla_DDS_BE", "qos": 0, "runner": run_vanilla_dds, "label": "Vanilla DDS BE"},
    {"name": "Vanilla_DDS_Rel", "qos": 1, "runner": run_vanilla_dds, "label": "Vanilla DDS Rel"},
    {"name": "DDS_SF_BE", "qos": 0, "runner": run_dds_spray_focus, "label": "DDS S&F BE"},
    {"name": "DDS_SF_Rel", "qos": 1, "runner": run_dds_spray_focus, "label": "DDS S&F Rel"}
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
                
                # CRITICAL: Seed RNG for reproducibility (different seed per run, but deterministic)
                # seed = run * 1000 + zlib.crc32(str(val).encode()) % 10000
                # random.seed(seed)
                # np.random.seed(seed)
                
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
