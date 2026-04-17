# UAV DTN Benchmark (DDS vs MQTT)

This repository contains a simulation benchmark comparing DDS and MQTT in a UAV-based Delay Tolerant Network (DTN). It models scenarios where UAVs act as data mules to collect messages from ground sensors and deliver them to a sink node.

## Evaluated Protocols
The benchmark runs 8 variations across QoS 0 (Best Effort) and QoS 1 (Reliable):
- Baseline MQTT
- MQTT with Spray & Focus routing
- Baseline DDS
- DDS with Spray & Focus routing

## Benchmark Sweeps & Metrics
The main benchmark evaluates scalability and traffic load by sweeping:
- Number of UAVs (4 to 12)
- Number of Sensors (4 to 64)
- Area Size (500m to 1500m)

Outputs are generated as CSV files containing 95% confidence intervals for:
- Packet Delivery Ratio (PDR)
- Average and median latency
- Energy per message (mJ) and total system energy (kJ)
- Routing metrics (hop counts, delivery types, spray/focus events)

## Usage

Run the complete parameter sweep:
```bash
python main.py
