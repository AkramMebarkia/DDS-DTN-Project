"""
Generates publication-quality figures with:
- Default matplotlib font (DejaVu Sans)
- Separate files for each plot
- Consistent colors across all figures
- No arrows or highlight annotations
- Proper handling of overlapping lines
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path


# Use CSVs with batching ENABLED from Final Results folder
DATA_PREFIX = "sweep_20260131_161559"
DATA_DIR = Path(r"c:\Users\akrem\OneDrive - KFUPM\Desktop\Final Results\CSVs")
OUTPUT_DIR = Path("elsevier_plots_recent_ones_after_adjustments_submission")  # New output folder
OUTPUT_DIR.mkdir(exist_ok=True)

# Use default matplotlib font (DejaVu Sans) - explicitly reset any serif settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
    'axes.grid': False,
})

# ============================================================================
# CONSISTENT COLOR PALETTE - Same color per protocol family
# ============================================================================

# Tab10 colors for consistency
COLORS = {
    'mqtt_baseline': '#1f77b4',  # Blue
    'mqtt_sf':       '#ff7f0e',  # Orange
    'dds_vanilla':   '#2ca02c',  # Green
    'dds_sf':        '#d62728',  # Red
}

# Protocol styling with consistent colors per family
PROTOCOL_STYLES = {
    'MQTT Baseline QoS0': {'color': COLORS['mqtt_baseline'], 'marker': 'o', 'linestyle': '--', 'hatch': ''},
    'MQTT Baseline QoS1': {'color': COLORS['mqtt_baseline'], 'marker': 's', 'linestyle': '-',  'hatch': '//'},
    'MQTT S&F QoS0':      {'color': COLORS['mqtt_sf'], 'marker': '^', 'linestyle': '--', 'hatch': ''},
    'MQTT S&F QoS1':      {'color': COLORS['mqtt_sf'], 'marker': 'v', 'linestyle': '-',  'hatch': '\\\\'},
    'Vanilla DDS BE':     {'color': COLORS['dds_vanilla'], 'marker': 'D', 'linestyle': '--', 'hatch': ''},
    'Vanilla DDS Rel':    {'color': COLORS['dds_vanilla'], 'marker': 'p', 'linestyle': '-',  'hatch': 'xx'},
    'DDS S&F BE':         {'color': COLORS['dds_sf'], 'marker': 'h', 'linestyle': '--', 'hatch': ''},
    'DDS S&F Rel':        {'color': COLORS['dds_sf'], 'marker': '*', 'linestyle': '-',  'hatch': '..'},
}

PROTOCOL_ORDER = [
    'MQTT Baseline QoS0', 'MQTT Baseline QoS1',
    'MQTT S&F QoS0', 'MQTT S&F QoS1',
    'Vanilla DDS BE', 'Vanilla DDS Rel',
    'DDS S&F BE', 'DDS S&F Rel'
]

# Short labels for legend
SHORT_LABELS = {
    'MQTT Baseline QoS0': 'MQTT-BL QoS0',
    'MQTT Baseline QoS1': 'MQTT-BL QoS1',
    'MQTT S&F QoS0': 'MQTT-SF QoS0',
    'MQTT S&F QoS1': 'MQTT-SF QoS1',
    'Vanilla DDS BE': 'DDS-BL BE',
    'Vanilla DDS Rel': 'DDS-BL Rel',
    'DDS S&F BE': 'DDS-SF BE',
    'DDS S&F Rel': 'DDS-SF Rel',
}

# ============================================================================
# REUSABLE PLOTTING FUNCTIONS
# ============================================================================

def plot_line_chart(df, x_col, y_col, ci_col, xlabel, ylabel, title, filename, 
                    protocols=None, add_zoom_inset=False, zoom_protocols=None,
                    xlim_start=None, ylim=None):
    """
    Create a single line chart with markers and error bars.
    Style matches reference image 2 (clean lines with markers).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if protocols is None:
        protocols = [p for p in PROTOCOL_ORDER if p in df['protocol'].unique()]
    
    for i, proto in enumerate(protocols):
        subset = df[df['protocol'] == proto].sort_values(x_col)
        if subset.empty:
            continue
            
        style = PROTOCOL_STYLES[proto]
        x = subset[x_col].values
        y = subset[y_col].values
        yerr = subset[ci_col].values if ci_col in subset.columns else None
        
        # Slight offset for overlapping lines (MQTT S&F vs DDS S&F)
        if 'S&F' in proto:
            offset = 0.02 * (i - 4) if i >= 4 else 0
        else:
            offset = 0
        
        ax.errorbar(
            x, y * (1 + offset), yerr=yerr,
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            label=SHORT_LABELS[proto],
            capsize=3,
            capthick=0.8,
            elinewidth=0.8,
            markeredgecolor='white',
            markeredgewidth=0.5,
            markersize=7,
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim_start is not None:
        ax.set_xlim(xlim_start, None)  # Start x-axis from specified value
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])  # Set y-axis bounds
    ax.legend(loc='best', framealpha=0.9, edgecolor='none', ncol=2)
    
    # Add zoom inset for overlapping S&F lines if requested
    if add_zoom_inset and zoom_protocols:
        _add_zoom_inset(ax, df, x_col, y_col, ci_col, zoom_protocols)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf')
    fig.savefig(OUTPUT_DIR / f'{filename}.png')
    print(f"Saved: {filename}")
    plt.close(fig)


def _add_zoom_inset(ax, df, x_col, y_col, ci_col, protocols):
    """Add zoomed inset for overlapping lines."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    # Get data range for zoom
    sf_data = df[df['protocol'].isin(protocols)]
    if sf_data.empty:
        return
    
    x_all = sf_data[x_col].unique()
    mid_idx = len(x_all) // 2
    x_zoom = x_all[max(0, mid_idx-1):min(len(x_all), mid_idx+2)]
    
    zoom_df = sf_data[sf_data[x_col].isin(x_zoom)]
    y_min = zoom_df[y_col].min() * 0.95
    y_max = zoom_df[y_col].max() * 1.05
    
    # Create inset
    axins = inset_axes(ax, width="35%", height="35%", loc='lower right',
                       bbox_to_anchor=(0.02, 0.02, 1, 1), bbox_transform=ax.transAxes)
    
    for proto in protocols:
        subset = df[df['protocol'] == proto].sort_values(x_col)
        if subset.empty:
            continue
        style = PROTOCOL_STYLES[proto]
        x = subset[x_col].values
        y = subset[y_col].values
        axins.plot(x, y, color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], markersize=5, linewidth=1.2)
    
    axins.set_xlim(x_zoom[0] * 0.98, x_zoom[-1] * 1.02)
    axins.set_ylim(y_min, y_max)
    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.3)


def plot_bar_chart(df, y_col, ci_col, ylabel, title, filename, protocols=None, 
                   filter_col=None, filter_val=None):
    """
    Create a grouped bar chart with error bars.
    MODIFIED: 
    1. All bars have the SAME color (Standard Research Blue).
    2. No patterns/hatching.
    3. Full box borders kept.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Filter data if needed
    plot_df = df.copy()
    if filter_col and filter_val is not None:
        plot_df = plot_df[plot_df[filter_col] == filter_val]
    
    if protocols is None:
        protocols = [p for p in PROTOCOL_ORDER if p in plot_df['protocol'].unique()]
    
    x = np.arange(len(protocols))
    width = 0.65
    
    values = []
    errors = []
    
    for proto in protocols:
        row = plot_df[plot_df['protocol'] == proto]
        if not row.empty:
            values.append(row[y_col].values[0])
            errors.append(row[ci_col].values[0] if ci_col in row.columns else 0)
        else:
            values.append(0)
            errors.append(0)
    
    # Plot bars: UNIFORM Color, Black edges, NO hatching
    # Using #1f77b4 (Standard muted blue) for a professional look
    ax.bar(x, values, width, yerr=errors, capsize=4,
           color='#1f77b4', edgecolor='black', linewidth=1.0, zorder=3)
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    
    # Improved Label Alignment
    ax.set_xticklabels([SHORT_LABELS[p] for p in protocols], 
                       rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
    
    # Clean background but KEEP borders
    ax.grid(False)
    
    # Ensure all spines (borders) are visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    # Ensure bottom is 0 unless data dictates otherwise
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf')
    fig.savefig(OUTPUT_DIR / f'{filename}.png')
    print(f"Saved: {filename}")
    plt.close(fig)


def plot_energy_efficiency_bars(df, filename='energy_efficiency'):
    """
    Energy efficiency bar chart like reference image 3.
    Compare MQTT S&F QoS1 vs DDS S&F Rel across sensor counts.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    protocols = ['MQTT S&F QoS1', 'DDS S&F Rel']
    plot_df = df[df['protocol'].isin(protocols)]
    
    sensor_values = sorted(plot_df['param_value'].unique())
    x = np.arange(len(sensor_values))
    width = 0.35
    
    for i, proto in enumerate(protocols):
        subset = plot_df[plot_df['protocol'] == proto].sort_values('param_value')
        values = subset['energy_per_msg_mJ_mean'].values
        errors = subset['energy_per_msg_mJ_ci95'].values
        
        offset = -width/2 + i * width
        bars = ax.bar(x + offset, values, width, yerr=errors, capsize=3,
                      label=SHORT_LABELS[proto],
                      color=PROTOCOL_STYLES[proto]['color'],
                      edgecolor='black', linewidth=0.8,
                      hatch=PROTOCOL_STYLES[proto]['hatch'])
    
    ax.set_xlabel('Number of Sensors, $S$')
    ax.set_ylabel('Energy per Message, $E_{msg}$ (mJ)')
    ax.set_title('Energy Efficiency Under Traffic Load')
    ax.set_xticks(x)
    ax.set_xticklabels(sensor_values)
    ax.legend(loc='best', framealpha=0.9, edgecolor='none')
    # Clean background - no grid lines
    ax.grid(False)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf')
    fig.savefig(OUTPUT_DIR / f'{filename}.png')
    print(f"Saved: {filename}")
    plt.close(fig)



def plot_routing_overhead_stacked(df, filename='routing_overhead_stacked'):
    """
    Stacked bar chart showing routing overhead breakdown like reference image 4.
    Shows Spray Phase, Focus Phase, and Sink Delivery for S&F protocols.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter to S&F Reliable protocols only and baseline condition (8 sensors)
    sf_protocols = ['MQTT S&F QoS1', 'DDS S&F Rel']
    plot_df = df[(df['protocol'].isin(sf_protocols)) & (df['param_value'] == 8)]
    
    if plot_df.empty:
        print("No S&F data for routing overhead plot")
        return
    
    protocols = [p for p in sf_protocols if p in plot_df['protocol'].unique()]
    x = np.arange(len(protocols))
    width = 0.6
    
    # Component colors
    spray_color = COLORS['mqtt_baseline']  # Blue
    focus_color = COLORS['mqtt_sf']         # Orange
    sink_color = COLORS['dds_vanilla']      # Green
    
    spray_vals = []
    focus_vals = []
    sink_vals = []
    
    for proto in protocols:
        row = plot_df[plot_df['protocol'] == proto]
        if not row.empty:
            spray = row['spray_events_mean'].values[0]
            focus = row['focus_events_mean'].values[0]
            delivered = row['total_delivered_mean'].values[0]
            spray_vals.append(spray)
            focus_vals.append(focus)
            sink_vals.append(delivered)
        else:
            spray_vals.append(0)
            focus_vals.append(0)
            sink_vals.append(0)
    
    # Stacked bars with hatching
    bars1 = ax.bar(x, spray_vals, width, label='Spray Phase', 
                   color=spray_color, edgecolor='black', linewidth=0.8, hatch='//')
    bars2 = ax.bar(x, focus_vals, width, bottom=spray_vals, label='Focus Phase',
                   color=focus_color, edgecolor='black', linewidth=0.8, hatch='\\\\')
    bars3 = ax.bar(x, sink_vals, width, 
                   bottom=np.array(spray_vals) + np.array(focus_vals),
                   label='Sink Delivery', color=sink_color, edgecolor='black', 
                   linewidth=0.8, hatch='')
    
    ax.set_xlabel('DTN Protocol')
    ax.set_ylabel('Message Events')
    ax.set_title('Routing Overhead Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[p] for p in protocols], rotation=15, ha='right')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,
              framealpha=0.9, edgecolor='none')
    # Clean background - no grid lines
    ax.grid(False)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf')
    fig.savefig(OUTPUT_DIR / f'{filename}.png')
    print(f"Saved: {filename}")
    plt.close(fig)


# ============================================================================
# GENERATE ALL FIGURES
# ============================================================================

def main():
    print("=" * 60)
    print("Q1 Journal Publication Figure Generator")
    print("Computer Networks Elsevier - Wireless Ad Hoc Networks")
    print("=" * 60)
    print()
    
    # Load data
    files = {
        'AREA_SIZE': DATA_DIR / f"{DATA_PREFIX}_AREA_SIZE.csv",
        'NUM_UAVS': DATA_DIR / f"{DATA_PREFIX}_NUM_UAVS.csv",
        'NUM_SENSORS': DATA_DIR / f"{DATA_PREFIX}_NUM_SENSORS.csv",
        'SINK_MOBILE': DATA_DIR / f"{DATA_PREFIX}_SINK_MOBILE.csv",
        'WIFI_PAYLOAD': DATA_DIR / f"{DATA_PREFIX}_WIFI_PAYLOAD_BYTES.csv",
    }
    
    data = {}
    for name, path in files.items():
        if path.exists():
            data[name] = pd.read_csv(path)
            print(f"Loaded {name}: {len(data[name])} rows")
        else:
            print(f"Missing: {path}")
    print()
    
    # ========================================
    # AREA SIZE PLOTS (separate files)
    # ========================================
    if 'AREA_SIZE' in data:
        print("\n--- Area Size Sweep ---")
        df = data['AREA_SIZE']
        
        plot_line_chart(df, 'param_value', 'pdr_mean', 'pdr_ci95',
                        'Coverage Area, $D$ (m)', 'Packet Delivery Ratio, PDR (%)',
                        'PDR vs Network Area', 'area_pdr')
        
        plot_line_chart(df, 'param_value', 'avg_latency_mean', 'avg_latency_ci95',
                        'Coverage Area, $D$ (m)', 'Average Latency, $\ell$ (s)',
                        'Latency vs Network Area', 'area_latency')
        
        plot_line_chart(df, 'param_value', 'energy_per_msg_mJ_mean', 'energy_per_msg_mJ_ci95',
                        'Coverage Area, $D$ (m)', 'Energy per Message, $E_{msg}$ (mJ)',
                        'Energy vs Network Area', 'area_energy')
    
    # ========================================
    # NUM UAVS PLOTS (separate files)
    # ========================================
    if 'NUM_UAVS' in data:
        print("\n--- UAV Count Sweep ---")
        df = data['NUM_UAVS']
        
        plot_line_chart(df, 'param_value', 'pdr_mean', 'pdr_ci95',
                        'Number of UAVs, $U$', 'Packet Delivery Ratio, PDR (%)',
                        'PDR vs UAV Count', 'uavs_pdr')
        
        # Filter to start from 6 UAVs for latency plot
        df_filtered = df[df['param_value'] >= 6]
        plot_line_chart(df_filtered, 'param_value', 'avg_latency_mean', 'avg_latency_ci95',
                        'Number of UAVs, $U$', 'Average Latency, $\ell$ (s)',
                        'Latency vs UAV Count', 'uavs_latency')
        
        plot_line_chart(df, 'param_value', 'energy_per_msg_mJ_mean', 'energy_per_msg_mJ_ci95',
                        'Number of UAVs, $U$', 'Energy per Message, $E_{msg}$ (mJ)',
                        'Energy Efficiency vs UAV Count', 'uavs_energy')
    
    # ========================================
    # NUM SENSORS PLOTS (separate files)  
    # ========================================
    if 'NUM_SENSORS' in data:
        print("\n--- Sensor Count Sweep ---")
        df = data['NUM_SENSORS']
        
        # Filter to start from 8 sensors (exclude 4 sensor data point)
        df_filtered = df[df['param_value'] >= 8]
        
        plot_line_chart(df_filtered, 'param_value', 'pdr_mean', 'pdr_ci95',
                        'Number of Sensors, $S$', 'Packet Delivery Ratio, PDR (%)',
                        'PDR vs Sensor Count', 'sensors_pdr')
        
        plot_line_chart(df, 'param_value', 'avg_latency_mean', 'avg_latency_ci95',
                        'Number of Sensors, $S$', 'Average Latency, $\ell$ (s)',
                        'Latency vs Sensor Count', 'sensors_latency')
        
        plot_line_chart(df, 'param_value', 'energy_per_msg_mJ_mean', 'energy_per_msg_mJ_ci95',
                        'Number of Sensors, $S$', 'Energy per Message, $E_{msg}$ (mJ)',
                        'Energy vs Sensor Count', 'sensors_energy')
        
        # Energy efficiency comparison (like ref image 3)
        plot_energy_efficiency_bars(df, 'energy_efficiency')
        
        # Routing overhead stacked (like ref image 4)
        plot_routing_overhead_stacked(df, 'routing_overhead_stacked')
    
    # ========================================
    # PAYLOAD SIZE PLOTS (separate files)
    # ========================================
    if 'WIFI_PAYLOAD' in data:
        print("\n--- Payload Size Sweep ---")
        df = data['WIFI_PAYLOAD']
        
        plot_line_chart(df, 'param_value', 'pdr_mean', 'pdr_ci95',
                        'Payload Size (bytes)', 'Packet Delivery Ratio (%)',
                        'PDR vs Payload Size', 'payload_pdr')
        
        plot_line_chart(df, 'param_value', 'avg_latency_mean', 'avg_latency_ci95',
                        'Payload Size (bytes)', 'Average Latency (s)',
                        'Latency vs Payload Size', 'payload_latency')
        
        plot_line_chart(df, 'param_value', 'energy_per_msg_mJ_mean', 'energy_per_msg_mJ_ci95',
                        'Payload Size (bytes)', 'Energy per Message (mJ)',
                        'Energy vs Payload Size', 'payload_energy')
    
    # ========================================
    # SINK MOBILITY COMPARISON
    # ========================================
    if 'SINK_MOBILE' in data:
        print("\n--- Sink Mobility Comparison ---")
        df = data['SINK_MOBILE']
        
        # Create grouped bar chart for mobile vs static sink
        fig, ax = plt.subplots(figsize=(8, 5))
        
        protocols = [p for p in PROTOCOL_ORDER if p in df['protocol'].unique()]
        x = np.arange(len(protocols))
        width = 0.35
        
        # Get data for static (False) and mobile (True) sinks
        static_df = df[df['param_value'] == False]
        mobile_df = df[df['param_value'] == True]
        
        static_vals = [static_df[static_df['protocol'] == p]['pdr_mean'].values[0] if not static_df[static_df['protocol'] == p].empty else 0 for p in protocols]
        mobile_vals = [mobile_df[mobile_df['protocol'] == p]['pdr_mean'].values[0] if not mobile_df[mobile_df['protocol'] == p].empty else 0 for p in protocols]
        
        static_errs = [static_df[static_df['protocol'] == p]['pdr_ci95'].values[0] if not static_df[static_df['protocol'] == p].empty else 0 for p in protocols]
        mobile_errs = [mobile_df[mobile_df['protocol'] == p]['pdr_ci95'].values[0] if not mobile_df[mobile_df['protocol'] == p].empty else 0 for p in protocols]
        
        bars1 = ax.bar(x - width/2, static_vals, width, yerr=static_errs, capsize=3,
                       label='Static Sink', color='#2ca02c', edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x + width/2, mobile_vals, width, yerr=mobile_errs, capsize=3,
                       label='Mobile Sink', color='#ff7f0e', edgecolor='black', linewidth=0.8, hatch='//')
        
        ax.set_ylabel('Packet Delivery Ratio (%)')
        ax.set_title('Impact of Sink Mobility on PDR')
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[p] for p in protocols], rotation=45, ha='right', fontsize=9)
        ax.legend(loc='best', framealpha=0.9, edgecolor='none')
        ax.grid(False)
        
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / 'sink_mobility_pdr.pdf')
        fig.savefig(OUTPUT_DIR / 'sink_mobility_pdr.png')
        print("Saved: sink_mobility_pdr")
        plt.close(fig)
        
        # Energy comparison for sink mobility
        fig, ax = plt.subplots(figsize=(8, 5))
        
        static_energy = [static_df[static_df['protocol'] == p]['energy_per_msg_mJ_mean'].values[0] if not static_df[static_df['protocol'] == p].empty else 0 for p in protocols]
        mobile_energy = [mobile_df[mobile_df['protocol'] == p]['energy_per_msg_mJ_mean'].values[0] if not mobile_df[mobile_df['protocol'] == p].empty else 0 for p in protocols]
        
        static_energy_err = [static_df[static_df['protocol'] == p]['energy_per_msg_mJ_ci95'].values[0] if not static_df[static_df['protocol'] == p].empty else 0 for p in protocols]
        mobile_energy_err = [mobile_df[mobile_df['protocol'] == p]['energy_per_msg_mJ_ci95'].values[0] if not mobile_df[mobile_df['protocol'] == p].empty else 0 for p in protocols]
        
        bars1 = ax.bar(x - width/2, static_energy, width, yerr=static_energy_err, capsize=3,
                       label='Static Sink', color='#2ca02c', edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x + width/2, mobile_energy, width, yerr=mobile_energy_err, capsize=3,
                       label='Mobile Sink', color='#ff7f0e', edgecolor='black', linewidth=0.8, hatch='//')
        
        ax.set_ylabel('Energy per Message (mJ)')
        ax.set_title('Impact of Sink Mobility on Energy Consumption')
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[p] for p in protocols], rotation=45, ha='right', fontsize=9)
        ax.legend(loc='best', framealpha=0.9, edgecolor='none')
        ax.grid(False)
        
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / 'sink_mobility_energy.pdf')
        fig.savefig(OUTPUT_DIR / 'sink_mobility_energy.png')
        print("✓ Saved: sink_mobility_energy")
        plt.close(fig)
    
    # ========================================
    # BASELINE BAR CHARTS @ 6 UAVs (matching sensor sweep config)
    # ========================================
    if 'NUM_UAVS' in data:
        print("\n--- Baseline Comparison @ 6 UAVs ---")
        df = data['NUM_UAVS']
        
        # Filter to 6 UAVs (matching baseline config used in sensor sweep)
        baseline_df = df[df['param_value'] == 6]
        
        if not baseline_df.empty:
            plot_bar_chart(baseline_df, 'pdr_mean', 'pdr_ci95',
                          'Packet Delivery Ratio, PDR (%)', 
                          'Protocol Comparison - Packet Delivery Ratio',
                          'baseline_pdr')
            
            plot_bar_chart(baseline_df, 'avg_latency_mean', 'avg_latency_ci95',
                          'Average Latency, $\ell$ (s)',
                          'Protocol Comparison - Average Latency',
                          'baseline_latency')
            
            plot_bar_chart(baseline_df, 'energy_per_msg_mJ_mean', 'energy_per_msg_mJ_ci95',
                          'Energy per Message, $E_{msg}$ (mJ)',
                          'Protocol Comparison - Energy Efficiency',
                          'baseline_energy')
        else:
            print("No data for 10 UAVs baseline")
    
    print()
    print("=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR.absolute()}")
    print("Formats: PDF (vector) + PNG (raster @ 300 DPI)")
    print("=" * 60)


if __name__ == "__main__":
    main()
    
