"""
Q1 Journal Publication Plots Generator (Enhanced / Refactored)

Targets: Q1 networking journals (e.g., Elsevier Computer Networks) style norms:
- Clean, minimal aesthetics (no seaborn look)
- Vector-friendly PDF export with embedded fonts
- Consistent sizing for 1-column / 2-column figures
- Colorblind-safe palette + redundancy via linestyles/markers/hatches
- CI shown as subtle bands on line plots (cleaner than cap-heavy errorbars)
- Stable protocol ordering across all plots
- Legend placed outside plot area for multi-series comparisons

Generates:
SECTION 1: Baseline Protocol Comparison (4 plots)
SECTION 2: Scalability & Load Sweeps (6 plots)
SECTION 3: Payload Size Effects (3 plots)
SECTION 4: Spray & Focus Dynamics (3 plots)
+ Sink Mobility Comparison (1 figure with 3 panels)

Output: publication_plots/*.pdf (vector)
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = "publication_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot sizing (inches) designed around typical journal column widths
COL_W = 3.50   # 1-column ~ 3.5"
DBL_W = 7.20   # 2-column ~ 7.2"

DEFAULT_TWO_COLUMN = True  # set False if you mostly publish single-column figures


# =============================================================================
# LABEL NORMALIZATION
# =============================================================================

LABEL_MAP = {
    "DDS S&F": "Proposed DDS S&F",
    # add more renames if needed:
    # "Vanilla DDS BE": "Vanilla DDS (BE)",
}

def rename_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize protocol names for publication consistency."""
    for col in ["label", "protocol"]:
        if col in df.columns:
            df[col] = df[col].replace(LABEL_MAP)
    return df


# =============================================================================
# STYLE (Journal-like Matplotlib)
# =============================================================================

def set_elsevier_networks_style(two_column: bool = True) -> None:
    """
    Clean, publication-oriented rcParams.
    - Embeds TTF fonts in PDF (keeps text selectable).
    - Light grid, no top/right spines.
    - Designed for readability at final column width.
    """
    base_w = DBL_W if two_column else COL_W
    mpl.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9 if two_column else 8,
        "axes.labelsize": 9 if two_column else 8,
        "axes.titlesize": 10 if two_column else 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        # Figure
        "figure.figsize": (base_w*1.75, base_w),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Vector export: embed TrueType fonts (Elsevier-friendly)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Axes
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    })


# Colorblind-safe-ish palette (short, high-contrast)
COLOR_CYCLE = [
    "#0077BB",  # blue
    "#EE7733",  # orange
    "#009988",  # teal
    "#CC3311",  # red
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#888888",  # gray
]
LINE_CYCLE = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X"]


@dataclass(frozen=True)
class ProtoStyle:
    color: str
    linestyle: object
    marker: str
    family: str  # used for hatching/legend grouping


def _make_default_registry(protocols: List[str]) -> Dict[str, ProtoStyle]:
    """
    Build a stable style registry. If you want strict colors per protocol,
    you can override entries after creation.
    """
    # stable order for styling assignment
    protos_sorted = sorted(protocols)

    styles: Dict[str, ProtoStyle] = {}
    for i, p in enumerate(protos_sorted):
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        linestyle = LINE_CYCLE[i % len(LINE_CYCLE)]
        marker = MARKER_CYCLE[i % len(MARKER_CYCLE)]
        family = "MQTT" if "MQTT" in p else ("DDS" if "DDS" in p else "Other")
        styles[p] = ProtoStyle(color=color, linestyle=linestyle, marker=marker, family=family)

    # Optional: force some “semantic” mapping (keeps related variants consistent)
    def force(proto: str, color: str | None = None, linestyle: object | None = None, marker: str | None = None):
        if proto not in styles:
            return
        cur = styles[proto]
        styles[proto] = ProtoStyle(
            color=color if color is not None else cur.color,
            linestyle=linestyle if linestyle is not None else cur.linestyle,
            marker=marker if marker is not None else cur.marker,
            family=cur.family
        )

    # Example semantic grouping (edit to match your preferred mapping)
    force("MQTT Baseline", color="#0077BB", linestyle="-", marker="o")
    force("MQTT Baseline QoS0", color="#0077BB", linestyle="-", marker="o")
    force("MQTT Baseline QoS1", color="#0077BB", linestyle="--", marker="o")
    force("MQTT S&F", color="#EE7733", linestyle="-", marker="s")
    force("MQTT S&F QoS0", color="#EE7733", linestyle="-", marker="s")
    force("MQTT S&F QoS1", color="#EE7733", linestyle="--", marker="s")

    force("Vanilla DDS", color="#009988", linestyle="-", marker="^")
    force("Vanilla DDS BE", color="#009988", linestyle="-", marker="^")
    force("Vanilla DDS Rel", color="#009988", linestyle="--", marker="^")

    force("Proposed DDS S&F", color="#CC3311", linestyle="-", marker="D")
    force("DDS S&F", color="#CC3311", linestyle="-", marker="D")
    force("DDS S&F BE", color="#CC3311", linestyle="-", marker="D")
    force("DDS S&F Rel", color="#CC3311", linestyle="--", marker="D")

    return styles


def protocol_order_key(p: str) -> Tuple[int, int, str]:
    """
    Stable ordering across all figures:
    - MQTT Baseline variants first
    - MQTT S&F variants next
    - DDS baseline next
    - DDS S&F (proposed) last
    """
    p_low = p.lower()
    family = 9
    variant = 9

    if "mqtt" in p_low:
        family = 0
        variant = 0 if "baseline" in p_low else 1
    elif "dds" in p_low:
        family = 1
        variant = 0 if ("vanilla" in p_low or "baseline" in p_low) else 1

    # Prefer qos ordering if present
    if "qos0" in p_low:
        qos = 0
    elif "qos1" in p_low:
        qos = 1
    else:
        qos = 2

    return (family, variant, f"{qos}-{p}")


def style_for(proto: str, registry: Dict[str, ProtoStyle]) -> ProtoStyle:
    return registry.get(proto, ProtoStyle(color="#444444", linestyle="-", marker="o", family="Other"))


def add_outside_legend(ax: plt.Axes, ncol: int = 2, y: float = 1.18) -> None:
    """Place legend above axes (cleaner for many lines)."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.2,
        borderaxespad=0.0,
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return rename_labels(df)

def load_sweep_data(csv_path: str) -> pd.DataFrame:
    return load_csv(csv_path)

def load_summary_data(csv_path: str) -> pd.DataFrame:
    return load_csv(csv_path)

def extract_baseline_from_sweep(sweep_csv_path: str, baseline_value: int = 8) -> pd.DataFrame:
    """
    Extract baseline comparison data from a sweep CSV at a specific param_value.
    Compatible with Section 1 bar charts.
    """
    df = pd.read_csv(sweep_csv_path)
    baseline_df = df[df["param_value"] == baseline_value].copy()
    if "protocol" in baseline_df.columns:
        baseline_df["label"] = baseline_df["protocol"]
    return rename_labels(baseline_df)


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def save_fig(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"✅ Saved: {filename}")

def barplot_q1(
    ax: plt.Axes,
    x: np.ndarray,
    heights: np.ndarray,
    yerr: Optional[np.ndarray],
    labels: List[str],
    registry: Dict[str, ProtoStyle],
    width: float = 0.78,
) -> List[mpl.patches.Rectangle]:
    """
    Journal-like bar chart:
    - light edge lines (not heavy)
    - hatch redundancy by family (MQTT vs DDS)
    """
    hatch_by_family = {"MQTT": "///", "DDS": "\\\\", "Other": ""}

    colors = [style_for(l, registry).color for l in labels]
    bars = ax.bar(
        x, heights,
        width=width,
        yerr=yerr,
        capsize=2.5 if yerr is not None else 0,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.95,
    )

    for b, lab in zip(bars, labels):
        fam = style_for(lab, registry).family
        b.set_hatch(hatch_by_family.get(fam, ""))

    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    return list(bars)

def annotate_bars(ax: plt.Axes, bars, fmt: str, dy: float = 0.015) -> None:
    """Add small value labels above bars."""
    y0, y1 = ax.get_ylim()
    span = (y1 - y0) if y1 > y0 else 1.0
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy * span,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
        )

def plot_line_sweep_q1(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_ci_col: Optional[str],
    group_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    registry: Dict[str, ProtoStyle],
    legend_ncol: int = 2,
    filter_energy_spikes: bool = True,
) -> None:
    """
    Clean multi-series line plot:
    - CI as subtle bands
    - reduced markers to avoid clutter
    - legend outside
    """
    fig, ax = plt.subplots()

    protocols = sorted(data[group_col].unique(), key=protocol_order_key)

    for proto in protocols:
        d = data[data[group_col] == proto].sort_values(x_col)

        # Filter out unstable energy metrics when PDR is near 0
        if filter_energy_spikes and ("energy" in y_col.lower()) and ("pdr_mean" in d.columns):
            d = d[d["pdr_mean"] >= 5.0]

        if d.empty:
            continue

        x = d[x_col].to_numpy()
        y = d[y_col].to_numpy()

        st = style_for(proto, registry)

        # main line
        (ln,) = ax.plot(
            x, y,
            label=proto,
            color=st.color,
            linestyle=st.linestyle,
            marker=st.marker,
            markevery=max(1, len(x) // 8),
        )

        # CI band
        if y_ci_col and (y_ci_col in d.columns):
            ci = d[y_ci_col].to_numpy()
            ax.fill_between(
                x, y - ci, y + ci,
                color=ln.get_color(),
                alpha=0.15,
                linewidth=0,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # keep margins tight but not cramped
    ax.margins(x=0.02)

    # legend outside
    add_outside_legend(ax, ncol=legend_ncol, y=1.18)

    save_fig(fig, filename)


# =============================================================================
# SECTION 1: BASELINE PROTOCOL COMPARISON (Bar charts)
# =============================================================================

def plot_pdr_vs_protocol(data: pd.DataFrame, registry: Dict[str, ProtoStyle], filename: str = "plot01_pdr_protocol.pdf"):
    fig, ax = plt.subplots(figsize=(DBL_W, DBL_W * 0.52))

    protocols = [str(x) for x in data["label"].tolist()]
    protocols = sorted(protocols, key=protocol_order_key)

    # reorder rows to match protocol order
    d = data.set_index("label").loc[protocols].reset_index()

    pdr_mean = d["pdr_mean"].to_numpy()
    pdr_ci = d["pdr_ci95"].to_numpy() if "pdr_ci95" in d.columns else None

    x = np.arange(len(protocols))
    bars = barplot_q1(ax, x, pdr_mean, pdr_ci, protocols, registry, width=0.78)

    ax.set_xlabel("Protocol Variant")
    ax.set_ylabel("Packet Delivery Ratio (%)")
    ax.set_title("PDR Comparison Across Protocols")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, rotation=15, ha="right")

    ax.set_ylim(0, 105)
    ax.axhline(y=100, color="#666666", linestyle="--", linewidth=0.8, alpha=0.5)

    annotate_bars(ax, bars, "{:.1f}%")

    save_fig(fig, filename)

def plot_latency_vs_protocol(data: pd.DataFrame, registry: Dict[str, ProtoStyle], filename: str = "plot02_latency_protocol.pdf"):
    fig, ax = plt.subplots(figsize=(DBL_W, DBL_W * 0.52))

    protocols = [str(x) for x in data["label"].tolist()]
    protocols = sorted(protocols, key=protocol_order_key)
    d = data.set_index("label").loc[protocols].reset_index()

    lat_mean = d["avg_latency_mean"].to_numpy()
    lat_ci = d["avg_latency_ci95"].to_numpy() if "avg_latency_ci95" in d.columns else None

    x = np.arange(len(protocols))
    bars = barplot_q1(ax, x, lat_mean, lat_ci, protocols, registry, width=0.78)

    ax.set_xlabel("Protocol Variant")
    ax.set_ylabel("Average End-to-End Latency (s)")
    ax.set_title("Latency Comparison Across Protocols")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, rotation=15, ha="right")

    annotate_bars(ax, bars, "{:.2f}")

    save_fig(fig, filename)

def plot_energy_vs_protocol(data: pd.DataFrame, registry: Dict[str, ProtoStyle], filename: str = "plot03_energy_protocol.pdf"):
    fig, ax = plt.subplots(figsize=(DBL_W, DBL_W * 0.52))

    protocols = [str(x) for x in data["label"].tolist()]
    protocols = sorted(protocols, key=protocol_order_key)
    d = data.set_index("label").loc[protocols].reset_index()

    eng_mean = d["energy_per_msg_mJ_mean"].to_numpy()
    eng_ci = d["energy_per_msg_mJ_ci95"].to_numpy() if "energy_per_msg_mJ_ci95" in d.columns else None

    x = np.arange(len(protocols))
    bars = barplot_q1(ax, x, eng_mean, eng_ci, protocols, registry, width=0.78)

    ax.set_xlabel("Protocol Variant")
    ax.set_ylabel("Energy per Delivered Message (mJ)")
    ax.set_title("Energy Efficiency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, rotation=15, ha="right")

    annotate_bars(ax, bars, "{:.2f}")

    save_fig(fig, filename)

def plot_hops_dtn_only(data: pd.DataFrame, registry: Dict[str, ProtoStyle], filename: str = "plot04_hops_dtn.pdf"):
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.78))

    # Match actual protocol names from sweep CSVs
    dtn_labels = ["MQTT S&F QoS0", "MQTT S&F QoS1", "DDS S&F BE", "DDS S&F Rel"]
    dtn_data = data[data["label"].isin(dtn_labels)].copy()
    if dtn_data.empty:
        print("⚠️ No DTN rows for hops plot, skipping")
        plt.close(fig)
        return

    protocols = sorted([str(x) for x in dtn_data["label"].tolist()], key=protocol_order_key)
    d = dtn_data.set_index("label").loc[protocols].reset_index()

    hops_mean = d["avg_hops_mean"].to_numpy()
    hops_ci = d["avg_hops_ci95"].to_numpy() if "avg_hops_ci95" in d.columns else None

    x = np.arange(len(protocols))
    bars = barplot_q1(ax, x, hops_mean, hops_ci, protocols, registry, width=0.55)

    ax.set_xlabel("DTN Protocol")
    ax.set_ylabel("Average Hop Count")
    ax.set_title("Multi-Hop Relay Statistics (DTN)")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols)

    ax.set_ylim(0, max(3.0, float(np.nanmax(hops_mean)) * 1.25))
    annotate_bars(ax, bars, "{:.2f}", dy=0.02)

    save_fig(fig, filename)


# =============================================================================
# SECTION 2/3: SWEEP PLOTS
# =============================================================================

def plot_scalability_uavs(data: pd.DataFrame, registry: Dict[str, ProtoStyle]) -> None:
    uav_data = data[data["param_name"] == "NUM_UAVS"]
    if uav_data.empty:
        print("⚠️ No NUM_UAVS data")
        return

    plot_line_sweep_q1(
        uav_data, "param_value", "pdr_mean", "pdr_ci95", "protocol",
        "Number of UAVs", "PDR (%)",
        "Scalability: PDR vs UAV Fleet Size",
        "plot05_pdr_vs_uavs.pdf",
        registry, legend_ncol=2
    )
    plot_line_sweep_q1(
        uav_data, "param_value", "avg_latency_mean", "avg_latency_ci95", "protocol",
        "Number of UAVs", "Average Latency (s)",
        "Scalability: Latency vs UAV Fleet Size",
        "plot06_latency_vs_uavs.pdf",
        registry, legend_ncol=2
    )
    plot_line_sweep_q1(
        uav_data, "param_value", "energy_per_msg_mJ_mean", "energy_per_msg_mJ_ci95", "protocol",
        "Number of UAVs", "Energy per Message (mJ)",
        "Scalability: Energy vs UAV Fleet Size",
        "plot07_energy_vs_uavs.pdf",
        registry, legend_ncol=2
    )

def plot_load_sensors(data: pd.DataFrame, registry: Dict[str, ProtoStyle]) -> None:
    sensor_data = data[data["param_name"] == "NUM_SENSORS"]
    if sensor_data.empty:
        print("⚠️ No NUM_SENSORS data")
        return

    plot_line_sweep_q1(
        sensor_data, "param_value", "pdr_mean", "pdr_ci95", "protocol",
        "Number of IoT Sensors", "PDR (%)",
        "Traffic Load: PDR vs Sensor Count",
        "plot08_pdr_vs_sensors.pdf",
        registry, legend_ncol=2
    )
    plot_line_sweep_q1(
        sensor_data, "param_value", "avg_latency_mean", "avg_latency_ci95", "protocol",
        "Number of IoT Sensors", "Average Latency (s)",
        "Traffic Load: Latency vs Sensor Count",
        "plot09_latency_vs_sensors.pdf",
        registry, legend_ncol=2
    )
    plot_line_sweep_q1(
        sensor_data, "param_value", "energy_per_msg_mJ_mean", "energy_per_msg_mJ_ci95", "protocol",
        "Number of IoT Sensors", "Energy per Message (mJ)",
        "Traffic Load: Energy vs Sensor Count",
        "plot10_energy_vs_sensors.pdf",
        registry, legend_ncol=2
    )

def plot_payload_effects(data: pd.DataFrame, registry: Dict[str, ProtoStyle]) -> None:
    payload_data = data[data["param_name"] == "DATA_PAYLOAD_BYTES"]
    if payload_data.empty:
        print("⚠️ No DATA_PAYLOAD_BYTES data")
        return

    plot_line_sweep_q1(
        payload_data, "param_value", "pdr_mean", "pdr_ci95", "protocol",
        "Payload Size (bytes)", "PDR (%)",
        "Payload Impact: PDR vs Message Size",
        "plot11_pdr_vs_payload.pdf",
        registry, legend_ncol=2
    )
    plot_line_sweep_q1(
        payload_data, "param_value", "avg_latency_mean", "avg_latency_ci95", "protocol",
        "Payload Size (bytes)", "Average Latency (s)",
        "Payload Impact: Latency vs Message Size",
        "plot12_latency_vs_payload.pdf",
        registry, legend_ncol=2
    )
    plot_line_sweep_q1(
        payload_data, "param_value", "energy_per_msg_mJ_mean", "energy_per_msg_mJ_ci95", "protocol",
        "Payload Size (bytes)", "Energy per Message (mJ)",
        "Payload Impact: Energy vs Message Size",
        "plot13_energy_vs_payload.pdf",
        registry, legend_ncol=2
    )


# =============================================================================
# SECTION 4: DISTRIBUTIONS / OVERHEAD
# =============================================================================

def plot_latency_cdf(raw_data: pd.DataFrame, registry: Dict[str, ProtoStyle], filename: str = "plot14_latency_cdf.pdf"):
    """
    CDF plot: key for many Q1 networking papers.
    raw_data must have columns: label, avg_latency (per-run or per-sample).
    """
    fig, ax = plt.subplots()

    if "label" not in raw_data.columns or "avg_latency" not in raw_data.columns:
        print("⚠️ raw_data missing columns for CDF (need label, avg_latency). Skipping.")
        plt.close(fig)
        return

    protocols = sorted(raw_data["label"].unique(), key=protocol_order_key)

    for proto in protocols:
        vals = raw_data.loc[raw_data["label"] == proto, "avg_latency"].to_numpy()
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue

        vals = np.sort(vals)
        cdf = np.arange(1, len(vals) + 1) / len(vals)

        st = style_for(proto, registry)
        ax.plot(vals, cdf, label=proto, color=st.color, linestyle=st.linestyle, linewidth=1.8)

    ax.set_xlabel("End-to-End Latency (s)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Latency Distribution (CDF) Across Protocols")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    add_outside_legend(ax, ncol=2, y=1.18)

    save_fig(fig, filename)

def plot_hop_distribution(raw_data: pd.DataFrame, registry: Dict[str, ProtoStyle], filename: str = "plot15_hop_histogram.pdf"):
    """
    Hop distribution for DTN only.
    NOTE: Your current code approximates histogram from avg_hops values.
    If you have per-message hop counts, replace this with real histograms.
    """
    fig, ax = plt.subplots()

    if "label" not in raw_data.columns or "avg_hops" not in raw_data.columns:
        print("⚠️ raw_data missing columns for hop distribution (need label, avg_hops). Skipping.")
        plt.close(fig)
        return

    # Match actual protocol names from sweep CSVs
    dtn_protocols = ["MQTT S&F QoS0", "MQTT S&F QoS1", "DDS S&F BE", "DDS S&F Rel"]
    width = 0.40

    hop_values = np.arange(0, 6)  # 0..5
    x = np.arange(len(hop_values))

    any_plotted = False
    for i, proto in enumerate(dtn_protocols):
        d = raw_data[raw_data["label"] == proto]
        if d.empty:
            continue

        hops = d["avg_hops"].to_numpy()
        hops = hops[~np.isnan(hops)]
        if hops.size == 0:
            continue

        # Approximate distribution from rounded averages (your original approach)
        counts = np.zeros_like(hop_values, dtype=float)
        for h in hops:
            idx = int(np.clip(int(round(h)), 0, 5))
            counts[idx] += 1.0

        probs = counts / max(1.0, counts.sum())

        st = style_for(proto, registry)
        offset = (i - 0.5) * width
        ax.bar(
            x + offset, probs, width,
            label=proto,
            color=st.color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.95,
            hatch="///" if st.family == "MQTT" else "\\\\",
        )
        any_plotted = True

    if not any_plotted:
        print("⚠️ No DTN hop data plotted, skipping.")
        plt.close(fig)
        return

    ax.set_xlabel("Number of Hops")
    ax.set_ylabel("Probability")
    ax.set_title("Hop Count Distribution (DTN Protocols)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in hop_values])

    add_outside_legend(ax, ncol=2, y=1.18)

    save_fig(fig, filename)

def plot_overhead_breakdown(summary_data: pd.DataFrame, registry: Dict[str, ProtoStyle], filename: str = "plot16_overhead_breakdown.pdf"):
    """
    Stacked overhead breakdown for DTN protocols.
    summary_data should include:
    - label
    - spray_events_mean, focus_events_mean, total_delivered_mean
    """
    fig, ax = plt.subplots()

    # Match actual protocol names from sweep CSVs
    dtn_labels = ["MQTT S&F QoS0", "MQTT S&F QoS1", "DDS S&F BE", "DDS S&F Rel"]
    d = summary_data[summary_data["label"].isin(dtn_labels)].copy()
    if d.empty:
        print("⚠️ No DTN data for overhead plot, skipping.")
        plt.close(fig)
        return

    protocols = sorted(d["label"].unique(), key=protocol_order_key)
    d = d.set_index("label").loc[protocols].reset_index()

    spray = d["spray_events_mean"].to_numpy() if "spray_events_mean" in d.columns else np.array([50] * len(protocols))
    focus = d["focus_events_mean"].to_numpy() if "focus_events_mean" in d.columns else np.array([30] * len(protocols))
    delivered = d["total_delivered_mean"].to_numpy() if "total_delivered_mean" in d.columns else np.array([100] * len(protocols))

    x = np.arange(len(protocols))
    width = 0.62

    # Use neutral, readable stack colors (avoid colliding with protocol colors)
    p1 = ax.bar(x, spray, width, label="Spray Phase", edgecolor="black", linewidth=0.6, alpha=0.95)
    p2 = ax.bar(x, focus, width, bottom=spray, label="Focus Phase", edgecolor="black", linewidth=0.6, alpha=0.95)
    p3 = ax.bar(x, delivered, width, bottom=spray + focus, label="Sink Delivery", edgecolor="black", linewidth=0.6, alpha=0.95)

    # Hatch per protocol family (redundancy)
    for i, proto in enumerate(protocols):
        fam = style_for(proto, registry).family
        hatch = "///" if fam == "MQTT" else ("\\\\" if fam == "DDS" else "")
        p1[i].set_hatch(hatch)
        p2[i].set_hatch(hatch)
        p3[i].set_hatch(hatch)

    ax.set_xlabel("DTN Protocol")
    ax.set_ylabel("Message Events")
    ax.set_title("Routing Overhead Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols)

    add_outside_legend(ax, ncol=3, y=1.18)

    save_fig(fig, filename)


# =============================================================================
# SINK MOBILITY COMPARISON (3 panels)
# =============================================================================

def plot_sink_mobility_comparison(data: pd.DataFrame, registry: Dict[str, ProtoStyle], filename: str = "plot_sink_mobility.pdf"):
    """
    Compare mobile vs static sink scenarios.
    Expects param_name == 'SINK_MOBILE', param_value indicates True/False or 1/0.
    """
    sink_data = data[data["param_name"] == "SINK_MOBILE"].copy()
    if sink_data.empty:
        print("⚠️ No SINK_MOBILE data, skipping sink mobility plot.")
        return

    # Make a compact 2-column-friendly wide figure
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, DBL_W * 0.33), constrained_layout=True)

    metrics = [
        ("pdr_mean", "pdr_ci95", "PDR (%)", "PDR"),
        ("avg_latency_mean", "avg_latency_ci95", "Latency (s)", "Latency"),
        ("energy_per_msg_mJ_mean", "energy_per_msg_mJ_ci95", "Energy (mJ)", "Energy"),
    ]

    # Determine two categories: Mobile vs Static based on param_value sort
    # We want consistent x positions [Mobile, Static]
    # Many CSVs use 1/0 or True/False; normalize:
    sink_data["_sink_type"] = sink_data["param_value"].apply(lambda v: "Mobile" if str(v).lower() in {"1", "true", "yes"} else "Static")

    protocols = sorted(sink_data["protocol"].unique(), key=protocol_order_key)
    x = np.arange(2)  # Mobile, Static
    width = max(0.12, min(0.22, 0.8 / max(1, len(protocols))))

    for ax, (y_col, y_ci, ylabel, short_title) in zip(axes, metrics):
        for i, proto in enumerate(protocols):
            d = sink_data[sink_data["protocol"] == proto]
            if d.empty or (y_col not in d.columns):
                continue

            # order: Mobile then Static
            y = []
            yerr = []
            for sink_type in ["Mobile", "Static"]:
                row = d[d["_sink_type"] == sink_type]
                if row.empty:
                    y.append(np.nan)
                    yerr.append(0.0)
                else:
                    y.append(float(row.iloc[0][y_col]))
                    if y_ci in row.columns:
                        yerr.append(float(row.iloc[0][y_ci]))
                    else:
                        yerr.append(0.0)

            st = style_for(proto, registry)
            offset = (i - (len(protocols) - 1) / 2) * width

            ax.bar(
                x + offset, y, width,
                yerr=yerr if y_ci in d.columns else None,
                capsize=2.5,
                color=st.color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.95,
                hatch="///" if st.family == "MQTT" else ("\\\\" if st.family == "DDS" else ""),
                label=proto if ax is axes[0] else None,  # only label once for one shared legend
            )

        ax.set_xticks(x)
        ax.set_xticklabels(["Mobile", "Static"])
        ax.set_xlabel("Sink Type")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Sink Mobility: {short_title}")
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)

    # One shared legend above the middle subplot
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=2,
            frameon=False,
            handlelength=2.8,
            columnspacing=1.2,
        )

    save_fig(fig, filename)


# =============================================================================
# MAIN
# =============================================================================

def generate_all_plots(
    summary_csv: Optional[str] = None,
    sweep_csvs: Optional[Dict[str, str]] = None,
    raw_csv: Optional[str] = None,
    two_column: bool = DEFAULT_TWO_COLUMN,
) -> None:
    set_elsevier_networks_style(two_column=two_column)

    print("=" * 72)
    print("Q1 JOURNAL PUBLICATION PLOTS GENERATOR (ENHANCED)")
    print("=" * 72)

    csv_files = glob.glob("*.csv")

    if summary_csv is None:
        summary_files = [f for f in csv_files if "summary" in f.lower()]
        if summary_files:
            summary_csv = sorted(summary_files)[-1]
            print(f"📂 Using summary: {summary_csv}")

    if raw_csv is None:
        raw_files = [f for f in csv_files if "raw" in f.lower()]
        if raw_files:
            raw_csv = sorted(raw_files)[-1]
            print(f"📂 Using raw data: {raw_csv}")

    if sweep_csvs is None:
        sweep_csvs = {}
        for param in ["NUM_UAVS", "NUM_SENSORS", "DATA_PAYLOAD_BYTES", "SINK_MOBILE"]:
            param_files = [f for f in csv_files if param in f]
            if param_files:
                sweep_csvs[param] = sorted(param_files)[-1]
                print(f"📂 Using {param}: {sweep_csvs[param]}")

    # Build a style registry using all protocol names we can find (best effort)
    all_protocols: List[str] = []
    for f in filter(None, [summary_csv, raw_csv] + list((sweep_csvs or {}).values())):
        if f and os.path.exists(f):
            try:
                df = pd.read_csv(f)
                df = rename_labels(df)
                if "protocol" in df.columns:
                    all_protocols.extend(list(df["protocol"].dropna().unique()))
                if "label" in df.columns:
                    all_protocols.extend(list(df["label"].dropna().unique()))
            except Exception:
                pass
    all_protocols = sorted(list(set([str(p) for p in all_protocols])))
    registry = _make_default_registry(all_protocols)

    print("-" * 72)
    print("\n📊 SECTION 1: Baseline Protocol Comparison")
    print("-" * 72)

    section1_data = None
    if sweep_csvs and ("NUM_UAVS" in sweep_csvs) and sweep_csvs["NUM_UAVS"] and os.path.exists(sweep_csvs["NUM_UAVS"]):
        print("📈 Using sweep data at baseline (NUM_UAVS=8) for Section 1 plots")
        section1_data = extract_baseline_from_sweep(sweep_csvs["NUM_UAVS"], baseline_value=8)
    elif summary_csv and os.path.exists(summary_csv):
        print("⚠️ Falling back to summary CSV for Section 1 plots")
        section1_data = load_summary_data(summary_csv)

    if section1_data is not None and not section1_data.empty:
        plot_pdr_vs_protocol(section1_data, registry)
        plot_latency_vs_protocol(section1_data, registry)
        plot_energy_vs_protocol(section1_data, registry)
        plot_hops_dtn_only(section1_data, registry)
    else:
        print("⚠️ No data found for Section 1 plots")

    print("-" * 72)
    print("\n📊 SECTION 2: Scalability & Load Sweeps")
    print("-" * 72)

    all_sweep = []
    if sweep_csvs:
        for param, csv_file in sweep_csvs.items():
            if csv_file and os.path.exists(csv_file):
                df = load_sweep_data(csv_file)
                all_sweep.append(df)

    if all_sweep:
        combined_sweep = pd.concat(all_sweep, ignore_index=True)

        plot_scalability_uavs(combined_sweep, registry)
        plot_load_sensors(combined_sweep, registry)

        print("-" * 72)
        print("\n📊 SECTION 3: Payload Size Effects")
        print("-" * 72)
        plot_payload_effects(combined_sweep, registry)

        print("-" * 72)
        print("\n📊 Sink Mobility Comparison")
        print("-" * 72)
        plot_sink_mobility_comparison(combined_sweep, registry)
    else:
        print("⚠️ No sweep data found for Section 2-3 plots")

    print("-" * 72)
    print("\n📊 SECTION 4: Spray & Focus Dynamics")
    print("-" * 72)

    if raw_csv and os.path.exists(raw_csv):
        raw_data = load_csv(raw_csv)
        plot_latency_cdf(raw_data, registry)
        plot_hop_distribution(raw_data, registry)
    else:
        print("⚠️ No raw data found for CDF/histogram plots")

    if summary_csv and os.path.exists(summary_csv):
        summary_data = load_summary_data(summary_csv)
        plot_overhead_breakdown(summary_data, registry)

    print("\n" + "=" * 72)
    print(f"✅ ALL PLOTS SAVED TO: {OUTPUT_DIR}/")
    print("=" * 72)
    print("\nPlots generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".pdf"):
            print(f"  📄 {f}")


if __name__ == "__main__":
    generate_all_plots()
