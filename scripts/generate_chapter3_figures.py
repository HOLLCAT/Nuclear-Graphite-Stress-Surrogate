#!/usr/bin/env python3
"""Generate the proposed Chapter 3 methodology and data figures.

The script reads only the existing project data and formal QC outputs. It creates:

2. Four spatial fields from one example development case for Section 3.2.
3. Input ranges across the 149 development cases for Section 3.2.

PNG files are provided for quick inspection. PDF files are vector outputs for LaTeX.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / ".cache" / "matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


RAW_DATA_DIR = PROJECT_ROOT / "FE_Results_Cases_All"
QC_DIR = PROJECT_ROOT / "outputs" / "00_qc_sensitivity_ablation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "thesis_figures" / "chapter3"

INK = "#263238"
BLUE = "#3973A5"
BLUE_LIGHT = "#E8F1F8"
GOLD = "#C58A2A"
GOLD_LIGHT = "#F7EEDB"
GREY = "#68747D"
GREY_LIGHT = "#EFF2F4"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = BLUE,
    fontsize: float = 10,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    axis.add_patch(box)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    connectionstyle: str = "arc3,rad=0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.4,
        color=GREY,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    axis.add_patch(arrow)




def read_case(case_number: int) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"FE_Results_Case_{case_number}.txt"
    if not path.exists():
        raise FileNotFoundError(f"FE case file not found: {path}")
    data = pd.read_csv(path, skipinitialspace=True)
    data.columns = data.columns.str.strip()
    required = {
        "ElementID",
        "X",
        "Y",
        "Z",
        "FluenceRate",
        "Temperature",
        "WeightLossRate",
        "MaxPrincipalStress",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required fields in {path.name}: {sorted(missing)}")
    return data


def assert_development_case(case_number: int) -> None:
    summary_path = QC_DIR / "development_case_summary.csv"
    summary = pd.read_csv(summary_path)
    if case_number not in set(summary["case_number"].astype(int)):
        raise ValueError(
            f"Case {case_number} is not listed in development_case_summary.csv. "
            "Choose a development case for the Section 3.2 example."
        )


def plot_example_spatial_fields(
    output_dir: Path,
    case_number: int,
    angular_half_width: float,
    max_plot_points: int,
) -> None:
    """Plot a radial-axial view through an angular band of one FE case."""
    assert_development_case(case_number)
    data = read_case(case_number)

    x = data["X"].to_numpy(dtype=float)
    y = data["Y"].to_numpy(dtype=float)
    data = data.assign(
        rho=np.hypot(x, y),
        theta=np.arctan2(y, x),
    )
    central_theta = float(data["theta"].median())
    view = data.loc[
        (data["theta"] - central_theta).abs() <= angular_half_width
    ].copy()
    if view.empty:
        raise ValueError("The selected angular band contains no elements.")

    if len(view) > max_plot_points:
        positions = np.linspace(0, len(view) - 1, max_plot_points, dtype=int)
        view = view.iloc[positions].copy()

    panels = [
        ("FluenceRate", "Fluence rate", "cividis"),
        ("Temperature", "Temperature", "cividis"),
        ("WeightLossRate", "Weight-loss rate", "cividis"),
        ("MaxPrincipalStress", "Maximum principal stress", "magma"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.0), sharex=True, sharey=True)
    for axis, (column, title, colour_map) in zip(axes.ravel(), panels):
        scatter = axis.scatter(
            view["rho"],
            view["Z"],
            c=view[column],
            s=3.2,
            cmap=colour_map,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(title, weight="semibold")
        axis.set_xlabel(r"Radial coordinate, $\rho$")
        axis.set_ylabel(r"Axial coordinate, $z$")
        axis.grid(False)
        colour_bar = fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.03)
        colour_bar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Example FE input and response fields: development case {case_number:02d}",
        fontsize=14,
        weight="semibold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        (
            f"Radial–axial view from |theta - median(theta)| <= "
            f"{angular_half_width:.3f} rad; {len(view):,} element records plotted. "
            "Units are not shown because they are not confirmed in the supplied metadata."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=GREY,
    )
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.10, top=0.93, wspace=0.20, hspace=0.18)
    save_figure(
        fig,
        output_dir,
        f"figure_3_2_example_fe_fields_case_{case_number:02d}",
    )


def plot_development_input_ranges(output_dir: Path) -> None:
    """Plot within-case ranges for the three FE input variables."""
    summary_path = QC_DIR / "development_case_summary.csv"
    summary = pd.read_csv(summary_path)
    if len(summary) != 149:
        raise ValueError(
            f"Expected 149 development cases in {summary_path.name}; found {len(summary)}."
        )

    variables = [
        ("fluence_rate", "Fluence rate"),
        ("temperature", "Temperature"),
        ("weight_loss_rate", "Weight-loss rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 8.3), sharey=True)

    for axis, (prefix, label) in zip(axes, variables):
        ordered = summary.sort_values(f"{prefix}_mean").reset_index(drop=True)
        rank = np.arange(1, len(ordered) + 1)
        axis.hlines(
            rank,
            ordered[f"{prefix}_min"],
            ordered[f"{prefix}_max"],
            color=BLUE,
            alpha=0.28,
            linewidth=1.0,
            label="Minimum–maximum",
        )
        axis.scatter(
            ordered[f"{prefix}_p95"],
            rank,
            s=10,
            facecolors="white",
            edgecolors=GOLD,
            linewidths=0.7,
            label="95th percentile",
            zorder=3,
        )
        axis.scatter(
            ordered[f"{prefix}_mean"],
            rank,
            s=10,
            color=INK,
            linewidths=0,
            label="Mean",
            zorder=4,
        )
        axis.set_title(label, weight="semibold")
        axis.set_xlabel("Recorded value")
        axis.grid(axis="x", color="#D9DEE2", linewidth=0.6)
        axis.grid(axis="y", visible=False)
        axis.set_ylim(150, 0)
        axis.set_yticks([1, 25, 50, 75, 100, 125, 149])

    axes[0].set_ylabel("Case rank after ordering by the panel mean")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Input-field ranges across the 149 development cases",
        fontsize=14,
        weight="semibold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.025,
        (
            "Cases are ordered separately in each panel. Each interval is calculated from "
            "all 400,360 element records in that case. Units are not shown because they are "
            "not confirmed in the supplied metadata."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=GREY,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.86, wspace=0.20)
    save_figure(fig, output_dir, "figure_3_3_development_input_ranges")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated PNG and PDF files.",
    )
    parser.add_argument(
        "--case-number",
        type=int,
        default=1,
        help="Development case used for the spatial-field example.",
    )
    parser.add_argument(
        "--angular-half-width",
        type=float,
        default=0.015,
        help="Half-width in radians around the median angular coordinate.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=25_000,
        help="Maximum element records plotted in the spatial-field figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    plot_example_spatial_fields(
        args.output_dir,
        args.case_number,
        args.angular_half_width,
        args.max_plot_points,
    )
    plot_development_input_ranges(args.output_dir)
    print(f"Chapter 3 figures written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
