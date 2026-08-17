#!/usr/bin/env python3
"""Plot the post-training Wave-2DGS pruning storage--quality trade-off."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    rows.sort(key=lambda row: float(row["threshold"]))
    return rows


def read_scene_curves(path, thresholds):
    grouped = defaultdict(lambda: defaultdict(list))
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["scene"]][float(row["threshold"])].append(
                float(row["delta_psnr_vs_tau0"])
            )
    curves = {}
    for scene, by_threshold in sorted(grouped.items()):
        if all(float(value) in by_threshold for value in thresholds):
            curves[scene] = np.asarray(
                [float(np.mean(by_threshold[float(value)])) for value in thresholds]
            )
    return curves


def threshold_label(value):
    labels = {
        0.0: r"$0$",
        1e-4: r"$10^{-4}$",
        3e-4: r"$3\!\times\!10^{-4}$",
        1e-3: r"$10^{-3}$",
        3e-3: r"$3\!\times\!10^{-3}$",
        1e-2: r"$10^{-2}$",
    }
    return labels.get(value, f"{value:g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="output/posttrain_wave_pruning_replica_3k/aggregate_by_threshold.csv",
    )
    parser.add_argument(
        "--output_prefix",
        default="output/paper_visuals/wave_pruning_tradeoff",
    )
    parser.add_argument(
        "--runs_input",
        default="output/posttrain_wave_pruning_replica_3k/all_runs.csv",
    )
    parser.add_argument(
        "--scene_curves", choices=("none", "gray", "color"), default="gray",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    rows = read_rows(args.input)
    thresholds = np.asarray([float(row["threshold"]) for row in rows])
    scene_curves = (
        read_scene_curves(args.runs_input, thresholds)
        if args.scene_curves != "none" else {}
    )
    active = 100.0 * np.asarray(
        [float(row["active_fraction_mean"]) for row in rows]
    )
    active_std = 100.0 * np.asarray(
        [float(row["active_fraction_std"]) for row in rows]
    )
    checkpoint_saved = 100.0 * np.asarray(
        [float(row["checkpoint_reduction_raw_fraction_mean"]) for row in rows]
    )
    checkpoint_saved_std = 100.0 * np.asarray(
        [float(row["checkpoint_reduction_raw_fraction_std"]) for row in rows]
    )
    delta_psnr = np.asarray(
        [float(row["delta_psnr_vs_tau0_mean"]) for row in rows]
    )
    delta_psnr_std = np.asarray(
        [float(row["delta_psnr_vs_tau0_std"]) for row in rows]
    )

    # tau=0 denotes the original dense checkpoint, not a sparse sidecar with
    # index overhead. Use it as the exact zero-storage-reduction reference.
    zero = np.isclose(thresholds, 0.0)
    checkpoint_saved[zero] = 0.0
    checkpoint_saved_std[zero] = 0.0
    wave_payload_removed = 100.0 - active

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(rows))
    fig_height = 3.45 if args.scene_curves == "color" else 3.2
    fig, storage_ax = plt.subplots(figsize=(7.15, fig_height))
    quality_ax = storage_ax.twinx()

    bar_color = "#4C78A8"
    line_color = "#E45756"
    bars = storage_ax.bar(
        x,
        checkpoint_saved,
        width=0.58,
        color=bar_color,
        alpha=0.88,
        yerr=checkpoint_saved_std,
        capsize=2.5,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
        label="Complete checkpoint reduction",
        zorder=2,
    )
    scene_handles = []
    if args.scene_curves == "gray":
        for values in scene_curves.values():
            quality_ax.plot(
                x, values, color="#777777", alpha=0.32,
                linewidth=1.0, zorder=2.7,
            )
        if scene_curves:
            scene_handles.append(
                Line2D(
                    [0], [0], color="#777777", alpha=0.5, linewidth=1.2,
                    label="Per-scene means (3 seeds)",
                )
            )
    elif args.scene_curves == "color":
        palette = plt.get_cmap("tab10")
        for index, (scene, values) in enumerate(scene_curves.items()):
            handle, = quality_ax.plot(
                x, values, color=palette(index % 10), alpha=0.68,
                linewidth=1.15, label=scene, zorder=2.7,
            )
            scene_handles.append(handle)
    quality_handle = quality_ax.errorbar(
        x,
        delta_psnr,
        yerr=delta_psnr_std,
        color=line_color,
        marker="o",
        markersize=5.0,
        linewidth=2.0,
        capsize=3.0,
        elinewidth=1.0,
        label=r"$\Delta$PSNR vs. unpruned",
        zorder=4,
    )
    quality_ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)

    if len(rows) <= 10:
        tick_indices = np.arange(len(rows))
    else:
        tick_indices = np.unique(
            np.append(np.arange(0, len(rows), 4), len(rows) - 1)
        )
    tick_labels = [
        f"{threshold_label(thresholds[index])}\n{active[index]:.1f}% active"
        for index in tick_indices
    ]
    storage_ax.set_xticks(x[tick_indices])
    storage_ax.set_xticklabels(tick_labels)
    storage_ax.set_xlabel(r"Post-training amplitude threshold $\tau$")
    storage_ax.set_ylabel("Complete checkpoint reduction (%)", color=bar_color)
    quality_ax.set_ylabel(
        r"$\Delta$PSNR vs. unpruned (dB) $\uparrow$",
        color=line_color,
    )
    storage_ax.tick_params(axis="y", colors=bar_color)
    quality_ax.tick_params(axis="y", colors=line_color)
    quality_ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    storage_top = max(8.7, float(np.max(checkpoint_saved + checkpoint_saved_std)) * 1.18)
    storage_ax.set_ylim(0.0, storage_top)
    scene_low = min(
        (float(np.min(values)) for values in scene_curves.values()), default=0.0)
    quality_low = min(
        -0.010, float(np.min(delta_psnr - delta_psnr_std)) * 1.16,
        scene_low * 1.12,
    )
    quality_ax.set_ylim(quality_low, 0.0015)
    storage_ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)
    storage_ax.set_axisbelow(True)

    label_indices = (set(range(len(rows))) if len(rows) <= 10
                     else set(int(index) for index in tick_indices))
    for index, (bar, value) in enumerate(zip(bars, checkpoint_saved)):
        if index not in label_indices:
            continue
        storage_ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(value, 0.0) + 0.18,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=bar_color,
        )

    last = len(rows) - 1
    if args.scene_curves == "color":
        storage_ax.legend(
            [bars, quality_handle] + scene_handles,
            ["Complete checkpoint reduction", r"$\Delta$PSNR vs. unpruned"]
            + [h.get_label() for h in scene_handles],
            loc="lower center", bbox_to_anchor=(0.5, 1.01),
            frameon=False, ncol=5, fontsize=7.2, columnspacing=0.9,
            handlelength=1.6,
        )
    else:
        storage_ax.legend(
            [bars] + scene_handles + [quality_handle],
            ["Complete checkpoint reduction"]
            + [h.get_label() for h in scene_handles]
            + [r"$\Delta$PSNR vs. unpruned"],
            loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, ncol=3,
            columnspacing=1.1, handlelength=1.8, fontsize=8.2,
        )

    for axis in (storage_ax, quality_ax):
        axis.spines["top"].set_visible(False)
    storage_ax.spines["right"].set_visible(False)
    quality_ax.spines["left"].set_visible(False)

    fig.tight_layout(pad=0.7)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(
        f"tau={thresholds[last]:g}: active={active[last]:.2f}% "
        f"(std {active_std[last]:.2f}), complete checkpoint reduction="
        f"{checkpoint_saved[last]:.2f}%, wave payload removed="
        f"{wave_payload_removed[last]:.2f}%, delta PSNR={delta_psnr[last]:+.6f} dB"
    )


if __name__ == "__main__":
    main()
