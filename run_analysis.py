#!/usr/bin/env python3
"""
All-in-One Cleaning, Aggregation, Statistical Analysis & Plotting Script
Project: Dissociating Stimulus Array Complexity vs. Perceptual Task Complexity
         in Audiovisual Integration (Talsma et al., 2010, Box 3 Q4)

Directory layout expected:
  parent_folder/
    ├── run_analysis.py          <-- Save and run this script here
    └── data/
        ├── P001_project_2026-11-18_12h32.41.865.csv
        ├── P004_project_2026-11-23_22h35.35.360.csv
        ├── 005_project_2026-11-23_22h11.36.804.csv
        └── P006_project_2026-11-23_22h43.22.563.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURATION: STRICTLY INCLUDE ONLY THE 4 SPECIFIED CSV FILES
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_FILES = [
    "P001_project_2026-11-18_12h32.41.865.csv",
    "P004_project_2026-11-23_22h35.35.360.csv",
    "005_project_2026-11-23_22h11.36.804.csv",
    "P006_project_2026-11-23_22h43.22.563.csv",
]

CONDITION_LABELS = {
    1: "B1: Simple Array\nSimple Task",
    2: "B2: Simple Array\nComplex Task",
    3: "B3: Complex Array\nSimple Task",
    4: "B4: Complex Array\nComplex Task",
}


# ==============================================================================
# 2. LOAD & CLEAN TRIAL-LEVEL DATA
# ==============================================================================
def normalize_pid(raw_val, filename):
    """Standardize participant IDs (e.g., '5' -> 'P005', 'P001' -> 'P001')."""
    s = str(raw_val).strip() if pd.notna(raw_val) else filename.split("_")[0]
    if s.isdigit():
        return f"P{int(s):03d}"
    return s.upper()


def load_and_clean(data_dir, target_files):
    all_trials = []
    for fname in target_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected CSV not found: {fpath}")

        df = pd.read_csv(fpath)
        # Keep only actual trial rows (excludes intro/block summary rows)
        trials = df[df["trials_loop.thisN"].notna()].copy()

        raw_pid = trials["participant"].dropna().iloc[0] if trials["participant"].notna().any() else None
        trials["pid"] = normalize_pid(raw_pid, fname)
        trials["source_file"] = fname
        all_trials.append(trials)

    raw_df = pd.concat(all_trials, ignore_index=True)

    # Standardize types & units
    raw_df["cond_id"] = raw_df["cond_id"].astype(int)
    raw_df["block_order_pos"] = raw_df["blocks_loop.thisN"].astype(int) + 1
    raw_df["trial_num_in_block"] = raw_df["trials_loop.thisN"].astype(int) + 1
    raw_df["probe"] = raw_df["probe"].astype(str).str.strip().str.lower()
    raw_df["tone"] = raw_df["tone"].astype(int)
    raw_df["array_type"] = raw_df["array_type"].astype(str).str.strip().str.lower()
    raw_df["task_type"] = raw_df["task_type"].astype(str).str.strip().str.lower()
    raw_df["resp_key"] = raw_df["key_resp_trial.keys"].astype(str).str.strip().str.lower()
    raw_df["corr_key"] = raw_df["corr_resp"].astype(str).str.strip().str.lower()
    raw_df["corr"] = raw_df["key_resp_trial.corr"].astype(int)
    raw_df["rt_ms"] = raw_df["key_resp_trial.rt"].astype(float) * 1000.0

    # Outlier trimming on correct trials within each participant x block:
    # Exclude anticipatory (< 150 ms) and slow lapse (> Mean + 2.5 SD) responses
    raw_df["rt_valid"] = False
    for (pid, cid), _ in raw_df.groupby(["pid", "cond_id"]):
        base_mask = (
            (raw_df["pid"] == pid)
            & (raw_df["cond_id"] == cid)
            & (raw_df["corr"] == 1)
            & (raw_df["rt_ms"] >= 150.0)
        )
        valid_rts = raw_df.loc[base_mask, "rt_ms"]
        upper_cutoff = valid_rts.mean() + 2.5 * valid_rts.std(ddof=1)
        raw_df.loc[base_mask & (raw_df["rt_ms"] <= upper_cutoff), "rt_valid"] = True

    clean_cols = [
        "pid", "cond_id", "block_order_pos", "trial_num_in_block",
        "array_type", "task_type", "probe", "tone",
        "corr_key", "resp_key", "corr", "rt_ms", "rt_valid", "source_file"
    ]
    return raw_df[clean_cols].sort_values(["pid", "block_order_pos", "trial_num_in_block"]).reset_index(drop=True)


# ==============================================================================
# 3. AGGREGATE PARTICIPANT x BLOCK METRICS (RT, DELTA-RT, SDT d' & c)
# ==============================================================================
def compute_participant_summary(clean_df):
    rows = []
    for (pid, cid, arr, tsk), grp in clean_df.groupby(["pid", "cond_id", "array_type", "task_type"]):
        n_trials = len(grp)
        acc = grp["corr"].mean()

        # Signal Detection Theory (Hautus log-linear correction for 0% / 100% rates)
        sig = grp[grp["corr_key"] == "y"]
        noise = grp[grp["corr_key"] == "n"]
        hits = int((sig["resp_key"] == "y").sum())
        fas = int((noise["resp_key"] == "y").sum())
        n_sig = len(sig)
        n_noise = len(noise)

        h_adj = (hits + 0.5) / (n_sig + 1.0)
        fa_adj = (fas + 0.5) / (n_noise + 1.0)
        d_prime = float(stats.norm.ppf(h_adj) - stats.norm.ppf(fa_adj))
        criterion_c = float(-0.5 * (stats.norm.ppf(h_adj) + stats.norm.ppf(fa_adj)))

        # Trimmed correct reaction times (ms)
        v = grp[grp["rt_valid"]]
        rt_all_corr = v["rt_ms"].mean()
        rt_av_target = v[(v["probe"] == "circle") & (v["tone"] == 1)]["rt_ms"].mean()
        rt_v_circle = v[(v["probe"] == "circle") & (v["tone"] == 0)]["rt_ms"].mean()

        # Delta RT: valid in Simple Task blocks (1 & 3) where both require 'y'
        delta_rt = (rt_v_circle - rt_av_target) if tsk == "simple" else np.nan

        # Lure-specific false alarm breakdown in Complex Task blocks (2 & 4)
        vis_lure = grp[(grp["probe"] == "circle") & (grp["tone"] == 0)]
        aud_lure = grp[(grp["probe"] == "triangle") & (grp["tone"] == 1)]
        fa_vis_lure = (vis_lure["resp_key"] == "y").mean() if tsk == "complex" else np.nan
        fa_aud_lure = (aud_lure["resp_key"] == "y").mean() if tsk == "complex" else np.nan

        rows.append({
            "pid": pid,
            "cond_id": cid,
            "array_type": arr,
            "task_type": tsk,
            "n_trials": n_trials,
            "n_rt_valid": int(grp["rt_valid"].sum()),
            "accuracy": acc,
            "hit_rate_raw": hits / n_sig,
            "fa_rate_raw": fas / n_noise,
            "d_prime": d_prime,
            "criterion_c": criterion_c,
            "rt_all_corr_ms": rt_all_corr,
            "rt_av_target_ms": rt_av_target,
            "rt_v_circle_ms": rt_v_circle,
            "delta_rt_ms": delta_rt,
            "fa_vis_lure": fa_vis_lure,
            "fa_aud_lure": fa_aud_lure,
        })

    return pd.DataFrame(rows).sort_values(["pid", "cond_id"]).reset_index(drop=True)


# ==============================================================================
# 4. STATISTICAL TESTS: 2x2 REPEATED-MEASURES ANOVA & PLANNED CONTRASTS
# ==============================================================================
def rm_anova_2x2(summary_df, dv):
    """Exact 2 (Array: simple/complex) x 2 (Task: simple/complex) within-subject RM-ANOVA."""
    piv = summary_df.pivot(index="pid", columns=["array_type", "task_type"], values=dv)
    cols = [
        ("simple", "simple"),
        ("simple", "complex"),
        ("complex", "simple"),
        ("complex", "complex"),
    ]
    Y = piv[cols].values
    n_subj = Y.shape[0]

    contrasts = {
        "Array Complexity (Complex - Simple)": np.array([-1, -1, 1, 1]) / 2.0,
        "Task Complexity (Complex - Simple)": np.array([-1, 1, -1, 1]) / 2.0,
        "Array x Task Interaction": np.array([1, -1, -1, 1]) / 2.0,
    }

    out = []
    for effect_name, c in contrasts.items():
        diff = Y @ c
        t_stat, p_val = stats.ttest_1samp(diff, 0.0)
        f_stat = float(t_stat ** 2)
        ss_eff = n_subj * (diff.mean() ** 2) / np.sum(c ** 2)
        ss_err = np.sum((diff - diff.mean()) ** 2) / np.sum(c ** 2)
        eta_p2 = float(ss_eff / (ss_eff + ss_err)) if (ss_eff + ss_err) > 0 else 0.0
        out.append({
            "Effect": effect_name,
            "F": f_stat,
            "df1": 1,
            "df2": n_subj - 1,
            "p": float(p_val),
            "eta_p2": eta_p2,
            "Mean_Diff": float(diff.mean()),
            "SD_Diff": float(diff.std(ddof=1)),
        })
    return pd.DataFrame(out)


# ==============================================================================
# 5. VISUALIZATION: 4-PANEL PUBLICATION SUMMARY FIGURE
# ==============================================================================
def plot_results(summary_df, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # Panel A: Audiovisual Target RT (Red Circle + Tone) across all 4 blocks
    ax = axes[0, 0]
    means = [summary_df[summary_df["cond_id"] == c]["rt_av_target_ms"].mean() for c in [1, 2, 3, 4]]
    sems = [summary_df[summary_df["cond_id"] == c]["rt_av_target_ms"].sem() for c in [1, 2, 3, 4]]
    bars = ax.bar([0, 1, 2, 3], means, yerr=sems, capsize=5, color=colors, edgecolor="black", alpha=0.88)
    for pid, pgrp in summary_df.groupby("pid"):
        pgrp = pgrp.sort_values("cond_id")
        ax.plot([0, 1, 2, 3], pgrp["rt_av_target_ms"], marker="o", color="gray", alpha=0.45, linewidth=1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([CONDITION_LABELS[c] for c in [1, 2, 3, 4]], fontsize=8.5)
    ax.set_ylabel("Reaction Time (ms)")
    ax.set_title("A. Audiovisual Target RT (Red Circle + Tone)", fontweight="bold")

    # Panel B: Signal Detection Sensitivity (d') across all 4 blocks
    ax = axes[0, 1]
    means_d = [summary_df[summary_df["cond_id"] == c]["d_prime"].mean() for c in [1, 2, 3, 4]]
    sems_d = [summary_df[summary_df["cond_id"] == c]["d_prime"].sem() for c in [1, 2, 3, 4]]
    ax.bar([0, 1, 2, 3], means_d, yerr=sems_d, capsize=5, color=colors, edgecolor="black", alpha=0.88)
    for pid, pgrp in summary_df.groupby("pid"):
        pgrp = pgrp.sort_values("cond_id")
        ax.plot([0, 1, 2, 3], pgrp["d_prime"], marker="o", color="gray", alpha=0.45, linewidth=1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([CONDITION_LABELS[c] for c in [1, 2, 3, 4]], fontsize=8.5)
    ax.set_ylabel("Sensitivity (d')")
    ax.set_title("B. Perceptual Sensitivity (d') by Condition", fontweight="bold")

    # Panel C: Multisensory Facilitation (Delta RT) in Simple Task (Block 1 vs Block 3)
    ax = axes[1, 0]
    b1 = summary_df[summary_df["cond_id"] == 1].sort_values("pid")["delta_rt_ms"]
    b3 = summary_df[summary_df["cond_id"] == 3].sort_values("pid")["delta_rt_ms"]
    ax.bar([0, 1], [b1.mean(), b3.mean()], yerr=[b1.sem(), b3.sem()], capsize=6,
           color=["#4C72B0", "#55A868"], edgecolor="black", width=0.5, alpha=0.88)
    for v1, v3 in zip(b1, b3):
        ax.plot([0, 1], [v1, v3], marker="o", color="gray", alpha=0.5, linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Block 1: Simple Array\n(Simple Task)", "Block 3: Complex Array\n(Simple Task)"], fontsize=9)
    ax.set_ylabel("Delta RT = RT(Visual) - RT(AV) [ms]")
    ax.set_title("C. Automatic Multisensory Facilitation (Delta RT)", fontweight="bold")

    # Panel D: Complex-Task Lure False Alarm Breakdown (Block 2 vs Block 4)
    ax = axes[1, 1]
    b2 = summary_df[summary_df["cond_id"] == 2]
    b4 = summary_df[summary_df["cond_id"] == 4]
    x = np.arange(2)
    w = 0.34
    vis_means = [b2["fa_vis_lure"].mean() * 100, b4["fa_vis_lure"].mean() * 100]
    aud_means = [b2["fa_aud_lure"].mean() * 100, b4["fa_aud_lure"].mean() * 100]
    vis_sems = [b2["fa_vis_lure"].sem() * 100, b4["fa_vis_lure"].sem() * 100]
    aud_sems = [b2["fa_aud_lure"].sem() * 100, b4["fa_aud_lure"].sem() * 100]
    ax.bar(x - w / 2, vis_means, yerr=vis_sems, width=w, capsize=4, label="Visual Lure (Circle + No Tone)", color="#DD8452", edgecolor="black")
    ax.bar(x + w / 2, aud_means, yerr=aud_sems, width=w, capsize=4, label="Auditory Lure (Triangle + Tone)", color="#8172B3", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(["Block 2: Simple Array\n(Complex Task)", "Block 4: Complex Array\n(Complex Task)"], fontsize=9)
    ax.set_ylabel("False Alarm Rate (%)")
    ax.set_title("D. Conjunction Lure Errors in Complex Tasks", fontweight="bold")
    ax.legend(fontsize=8.5, frameon=True)

    plt.suptitle("Multisensory Visual Search: Array Complexity vs. Task Complexity (N = 4)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    print("=" * 72)
    print("RUNNING MULTISENSORY VISUAL SEARCH CLEANING & ANALYSIS PIPELINE")
    print("=" * 72)

    clean_df = load_and_clean(DATA_DIR, TARGET_FILES)
    summary_df = compute_participant_summary(clean_df)

    # Save cleaned trial-level and aggregated participant-level datasets
    clean_csv = os.path.join(OUT_DIR, "cleaned_trials.csv")
    summary_csv = os.path.join(OUT_DIR, "participant_block_summary.csv")
    clean_df.to_csv(clean_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    n_total = len(clean_df)
    n_corr = int(clean_df["corr"].sum())
    n_valid_rt = int(clean_df["rt_valid"].sum())
    print(f"Participants included ({summary_df['pid'].nunique()}): {sorted(summary_df['pid'].unique())}")
    print(f"Total trials: {n_total} | Correct: {n_corr} ({n_corr/n_total:.1%}) | RT-valid after trimming: {n_valid_rt}")

    # Group descriptive statistics
    desc = (
        summary_df.groupby(["cond_id", "array_type", "task_type"])[
            ["accuracy", "d_prime", "criterion_c", "rt_all_corr_ms", "rt_av_target_ms", "delta_rt_ms", "fa_vis_lure", "fa_aud_lure"]
        ]
        .agg(["mean", "std"])
        .round(3)
    )

    # Planned contrasts on Delta RT
    b1_drt = summary_df[summary_df["cond_id"] == 1].sort_values("pid")["delta_rt_ms"].values
    b3_drt = summary_df[summary_df["cond_id"] == 3].sort_values("pid")["delta_rt_ms"].values
    t_b1, p_b1 = stats.ttest_1samp(b1_drt, 0.0)
    t_13, p_13 = stats.ttest_rel(b1_drt, b3_drt)

    # Build full statistical report
    report_lines = [
        "=== GROUP DESCRIPTIVE STATISTICS (MEAN & SD) ===",
        desc.to_string(),
        "",
        "=== PLANNED CONTRASTS ON MULTISENSORY FACILITATION (DELTA RT) ===",
        f"Block 1 (Simple Array, Simple Task) Delta RT vs. 0: Mean = {b1_drt.mean():.2f} ms, SD = {b1_drt.std(ddof=1):.2f} ms, t(3) = {t_b1:.3f}, p = {p_b1:.4f}",
        f"Block 3 (Complex Array, Simple Task) Delta RT:      Mean = {b3_drt.mean():.2f} ms, SD = {b3_drt.std(ddof=1):.2f} ms",
        f"Block 1 vs. Block 3 Paired Comparison:              Diff = {(b1_drt - b3_drt).mean():.2f} ms, t(3) = {t_13:.3f}, p = {p_13:.4f}",
    ]

    for dv_label, dv_col in [
        ("Audiovisual Target RT (rt_av_target_ms)", "rt_av_target_ms"),
        ("Overall Correct RT (rt_all_corr_ms)", "rt_all_corr_ms"),
        ("Perceptual Sensitivity (d_prime)", "d_prime"),
        ("Response Bias (criterion_c)", "criterion_c"),
        ("Overall Accuracy (accuracy)", "accuracy"),
    ]:
        anova_tbl = rm_anova_2x2(summary_df, dv_col).round(4)
        report_lines.extend(["", f"=== 2x2 REPEATED-MEASURES ANOVA: {dv_label} ===", anova_tbl.to_string()])

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = os.path.join(OUT_DIR, "statistical_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    fig_path = os.path.join(OUT_DIR, "results_summary.png")
    plot_results(summary_df, fig_path)

    print("\n" + "=" * 72)
    print("OUTPUT FILES SAVED TO:", OUT_DIR)
    print("  1. cleaned_trials.csv            (768 cleaned trial rows)")
    print("  2. participant_block_summary.csv (16 participant x block rows for JASP/R)")
    print("  3. statistical_results.txt       (Full descriptives, t-tests, 2x2 ANOVAs)")
    print("  4. results_summary.png           (4-panel summary chart)")
    print("=" * 72)


if __name__ == "__main__":
    main()
