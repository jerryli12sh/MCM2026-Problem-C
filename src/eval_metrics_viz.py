import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _parse_list(s: str) -> List[str]:
    if pd.isna(s) or s == "":
        return []
    return [x for x in str(s).split("|") if x != ""]


def compute_top1_accuracy(event_long: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    events = event_table.copy()
    events = events[events["m_elim"] == 1].copy()

    for _, e in events.iterrows():
        s = int(e["season"])
        w = int(e["week"])
        elim_obs = _parse_list(e["elim_obs_list"])
        if len(elim_obs) != 1:
            continue
        g = event_long[(event_long["season"] == s) & (event_long["week"] == w)].copy()
        if g.empty:
            continue
        pred = g.sort_values("c_hat", ascending=True).iloc[0]["celebrity_name"]
        acc = 1 if pred == elim_obs[0] else 0
        rows.append({"season": s, "week": w, "accuracy": acc})

    out = pd.DataFrame(rows)
    return out


def compute_cum_consistency(event_long: pd.DataFrame, event_table: pd.DataFrame,
                            *, require_posterior: bool = False) -> pd.DataFrame:
    rows = []

    for season, evs in event_table.groupby("season"):
        evs = evs.sort_values("week")
        obs_cum = set()
        pred_cum = set()
        terms = []
        k = 0

        for _, e in evs.iterrows():
            s = int(e["season"])
            w = int(e["week"])
            elim_obs = set(_parse_list(e["elim_obs_list"]))
            m = int(e["m_elim"])
            if require_posterior and ("has_posterior" in e) and (not bool(e["has_posterior"])):
                continue

            g = event_long[(event_long["season"] == s) & (event_long["week"] == w)].copy()
            if g.empty or m <= 0:
                continue

            k += 1
            pred_set = set(g.sort_values("c_hat", ascending=True).head(m)["celebrity_name"].tolist())
            obs_cum |= elim_obs
            pred_cum |= pred_set

            n_prime = len(pred_cum & obs_cum)
            n = len(obs_cum)
            if n == 0:
                term = 0.0
            else:
                term = (1.0 / k) * (n_prime / n)
            terms.append(term)

        if k == 0:
            continue
        H_k = sum(1.0 / i for i in range(1, k + 1))
        S_s = sum(terms) / H_k if H_k > 0 else np.nan
        rows.append({"season": int(season), "K_s": int(k), "S_s": float(S_s)})

    return pd.DataFrame(rows)


def compute_percentile_q(event_long: pd.DataFrame) -> pd.DataFrame:
    g = event_long.copy()
    g["q"] = (g["risk_rank"] - 1.0) / np.maximum(g["alive_n"] - 1.0, 1.0)
    elim = g[g["is_elim_obs"] == True].copy()
    return elim


def build_heatmap_q(event_long: pd.DataFrame) -> pd.DataFrame:
    q_df = compute_percentile_q(event_long)
    q_event = (q_df.groupby(["season", "week"], as_index=False)
               .agg(q_mean=("q", "mean")))
    return q_event


def plot_panel_2x2(acc_df: pd.DataFrame, S_df: pd.DataFrame, q_df: pd.DataFrame,
                   q_event: pd.DataFrame, out_path: str, label: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Season-by-season Top-1 accuracy line
    ax = axes[0, 0]
    if not acc_df.empty:
        acc_season = acc_df.groupby("season", as_index=False).agg(accuracy=("accuracy", "mean"))
        ax.plot(acc_season["season"], acc_season["accuracy"], marker="o", label=label)
    ax.set_title("(a) Top-1 Accuracy by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()

    # (b) Cumulative consistency S_s
    ax = axes[0, 1]
    if not S_df.empty:
        ax.plot(S_df["season"], S_df["S_s"], marker="o", linestyle="-")
    ax.set_title("(b) Cumulative Consistency $S_s$")
    ax.set_xlabel("Season")
    ax.set_ylabel("$S_s$")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # (c) ECDF of percentile q
    ax = axes[1, 0]
    if not q_df.empty:
        x = np.sort(q_df["q"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=label)
    ax.set_title("(c) Eliminated Percentile $q$ (ECDF)")
    ax.set_xlabel("$q$")
    ax.set_ylabel("ECDF")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()

    # (d) Season × event heatmap of q
    ax = axes[1, 1]
    if not q_event.empty:
        pivot = q_event.pivot(index="season", columns="week", values="q_mean")
        im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
        ax.set_title("(d) Season × Event Heatmap (mean $q$)")
        ax.set_xlabel("Week")
        ax.set_ylabel("Season")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.astype(int))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(int))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("mean $q$")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute metrics and generate 2x2 evaluation panel.")
    parser.add_argument("--event-table", type=str, default="event_table.csv")
    parser.add_argument("--event-long", type=str, default="event_long.csv")
    parser.add_argument("--out-panel", type=str, default="fig_eval_panel.png")
    parser.add_argument("--label", type=str, default="model")
    parser.add_argument("--season-min", type=int, default=None)
    parser.add_argument("--season-max", type=int, default=None)
    parser.add_argument("--require-posterior", action="store_true")
    args = parser.parse_args()

    event_table = pd.read_csv(args.event_table)
    event_long = pd.read_csv(args.event_long)

    if args.season_min is not None:
        event_table = event_table[event_table["season"] >= args.season_min].copy()
        event_long = event_long[event_long["season"] >= args.season_min].copy()
    if args.season_max is not None:
        event_table = event_table[event_table["season"] <= args.season_max].copy()
        event_long = event_long[event_long["season"] <= args.season_max].copy()

    acc_df = compute_top1_accuracy(event_long, event_table)
    S_df = compute_cum_consistency(event_long, event_table, require_posterior=args.require_posterior)
    q_df = compute_percentile_q(event_long)
    q_event = build_heatmap_q(event_long)

    plot_panel_2x2(acc_df, S_df, q_df, q_event, args.out_panel, args.label)

    # Output summary metrics table
    catastrophic = float(np.mean(q_df["q"] > 0.8)) if not q_df.empty else np.nan
    summary = {
        "top1_accuracy": float(acc_df["accuracy"].mean()) if not acc_df.empty else np.nan,
        "S_bar": float(S_df["S_s"].mean()) if not S_df.empty else np.nan,
        "q_mean": float(q_df["q"].mean()) if not q_df.empty else np.nan,
        "q_median": float(q_df["q"].median()) if not q_df.empty else np.nan,
        "catastrophic_rate_q_gt_0.8": catastrophic,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("eval_summary_metrics.csv", index=False)
    print("Wrote eval_summary_metrics.csv")


if __name__ == "__main__":
    main()
