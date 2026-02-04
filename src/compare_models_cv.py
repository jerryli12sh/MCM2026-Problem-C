import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import model
import xgb_baseline


def build_panel_from_tables() -> pd.DataFrame:
    df_elim_events, df_roster, df_weekly, df_long_judge, df_clean = xgb_baseline.load_tables()
    elim_long = xgb_baseline.build_elim_long(df_elim_events)
    base = xgb_baseline.build_base(df_roster, elim_long, df_clean)
    judge_percent = xgb_baseline.build_judge_percent(df_weekly, base)
    judge_rank_share = xgb_baseline.build_judge_rank_share(df_long_judge, base)
    panel = xgb_baseline.build_panel(base, judge_percent, judge_rank_share)
    return panel


def week_accuracy_from_posterior(post_df: pd.DataFrame) -> int:
    if post_df is None or post_df["elim_this_week_end"].sum() != 1:
        return None
    post_df = post_df.copy()
    post_df["C_hat"] = post_df["j_metric"] + post_df["p_mean"]
    pred_pos = int(np.argmin(post_df["C_hat"].to_numpy()))
    actual_pos = int(np.where(post_df["elim_this_week_end"].to_numpy())[0][0])
    return 1 if pred_pos == actual_pos else 0


def evaluate_model_inseason(
    model_module,
    panel: pd.DataFrame,
    *,
    seed: int,
    kappa: float,
    tau_like: float,
    B: int,
) -> pd.DataFrame:
    rows = []
    seasons = panel["season"].dropna().astype(int).unique().tolist()
    for s in sorted(seasons):
        panel_s = panel[panel["season"] == s].copy()
        _, pooled_fit, _ = model_module.train_pooled_model(panel_s, seed=seed)
        weeks_s, _, _ = xgb_baseline.build_train_weeks(panel_s)
        for _, row in weeks_s.iterrows():
            w = int(row["week"])
            post = model_module.posterior_mean_for_week(
                panel_s,
                pooled_fit,
                season=s,
                week=w,
                kappa=kappa,
                tau_like=tau_like,
                B=B,
                seed=seed + s * 100 + w,
            )
            acc = week_accuracy_from_posterior(post)
            if acc is None:
                continue
            rows.append({"season": s, "week": w, "accuracy": acc})
    return pd.DataFrame(rows)


def run_inseason(
    seed: int,
    kappa: float,
    tau_like: float,
    B: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    panel = build_panel_from_tables()

    acc_model = evaluate_model_inseason(
        model, panel, seed=seed, kappa=kappa, tau_like=tau_like, B=B
    )
    acc_model["model"] = "torch_model"

    acc_xgb = evaluate_model_inseason(
        xgb_baseline, panel, seed=seed, kappa=kappa, tau_like=tau_like, B=B
    )
    acc_xgb["model"] = "xgboost_baseline"

    acc_by_week = pd.concat([acc_model, acc_xgb], ignore_index=True)
    acc_summary = (
        acc_by_week.groupby(["model", "season"])
        .agg(accuracy=("accuracy", "mean"))
        .reset_index()
    )
    return acc_by_week, acc_summary


def plot_accuracy_bar(acc_summary: pd.DataFrame, out_path: str) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    try:
        ax = sns.barplot(data=acc_summary, x="model", y="accuracy", errorbar="sd")
    except TypeError:
        ax = sns.barplot(data=acc_summary, x="model", y="accuracy", ci="sd")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("In-Season Accuracy (Posterior-Corrected)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--tau_like", type=float, default=0.15)
    parser.add_argument("--B", type=int, default=1200)
    parser.add_argument("--out_csv", type=str, default="cv_accuracy_by_week.csv")
    parser.add_argument("--out_summary", type=str, default="cv_accuracy_summary.csv")
    parser.add_argument("--out_fig", type=str, default="model_accuracy_bar.png")
    args = parser.parse_args()

    acc_by_week, acc_summary = run_inseason(
        seed=args.seed, kappa=args.kappa, tau_like=args.tau_like, B=args.B
    )
    acc_by_week.to_csv(args.out_csv, index=False)
    acc_summary.to_csv(args.out_summary, index=False)
    plot_accuracy_bar(acc_summary, args.out_fig)


if __name__ == "__main__":
    main()
