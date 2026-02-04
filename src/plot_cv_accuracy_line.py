 import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="cv_accuracy_summary.csv")
    parser.add_argument("--out", type=str, default="model_accuracy_line.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["season"] = df["season"].astype(int)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)
    plt.figure(figsize=(10, 5.2))

    palette = {
        "torch_model": "#2B6CB0",
        "xgboost_baseline": "#D97706",
    }

    ax = sns.lineplot(
        data=df,
        x="season",
        y="accuracy",
        hue="model",
        marker="o",
        linewidth=2.6,
        markersize=6.5,
        palette=palette,
    )

    ax.set_xlabel("Season")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("In-Season Accuracy by Season", pad=12, weight="semibold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.margins(x=0.02, y=0.02)

    ax.legend(title="Model", frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
