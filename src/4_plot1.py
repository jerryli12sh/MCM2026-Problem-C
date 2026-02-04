import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) load
df = pd.read_csv("/Users/jerryli/Desktop/MCM/sim_results/sim_summary.csv")
df = df.sort_values(["scheme", "archetype", "week"]).copy()

# 2) cumulative survival: S(t)=prod alive_rate
#    用 log 累加更稳健
eps = 1e-12
df["cum_alive_rate"] = (
    df.groupby(["scheme", "archetype"])["alive_rate"]
      .apply(lambda s: np.exp(np.log(s.clip(eps, 1.0)).cumsum()))
      .reset_index(level=[0,1], drop=True)
)

# （可选）加 week=0 的起点，让生存曲线从 1 开始更直观
base = df[["scheme","archetype"]].drop_duplicates()
base["week"] = 0
base["alive_rate"] = np.nan
base["elim_rate"] = np.nan
base["avg_rank"] = np.nan
base["n"] = np.nan
base["cum_alive_rate"] = 1.0
df2 = pd.concat([base, df], ignore_index=True).sort_values(["scheme","archetype","week"])

sns.set_theme(style="whitegrid", font_scale=1.0)

archetypes = sorted(df2["archetype"].unique())
metrics = [
    ("cum_alive_rate", "Cumulative survival  $S(t)$", (0, 1)),
    ("avg_rank",       "Average rank (lower is better)", None),
    ("elim_rate",      "Weekly elimination rate", (0, 1)),
]

fig, axes = plt.subplots(
    nrows=len(metrics), ncols=len(archetypes),
    figsize=(4.6*len(archetypes), 3.1*len(metrics)),
    sharex=True
)

# 统一 scheme 颜色顺序（避免每个子图颜色乱跳）
scheme_order = sorted(df2["scheme"].unique())

handles, labels = None, None
for j, arch in enumerate(archetypes):
    sub_arch = df2[df2["archetype"] == arch]
    for i, (mcol, ylab, ylim) in enumerate(metrics):
        ax = axes[i, j] if len(archetypes) > 1 else axes[i]
        sns.lineplot(
            data=sub_arch, x="week", y=mcol,
            hue="scheme", hue_order=scheme_order,
            marker="o", linewidth=2.2, alpha=0.9,
            ax=ax
        )
        ax.set_title(arch if i == 0 else "")
        ax.set_ylabel(ylab)
        ax.set_xlabel("Week" if i == len(metrics)-1 else "")

        if ylim is not None:
            ax.set_ylim(*ylim)

        # avg_rank：反转 y 轴，让“更好(更小)”在上面
        if mcol == "avg_rank":
            ax.invert_yaxis()

        # 只保留一个 legend（最后统一放到图外）
        if (i, j) != (0, 0):
            ax.get_legend().remove()
        else:
            leg = ax.get_legend()
            handles, labels = ax.get_legend_handles_labels()

# 全局 legend
fig.legend(handles, labels, title="Scheme", loc="upper center", ncol=len(scheme_order))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("fig_scheme_archetype_week_grid.png", dpi=300, bbox_inches="tight")
plt.show()
