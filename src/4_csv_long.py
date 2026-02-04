import numpy as np
import pandas as pd

df = pd.read_csv("/Users/jerryli/Desktop/MCM/sim_results/sim_summary.csv").sort_values(["scheme","archetype","week"]).copy()
eps = 1e-12
df["cum_alive_rate"] = (
    df.groupby(["scheme","archetype"])["alive_rate"]
      .apply(lambda s: np.exp(np.log(s.clip(eps,1.0)).cumsum()))
      .reset_index(level=[0,1], drop=True)
)

# long format
long = df.melt(
    id_vars=["scheme","week","archetype"],
    value_vars=["cum_alive_rate","avg_rank","elim_rate"],
    var_name="metric",
    value_name="value"
)

# 可选：把 metric 改成更友好的标签
metric_map = {
    "cum_alive_rate": "Cumulative survival",
    "avg_rank": "Average rank (lower better)",
    "elim_rate": "Weekly elimination rate"
}
long["metric"] = long["metric"].map(metric_map)

long.to_csv("/Users/jerryli/Desktop/MCM/sim_results/sim_summary_long_for_flourish.csv", index=False)
print(long.head())
