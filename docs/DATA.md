# 数据与文件说明

## 数据来源与入口

| 文件 | 粒度 | 来源与用途 |
|---|---|---|
| `data/2026_MCM_Problem_C_Data.csv` | 选手—赛季，421 × 53 | COMAP 赛题随附数据，推断模型与规则回放的原始入口 |
| `data/intermediate/celebrity_signals.csv` | 选手—赛季，392 行 | 原分析的评委/观众标准化信号；保留第 1、6、11 周及决赛阶段特征 |
| `data/intermediate/data_3.csv` | 选手—赛季，392 行 | 上述信号与官方最终名次连接后的回归入口，由预处理重建 |
| `data/intermediate/contestant_archetypes.csv` | 选手—赛季，421 行 | 原分析保存的选手类型，供模拟结果按人气型/技术型/均衡型汇总 |
| `data/reference/*.csv` | 小型结果表 | 原分析保存的 Bottom-2 和模拟参考值，用于数值回归检查 |

官方来源：[题面 PDF](https://www.contest.comap.com/undergraduate/contests/mcm/contests/2026/problems/2026_MCM_Problem_C.pdf)、[原始 CSV](https://www.contest.comap.com/undergraduate/contests/mcm/contests/2026/problems/2026_MCM_Problem_C_Data.csv)。仓库保留项目使用的原始版本。

完整流程会从原始 CSV 重建周表、拟合 P/R、重算规则回放、模拟和敏感性结果。属性回归使用 `celebrity_signals.csv` 中的原分析信号，再从原始 CSV 映射官方最终名次，生成 `data_3.csv`。模拟分型从 `contestant_archetypes.csv` 读取。`celebrity_signals.csv` 与选手分型是随附的固定输入，运行程序不会覆盖。

## 原始字段

| 字段 | 含义 |
|---|---|
| `celebrity_name` | 选手姓名 |
| `ballroom_partner` | 职业舞伴 |
| `celebrity_industry` | 选手职业类别 |
| `celebrity_homestate` | 美国州籍 |
| `celebrity_homecountry/region` | 国家/地区；程序统一列名中的 `/` 为 `_` |
| `celebrity_age_during_season` | 当季年龄 |
| `season` | 赛季编号 1–34 |
| `results` | 淘汰周、退赛或最终名次的文字记录 |
| `placement` | 最终名次，1 为最好 |
| `week{t}_judge{j}_score` | 第 t 周第 j 位评委评分 |

`N/A` 包括未设置的评委席位及该赛季未播出周；淘汰后的 0 分是结构性填充。预处理根据在赛状态保留两者含义。

## 重建的中间表

运行 `python scripts/preprocess_run.py` 生成：

| 文件 | 主要字段与解释 |
|---|---|
| `df_clean.csv` | 原始静态信息、`elim_week_result`、`is_withdrew`、`season_max_week`、`active_until` |
| `df_weekly.csv` | 周评分总量、均值、有效评委数、是否表演、在赛标志、评委名次与评分占比 |
| `df_roster.csv` | 每周名单以及本周/下周的 `eligible` 状态 |
| `df_elim_events.csv` | 退出周、退出者列表、`m_elim`、决赛周标志 |
| `validation.csv` | 在赛边界、结构性零值、重复记录等数据逻辑检查 |

评委级长表在程序内构建并供模型使用；仓库只保存分析中经常查阅的周表，减少重复数据。

### 属性特征表

`data_3.csv` 包含选手基础属性，以及 `placement_z`、`week1/6/11_judge_score_placement_z`、`week1/6/final_p_score_placement_z`，数值方向统一为“越大越好”。392 行来自原分析剔除退赛者、第 1 季和全明星第 15 季后的样本。

`placement` 直接使用官方最终名次；`placement_z` 先计算相对本季全部参赛者的击败比例 `(N - placement)/(N - 1)`，再在 392 行样本内做总体标准差标准化。评委阶段信号沿用最近一次有效评分，因此退出选手仍可有第 6、11 周的评委特征；这些列表示截至该周可获得的技术信号。观众阶段信号保留原分析的缺失情况，各回归使用对应列的有效样本。

回归模块继续计算舞伴过往成绩、经验与行业分组。历史胜率、前三率的分母均为实际已参赛次数，新舞伴以 0 编码并同时保留历史参赛数。

`contestant_archetypes.csv` 包含 `delta_mean`、`n_weeks` 与 `archetype`：`relative_popular`、`relative_technical`、`balanced`。类型由原分析的观众与评委信号差异划分，在模拟中作为固定分组标签。

## 结果命名

所有表均在 `results/tables/`，图在 `results/figures/`。后缀 `P/R` 对应支持度模型；`P1E` 为基线与排名分析，`P3` 为属性分析，`V1/V2` 为两组模拟器，`SA` 为敏感性分析。

| 文件模式 | 内容 |
|---|---|
| `problem1_panel_{P,R}.csv` | 周面板、在赛标志、评委信号与年代 |
| `problem1_train_weeks_{P,R}.csv` | 218 个单人淘汰周及真实淘汰者 |
| `problem1_posterior_summary_{P,R}.csv` | 支持度均值、区间、PCP、ESS、是否使用周淘汰更新 |
| `problem1_top1_by_week/season_{P,R}.csv` | 逐周/逐季重构结果 |
| `problem1_event_*`、`problem1_cumulative_consistency_*` | 事件级得分和赛季路径指标 |
| `problem1_fit_meta_{P,R}.json` | 系数、选手效应、参数、损失曲线及训练周 |
| `problem1_fit_arrays_{P,R}.npz` | 运行时生成的拟合数组，Git 忽略；供后续规则与模拟重载 |
| `problem2_season_metrics_{P,R}.csv` | 各季规则比较指标 |
| `problem2_case_divergence_{P,R}.csv` | 6 位选手的名次差异与规则翻转 |
| `problem2_b2_metrics_{P,R}.csv` | 末两位概率、拯救反转与路径变化 |
| `problem2_phase_metrics_{P,R}.csv` | 4 种规则的赛季路径比较 |
| `problem3_*_P3.csv/json` | 回归系数、标准误、R²、舞伴效应、前向验证与动态关联 |
| `problem4_sim_summary_{V1,V2}.csv` | 规则 × 周 × 选手类型的模拟汇总 |
| `problem4_case_*_{V1,V2}.csv` | 6 位案例的逐周和整体模拟结果 |
| `problem4_shock_rates_{V1,V2}.csv` | 技术冲击频率 |
| `sensitivity_*_SA.csv` | 四类扰动的汇总及逐周诊断 |

### 重点字段

| 字段 | 解释 |
|---|---|
| `p_mean` | 周支持度后验均值，在同一周名单中加总为 1 |
| `ci_lo_05`, `ci_hi_95` | 支持度 90% 可信区间端点 |
| `pcp_weighted` | 重要性加权的淘汰一致概率 |
| `ess_ratio` | 有效样本数 / 抽样数 |
| `ci_rel_width` | 区间宽度相对于支持度均值的比例 |
| `correct` | 后验均值是否重构当周实际淘汰者 |
| `alive_rate` | 当前模拟周入场记录中，未在该周淘汰的比例 |
| `final_alive_rate` | 案例汇总的最后可见周未淘汰次数 / 该案例模拟次数 |
| `Shock_k` | 评委前 k 名选手被淘汰的频率，分母为各规则产生的淘汰事件总数 |

空单元格表示该对象缺少有效观测或该指标在当前分组未定义。JSON 保存统计摘要及参数；CSV 保存可检查的逐条结果。

## 文件取舍

仓库以原始 CSV、必要中间表、核心模块、结果 CSV/JSON 和 PNG 图为主。临时缓存、编辑器设置、大量重复模拟明细以及数据指纹文件不参与发布。原始 notebooks 与 LaTeX 工程保留在仓库历史中；当前入口采用统一脚本与论文 PDF。
