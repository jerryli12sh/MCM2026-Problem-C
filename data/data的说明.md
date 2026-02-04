## 1) 你现在有哪些表？它们的结构与作用

### A. `df_clean`（选手静态主表 + 宽表分数）

- **粒度**：一行 = 一个 *season* 的一个 *celebrity*

- **核心列（静态）**：
  
  - 身份信息：`celebrity_name, ballroom_partner, celebrity_industry, ...`
  
  - 年龄：`celebrity_age_during_season`
  
  - 赛果：`results, placement`
  
  - 解析出的时间逻辑：`elim_week_result, is_withdrew, is_place`
  
  - 你推断出的结构变量：`season_max_week, active_until, last_week_positive`

- **核心列（分数宽表）**：`week1_judge1_score ... week11_judge4_score`

- **用途**：
  
  1. 作为“静态信息维表”（industry/age 等）
  
  2. 作为所有长表/周表的来源
  
  3. 这里的分数已把“淘汰后的 0 填充”清掉（转为 NaN），避免污染统计

---

### B. `df_long_judge`（评委级长表：最细粒度）

- **粒度**：一行 = (season, celebrity, week, judge)

- **核心列**：
  
  - 索引：`season, celebrity_name, week, judge`
  
  - 打分：`judge_score`
  
  - 结构标记：`is_show_week, eligible`
  
  - 静态属性：行业/年龄/赛果等也跟着带过来（便于 groupby 分析）

- **用途**：
  
  1. 分析“评委数量变化、缺评委、评委偏好/方差”
  
  2. 做任何“评分分布/稳定性/离群评委”之类的 EDA
  
  3. 后续若要建“评委打分模型”（误差、方差分解），它是母表

---

### C. `df_weekly`（周级汇总表：后续建模主表）

- **粒度**：一行 = (season, celebrity, week)

- **核心列**：
  
  - 周内评委汇总：`total_judge_score, mean_judge_score, n_judges_scored`
  
  - 是否真的表演：`performed`（当周有正分才算）
  
  - 是否仍在比赛：`eligible`
  
  - 周内相对位置：`judge_rank`（越小越强）, `judge_percent`（当周总分占比）
  
  - 静态属性：行业/年龄/赛果等

- **用途**：
  
  1. 这是你后续“淘汰/存活/投票机制”的建模底座
  
  2. `judge_rank` / `judge_percent` 正是题面两套规则对应的关键输入
  
  3. 可以直接做“周内横截面”分析：谁分高、谁被淘汰、是否出现“爆冷”

---

### D. `df_roster`（每周在场名单表）

- **粒度**：一行 = (season, celebrity, week)

- **核心列**：`eligible`（这周是否仍是参赛者）

- **用途**：
  
  1. 明确每周“投票集合/竞争集合”是谁
  
  2. 任何概率模型都必须先定义样本空间：这张表就是样本空间

---

### E. `df_elim_events`（淘汰事件表）

- **粒度**：一行 = (season, week_end)

- **核心列**：
  
  - `eliminated`：一个 list，表示这一周末淘汰了谁（支持双淘汰/多淘汰）
  
  - `is_final_week_end`：是否只是“决赛结束”（不是淘汰机制）

- **用途**：
  
  1. 直接给你“监督信号”：哪周发生淘汰、淘汰了哪些人
  
  2. 很适合做事件研究：被淘汰者的 judge_rank 分布、爆冷频率等
  
  3. 后续建模可以把它 merge 回 `df_weekly` 得到标签
