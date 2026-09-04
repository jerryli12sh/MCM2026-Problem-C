# Repo 完整导览

这是一份面向仓库维护者的中文说明。目标不是重复论文，而是让你看到任何文件时都知道：
它为什么存在、是否需要提交、由什么生成，以及它在整个分析链条中的位置。

## 1. 先建立整体心智模型

这份仓库不是“比赛时所有文件的原样备份”，而是把比赛工作整理成一个可阅读、可验证的
研究型软件项目。它保留三层内容：

1. **思路层**：`review/notes/review_all.md`。它从题意、数据、模型、评价、机制比较到新规
   设计，给出完整的中文推理链。
2. **实现层**：`src/`、`scripts/`、`configs/`、`tests/`。它把思路拆成模块、命令、配置和
   自动测试。
3. **证据层**：`docs/`、`manifests/`、`evidence/`。它说明结果、偏差、决策、输入来源和
   代表性图表。

可以把整个项目读成下面这条流水线：

```text
题面 / 原始数据 / 原论文 / 旧实现 / review_all.md
                    |
                    v
          统一预处理：alive set、结构性 0、淘汰事件
                    |
                    v
     第一问：潜在观众票份额（Track P 与 Track R）
           /                 |                  \
          v                  v                   v
 第二问：赛制回放    第三问：特征与舞伴影响    第四问：新规模拟
           \                 |                  /
            +------------ 敏感性分析 ---------+
                              |
                              v
                  outputs（可重建，不提交）
                              |
                              v
               manifests + evidence（提交的证据）
```

## 2. 为什么 `review_all.md` 是核心

[`../review/notes/review_all.md`](../review/notes/review_all.md) 不是普通会议记录，而是一份
“从题目到模型”的设计说明。它完成了五件关键事情：

- 把问题识别为**机制反演**，而不是普通监督学习：观众票不可见，只能从淘汰结果反推；
- 先定义每个赛季、每周的 `alive set`，避免把淘汰后的结构性 0 当成真实得分；
- 用 pooled support、softmax 和 Dirichlet 分布描述不可唯一识别的观众票份额；
- 指出原方案重复使用淘汰结果的问题，并给出积分边际似然的修正方向；
- 把估计结果用于赛制回放、特征解释和 Monte Carlo 新规设计。

仓库中的 Track P 对应提交论文/旧实现；Track R 对应这份 review 提出的主要统计修正。
因此，读代码时不要问“哪个文件完整实现了 review”，而要沿下面的映射阅读：

| `review_all.md` 内容 | 主要实现位置 |
|---|---|
| alive set、赛季长度、结构性 0、淘汰事件 | `src/dwts_reproduction/preprocess.py` |
| `q = softmax(Xβ + u)`、softmin、Dirichlet/importance sampling | `problem1/` |
| 避免淘汰结果重复使用的边际似然 | `problem1/track_r.py` |
| rank / percentage / Bottom-2 + save | `problem2/rules.py`、`problem2/replay.py` |
| 年龄、行业、舞伴、surprise/growth | `problem3/` |
| fan compression、judge amplification、momentum bonus | `problem4/` |
| 超参数与结论是否稳定 | `sensitivity/` |

## 3. 顶层目录逐一解释

### `src/dwts_reproduction/`：可复用的核心代码

`src` layout 是现代 Python 项目的常见规范：包代码放在 `src/包名/`，防止测试意外导入
当前目录中的源码，而不是已安装的包。

- `__init__.py`：告诉 Python 这是一个包；也可暴露版本号等最小公共接口。
- `config.py`：把仓库路径与外部只读数据路径集中管理；生产代码不写死某台电脑的路径。
- `hashing.py`：计算 SHA-256，核对外部输入有没有被修改。
- `preprocess.py`：把 421×53 的选手宽表变成可建模的长表、周表、名单表和淘汰事件表。
- `run_manifest.py`：记录每次运行的命令、Git 提交、环境、输入与输出，使结果可追溯。
- `smoke.py`：做低成本的冒烟检查，快速判断数据与基本流程是否可用。

子包含义：

- `problem1/`
  - `panel.py`：建立选手—赛季—周面板和训练事件；
  - `softmin.py`：数值稳定的 softmax/softmin、Dirichlet 密度、加权分位数和 Adam；
  - `track_p.py`：论文忠实路线；
  - `track_r.py`：review 修正路线；
  - `evaluate.py`：Top-1、PCP、赛季路径一致性和后验摘要；
  - `baselines.py`：XGBoost 等基准；
  - `structural.py`、`figures.py`：结构效应与图表。
- `problem2/`
  - `rules.py`：四种淘汰机制的纯函数；
  - `replay.py`：把同一批后验票数放进不同赛制回放；
  - `mechanism_phase.py`：比较观众影响与技术一致性。
- `problem3/`
  - `regression.py`：明星属性的平行回归；
  - `partner.py`：职业舞伴影响；
  - `surprise.py`：超预期表现与粉丝增长；
  - `figures.py`：第三问图表。
- `problem4/`
  - `v1.py`：忠实复现旧的新规模拟；
  - `v2.py`：整理后的机制设计；
  - `features.py`：模拟输入特征；
  - `metrics.py`：公平性、存活率、技术冲击等指标；
  - `cases.py`：Jerry Rice 等争议案例；
  - `claims.py`：把文字结论转成可自动检查的断言；
  - `figures.py`：第四问图表。
- `sensitivity/`：改变 `κ`、`τ`、正则强度、评委信号和留一赛季设定，检查结论稳定性。
- `release/compare.py`：把新生成的结果与 20 个注册基准逐项比较。

### `scripts/`：可以直接运行的入口

核心逻辑放在 `src/`，`scripts/` 只负责读取参数、调用核心函数、保存文件。这种“薄脚本、
厚模块”的结构便于测试与复用。

- `problem1_run.py` 到 `problem4_run.py`：运行各问题；
- `sensitivity_run.py`：运行敏感性分析；
- `plot_*.py`：只从已保存表格绘图，避免图表偷偷依赖内存状态；
- `run_release.py`：按固定顺序执行 19 个阶段，并检查 20 个基准；
- `hash_inputs.py`、`inventory_sources.py`：输入哈希与旧材料清单；
- `build_baseline.py`、`build_conflict_matrix.py`、`build_traceability.py`：生成可审计清单；
- `smoke_test.py`：快速数据检查入口。

### `configs/`：可读配置

- `paths.yaml`：默认认为只读源材料在仓库父目录；也可用环境变量 `DWTS_SOURCE_ROOT`
  覆盖，不必修改 Git 中的文件。
- `phase0.yaml`：最初的审计/清点阶段使用的随机种子与清单位置。
- `problem1.yaml`：Track P 的正式超参数，例如 `κ=10`、两个 softmin temperature、
  Adam 学习率、迭代步数和抽样数。

把配置与代码分开有两个好处：参数变化可以被 Git 清楚记录；同一代码可以运行不同设定。

### `tests/`：自动化证据

测试文件按模块命名，例如 `test_preprocess.py` 对应 `preprocess.py`。测试分为：

- 单元测试：给一个很小的手算例子，检查 rank、percentage、Bottom-2 等规则；
- 不变量测试：概率和为 1、alive set 合法、淘汰者属于候选集合；
- 数值测试：梯度、积分、随机抽样和容差；
- 集成测试：用外部数据跑一段完整流程；
- 回归测试：历史上已确认的数字不能无意漂移。

公开 GitHub CI 只跑不需要原始数据的测试。完整测试需要本地只读 source bundle。

### `manifests/`：机器可读的审计账本

- `baseline.csv`：20 个基准结果、目标值与容差；
- `conflict_matrix.csv`：论文、review 与旧代码之间的 7 个冲突；
- `traceability_paper.csv`：论文 96 条要求分别对应哪段实现与测试；
- `traceability_review.csv`：review 40 条要求的实现映射；
- `legacy_inventory.csv`：174 个外部原始/旧文件的角色清单；
- `input_manifest.sha256`：上述外部输入的 SHA-256 指纹。

这些 CSV 不是“分析数据”，而是审计元数据。它们适合程序检查，也能证明工作不是只靠
README 自述。

### `evidence/`：给人看的结果快照

`evidence/figures/` 保留 10 张有代表性的图，而不是把全部 79 张图都提交。这样面试官
克隆仓库后能立刻看到结果，仓库又不会被大量可重建图片淹没。`evidence/README.md`
解释每张图表示什么、来自哪个 release，以及如何核对哈希。

### `outputs/`：运行时产物

这里会出现 CSV、JSON、NPZ、PNG、日志以及两个约 50–100 MB 的压缩模拟明细。它们都能
由脚本重建，所以 `.gitignore` 排除了 `outputs/**`，只保留 `.gitkeep` 让空目录存在。

判断规则：

- 想回答“读者第一次打开仓库是否必须看到？”——放 `evidence/`；
- 想回答“运行代码是否会重新得到？”——放 `outputs/`，不要提交。

### `docs/`：面向人的方法与复盘

- `STATUS.md`：当前数字与结论的唯一权威汇总；
- `METHOD_SPEC.md`：符号、Track P/R 和方法定义；
- `DECISIONS.md`：24 个关键歧义如何处理，是最能体现研究判断的文档之一；
- `CONFLICT_MATRIX.md`：7 个冲突的简表；
- `BASELINE_PAPER_OUTPUTS.md`：论文/旧实现的基准结果；
- `DATA_DICTIONARY.md`：输入表和派生表各字段的含义；
- `DEVELOPMENT.md`：从清点到发布的开发阶段复盘；
- `VERIFICATION.md`：当前发布前检查结果；
- `CI.md`：为什么 GitHub 只能跑 source-free gate；
- `ENVIRONMENT.md`：Python 环境与依赖；
- `RUN_MANIFEST.md`：一次可追溯运行必须记录什么；
- `TRACEABILITY_*.md`、`LEGACY_INVENTORY.md`：由脚本生成的人类可读清单；
- `requirements-lock.txt`：当时实际验证环境的第三方依赖版本快照。

### Git 与项目配置文件

- `pyproject.toml`：Python 包名称、最低版本、依赖、pytest/ruff/mypy 配置；
- `Makefile`：把长命令包装成 `make check`、`make verify-data`、`make release`；
- `.github/workflows/ci.yml`：GitHub Actions 自动检查；
- `.gitignore`：告诉 Git 不要追踪缓存、虚拟环境、数据与输出；
- `.gitattributes`：禁止 Git 改写 `review_all.md` 的换行符，保证镜像字节不变。

## 4. 常见陌生文件类型

| 后缀/名称 | 用途 | 是否应提交 |
|---|---|---|
| `.md` | Markdown 文档；GitHub 可直接渲染标题、表格、公式 | 核心说明应提交 |
| `.py` | Python 源代码或命令入口 | 提交 |
| `.yaml` / `.yml` | 层级化配置；前者用于模型，后者常用于 GitHub Actions | 提交 |
| `.toml` | Python 项目与工具配置 | 提交 |
| `requirements-lock.txt` | 精确依赖版本列表，供 CI/复现环境安装 | 提交 |
| `Makefile` | 一组便捷命令；没有扩展名 | 提交 |
| `.csv` | 逗号分隔表格；可用表格软件打开，也便于程序读取 | 小型清单提交；大结果不提交 |
| `.json` | 结构化键值记录；本项目用于运行/图片 provenance | 代表性证据可提交，生成结果不提交 |
| `.sha256` | 文件内容指纹清单；内容改一个字节，哈希通常就变 | 提交 |
| `.npz` | NumPy 多数组压缩包；用于模型参数和抽样数组 | 生成物，不提交 |
| `.png` | 无损位图 | 只提交精选 evidence |
| `.csv.gz` | gzip 压缩的 CSV；适合大规模模拟明细 | 生成物，不提交 |
| `.log` | 运行日志 | 通常不提交 |
| `.gitkeep` | Git 不追踪空目录，用这个空文件保留目录结构 | 提交 |
| `__init__.py` | Python 包标记 | 提交 |
| `__pycache__/`、`.pyc` | Python 自动生成的字节码缓存 | 不提交，可随时删除 |
| `.venv/` | 本机 Python 虚拟环境及第三方包 | 不提交，可重建 |
| `.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/` | 测试、类型检查、格式工具缓存 | 不提交，可删除 |
| `*.egg-info/` | editable install 自动生成的包元数据 | 不提交，可重建 |
| `.DS_Store` | macOS Finder 自动生成的目录显示信息 | 不提交，可删除 |

## 5. 推荐阅读顺序

如果只用 15 分钟：

1. `README.md`；
2. `review/notes/review_all.md` 的开头、核心模型、重复使用淘汰结果、新机制四部分；
3. `docs/STATUS.md`；
4. `evidence/README.md` 和几张图。

如果要准备面试：

1. 先能口述“为什么这是机制反演而非普通预测”；
2. 解释 alive set 与结构性 0 为什么决定整个后续分析是否正确；
3. 解释 `q`、`p`、Dirichlet、softmin 与 importance sampling 各自解决什么问题；
4. 主动指出 Track P 的 double-use，并说明为何保留 Track P 同时新增 Track R；
5. 用一个争议选手说明反事实赛制回放；
6. 讲清楚一个未复现结论，以及你为什么没有隐藏它；
7. 最后再讲测试、manifest、CI 如何把一次性比赛代码变成可信工程。

## 6. 哪些内容不在公开仓库

以下材料在外部只读 source bundle 中，不会随 GitHub 仓库发布：官方数据、题面 PDF、
提交论文 LaTeX、比赛时期旧 notebook/脚本和完整旧输出。原因不是它们没用，而是：

- 数据与题面/论文的再分发授权尚未在此仓库中声明；
- 原始 notebook、缓存和全部图片会掩盖整理后的主线；
- 外部材料已通过 174 条清单和 SHA-256 记录来源，仍可在本地核验。

公开仓库因此是“可审计实现与精选证据”，不是整个比赛硬盘的镜像。

## 7. 维护规则

- 修改方法或数字时，同时更新 `STATUS.md`、对应 decision、测试和 baseline；
- 新图先写入 `outputs/`，确认值得长期展示后再人工加入 `evidence/`；
- 不提交 `.venv`、缓存、`outputs`、原始数据、令牌或绝对机器路径；
- Track P 与 Track R 的输出始终带标签；
- 不把 fan vote posterior 称为真实投票；
- 发布前至少运行 `make check`；有 source bundle 时再运行 `make verify-data`。
