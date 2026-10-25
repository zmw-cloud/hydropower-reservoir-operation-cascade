# Cascade reservoir paper benchmark

本仓库整合论文双库正式实验：DDP200、SDP400、ISO-RF500、动作条件Double DQN5000、V16几何条件连续PPO6000。只有TRAIN与EVAL，无dry/wet正式情景。

DDP、SDP、ISO、DQN及物理源码采用V11；连续PPO采用V16；正式图的数据读取和统计采用经过审查的V6版本。原始目录和结果均未覆盖。

## Code and Data Sources（代码与数据来源）

### 代码与既有实验结果

- DDP、SDP、ISO、动作条件 Double DQN 及公共物理模型采用 `Platform-for-cascade-hydropower-reservoir-operation-11.0.0`；连续动作 PPO 采用 `Platform-for-cascade-hydropower-reservoir-operation-16.0.0` 的 `continuous_ppo_*` 实现。
- 正式模型、配置和评价结果来自 `Platform-for-cascade-hydropower-reservoir-operation-6.0.0` 中保存的实际结果包。比较统计与绘图依赖采用经审查的 V6/V16 对应实现；绘图代码的存放版本不代表模型的训练源码版本。
- 径流统计图复用了 V16 原绘图入口使用的 `Figure2_Jupyter_bundle`。该资料包原存放于单库 3.0.0 目录，本仓库已将其脚本和三个输入文件本地化到 `hydrology/figures/`；没有以该旧目录中的调度算法替代正式算法。
- 逐文件来源和 SHA-256 见 [来源清单](provenance/source_manifest.json)，路径及入口适配见 [整合变更记录](provenance/integration_changes.json)。代码许可证见 [LICENSE](LICENSE)。

### 工程参数与曲线

- 物理参数见 [xiaowan_nuozhadu_v1.yaml](configs/xiaowan_nuozhadu_v1.yaml)。正式曲线分别位于 [Reservoir X](data/xiaowan_nuozhadu/reservoir_1/) 和 [Reservoir N](data/xiaowan_nuozhadu/reservoir_2/)。
- 曲线与研究留存的 `4-水位-库容曲线.xlsx`、`7-尾水位泄流量曲线.xlsx` 中 `1小湾`、`4糯扎渡` 工作表对应。正式库容 CSV 分别包含 75/48 个运行区间控制点，单位为 10⁹ m³；原工程表的万 m³ 数值除以 100000 后转换为该单位。尾水位原始 CSV 分别包含 41/63 点，流量单位为 m³/s，水位单位为 m。
- Reservoir X 尾水位拟合另追加配置中的 `(15666 m³/s, 1012.71 m)`、`(20683 m³/s, 1016.64 m)` 两个锚点，Reservoir N 未另加锚点。现有归档尚未完整记录部分工程资料的原发布单位、报告编号及这两个锚点的上游工程出处。

### 历史与合成径流

- 生成器历史输入见 [interval_inflow.xlsx](hydrology/generator/input/interval_inflow.xlsx)，覆盖 1978—2008 年。包含 Reservoir X 天然入流、Reservoir N 天然入流，以及二者之差形成的区间天然入流。原始观测资料的具体提供单位和原始资料名称，现有归档尚未完整注明。
- 合成序列采用 `Target-Synthetic-Streamflow-Generator-monthly-dual-v1.0`，源码位于 [hydrology/generator](hydrology/generator/)，配置见 [config.json](hydrology/generator/config.json)。TRAIN 种子为 `2026090301`，EVAL 为 `2026090302`；每组 400 年、4800 个月，年份标签为 2009—2408。
- 正式联合输入见 [joint 目录](data/xiaowan_nuozhadu/inflow/target_synthetic_monthly_dual_v1_20260903/joint/)，包含上游天然入流与下游区间入流，单位为 m³/s。它们是合成序列，不是 400 年实测历史记录；正式实验只有 TRAIN/EVAL，不采用 dry/wet 情景。
- 生成器、模型和图件的文件来源分别记录于来源清单。部分工程资料和原始历史径流的上游出处与独立使用条款，现有归档尚未完整记录。

详细复现步骤及实验协议边界见 [REPRODUCTION.md](docs/REPRODUCTION.md)。当前文件核验使用 [发布清单](provenance/release_manifest.json)；本次 README 更新前的文档和清单原件保存在 [文档变更归档](provenance/documentation_updates/readme_sources_20261025/change.json)。

## 核验与安装

~~~text
python -m pip install -r environments/requirements-plots.txt
python -B scripts/audit_repository.py
python -B scripts/audit_repository.py --plots
python -B scripts/run_workflow.py inspect
python -B -m unittest discover -s tests -p test_integration.py -v
~~~

--plots只检查此前要求的图的数据和统计输入，不绘图、不执行策略。

| 方法 | λ | 年均发电量 TWh/yr | X保证率 % | N保证率 % |
|---|---:|---:|---:|---:|
| DDP | 200 | 45.810842 | 90.750000 | 82.270833 |
| SDP | 400 | 44.425634 | 84.875000 | 79.979167 |
| ISO-RF | 500 | 44.407936 | 79.729167 | 63.645833 |
| DQN | 5000 | 44.156822 | 79.020833 | 76.402778 |
| PPO | 6000 | 43.614492 | 81.305556 | 78.020833 |

DQN/PPO为种子1、42、100均值。逐站均值的最小值、逐种子最小值的均值、两站同时达标率、总功率达标率分别保存定义，不混用。

## 正式绘图

只纳入此前要求重画/新增的图：历史与合成径流、双库权衡、功率持续、水位、弃水、DQN–PPO调整、小提琴和性能迁移。单库曲线绘图不纳入。

~~~text
python -B scripts/plot_final_figures.py --check-only
python -B scripts/plot_final_figures.py --figures all
~~~

第二条才生成图片，每次写入新的outputs目录。需要Times New Roman，可使用系统字体或--font-dir指定本地字体目录；不复制系统商业字体到仓库。

当前正文图号登记在configs/paper/figure_registry.json，兼容原绘图函数的旧参数编号。年度小提琴/迁移图需要的单库正式结果已在本仓库独立保留，不依赖另一个仓库。

## 独立评价

~~~text
python -B scripts/run_workflow.py prepare --algorithm ppo --stage evaluate
python -B scripts/run_workflow.py execute outputs/runs/上一步生成的目录名
~~~

prepare只复制输入和固定模型；execute才运行。可用方法为sdp、iso、dqn、ppo，DDP使用solve。原算法在新的工作副本中执行，结果不会写回artifacts或reference_results。

正式算法需另行安装项目依赖：python -m pip install -e ".[all]"。冻结RF评价须匹配保存的scikit-learn1.9.1，入口在模型执行前检查版本。DQN/PPO使用各自原网络与检查点，不通过RF环境要求推断它们的历史完整环境。

## 数据与历史记录

- data：两站工程曲线和正式TRAIN/EVAL联合径流。
- artifacts：固定策略、权重、模型元数据及教师。
- reference_results：正式评价及绘图所需验证扫描。
- hydrology：已使用的生成器源码、仅TRAIN/EVAL配置及自包含径流图输入；本次未重新生成径流。
- provenance：源码档案、原配置、来源与整合差异。
- docs：复现步骤、SI依据和明确局限。

ISO仍保留既有400年教师后切320/80的实验，不重算成另一个实验。1223/812m验证初值合法，但来自全期教师；因此不声称教师信息严格隔离。独立EVAL数据与TRAIN是分离的。

双库RF原模型记录sklearn1.9.1，完整DQN/PPO训练环境锁文件尚不完整，详见environments/historical_versions.json。该事实不通过猜测补齐。
