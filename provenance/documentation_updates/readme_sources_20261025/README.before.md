# Cascade reservoir paper benchmark

本仓库整合论文双库正式实验：DDP200、SDP400、ISO-RF500、动作条件Double DQN5000、V16几何条件连续PPO6000。只有TRAIN与EVAL，无dry/wet正式情景。

DDP、SDP、ISO、DQN及物理源码采用V11；连续PPO采用V16；正式图的数据读取和统计采用经过审查的V6版本。原始目录和结果均未覆盖。

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
