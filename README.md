# Cascade reservoir paper benchmark

论文月尺度水电调度复现代码，包含 DDP200、SDP400、ISO-RF500、Double DQN5000、连续 PPO6000。正式数据划分为 TRAIN / 独立 EVAL。

仓库保留源码、README、复现说明、实验配置、必要输入数据、依赖声明、许可证和测试源码。训练模型、教师轨迹、验证扫描、评价结果、日志、图件及整合过程备份均由本地运行生成，不随源码提交。

## 安装与输入检查

在仓库根目录使用 Python 3.10+：

```bash
python -m pip install -e ".[all]"
python -B scripts/audit_repository.py
python -B scripts/run_workflow.py inspect
python -B -m unittest discover -s tests -v
```

输入检查只验证正式径流文件及其 SHA-256，不加载模型、不训练、不评价。

## 复现

先生成策略，再评价；运行结果保存在已被 Git 忽略的 `outputs/`。以下以 SDP 为例：

```bash
python -B scripts/run_workflow.py run --algorithm sdp --stage train
python -B scripts/run_workflow.py run --algorithm sdp --stage evaluate
```

完整阶段顺序、其他算法命令和绘图输入配置见 [复现说明](docs/REPRODUCTION.md)。命令附加 `--show-commands` 可只查看具体入口。初次克隆不包含预训练模型，不能跳过训练直接评价。

## 仓库结构

- `src/`：算法与公共物理模型。
- `scripts/`：运行、输入检查及绘图入口。
- `data/`：正式径流、工程曲线及必要配置。
- `configs/`：论文参数、扫描设置与结果路径模板。
- `tests/`：测试源码，不含测试产物。
- `environments/`：依赖要求及已知历史版本说明。
- `docs/REPRODUCTION.md`：复现步骤与协议边界。
- `hydrology/`：径流生成器、历史输入及径流图所需输入。

## 代码与数据来源

DDP、SDP、ISO、Double DQN 和物理模型采用梯级 V11 实现；连续 PPO 采用 V16 实现。绘图与统计采用原经审查的 V6/V16 对应实现。 代码许可证见 [LICENSE](LICENSE)。

### 工程参数与曲线

- 物理参数见 [xiaowan_nuozhadu_v1.yaml](configs/xiaowan_nuozhadu_v1.yaml)。正式曲线分别位于 [Reservoir X](data/xiaowan_nuozhadu/reservoir_1/) 和 [Reservoir N](data/xiaowan_nuozhadu/reservoir_2/)。
- 曲线与研究留存的 `4-水位-库容曲线.xlsx`、`7-尾水位泄流量曲线.xlsx` 中 `1小湾`、`4糯扎渡` 工作表对应。正式库容 CSV 分别包含 75/48 个运行区间控制点，单位为 10⁹ m³；原工程表的万 m³ 数值除以 100000 后转换为该单位。尾水位原始 CSV 分别包含 41/63 点，流量单位为 m³/s，水位单位为 m。
- Reservoir X 尾水位拟合另追加配置中的 `(15666 m³/s, 1012.71 m)`、`(20683 m³/s, 1016.64 m)` 两个锚点，Reservoir N 未另加锚点。现有归档尚未完整记录部分工程资料的原发布单位、报告编号及这两个锚点的上游工程出处。

### 历史与合成径流

- 生成器历史输入见 [interval_inflow.xlsx](hydrology/generator/input/interval_inflow.xlsx)，覆盖 1978—2008 年。包含 Reservoir X 天然入流、Reservoir N 天然入流，以及二者之差形成的区间天然入流。原始观测资料的具体提供单位和原始资料名称，现有归档尚未完整注明。
- 合成序列采用 `Target-Synthetic-Streamflow-Generator-monthly-dual-v1.0`，源码位于 [hydrology/generator](hydrology/generator/)，配置见 [config.json](hydrology/generator/config.json)。TRAIN 种子为 `2026090301`，EVAL 为 `2026090302`；每组 400 年、4800 个月，年份标签为 2009—2408。
- 正式联合输入见 [joint 目录](data/xiaowan_nuozhadu/inflow/target_synthetic_monthly_dual_v1_20260903/joint/)，包含上游天然入流与下游区间入流，单位为 m³/s。它们是合成序列，不是 400 年实测历史记录；正式实验只有 TRAIN/EVAL，不采用 dry/wet 情景。
- 生成器、模型和图件的文件来源分别记录于来源清单。部分工程资料和原始历史径流的上游出处与独立使用条款，现有归档尚未完整记录。


## 复现边界

完整历史训练环境锁文件尚不齐全，已记录版本见 `environments/historical_versions.json`。重新训练所得结果可能受软件版本、随机性和硬件影响，不保证逐位一致。

双库 ISO 保留既有实验协议：先生成 400 年教师，再切分 320/80 年用于拟合与验证；验证初值来自全期教师，因此不声称教师信息严格隔离。独立 EVAL 与 TRAIN 分离。所提供 ISO 配置为固定 RF 参数的代表实验，不等同于完整历史超参数搜索。

## 结果管理

结果与模型保留在 `outputs/`；`.gitignore` 同时排除模型权重、缓存、日志及旧结果归档目录。输入数据和实验配置需要纳入版本控制，因此不按 CSV、Excel 或 JSON 扩展名一律忽略。
