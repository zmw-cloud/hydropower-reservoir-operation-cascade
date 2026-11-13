# 从源码复现

所有命令均在当前仓库根目录运行。输入数据与论文配置随仓库保留；模型、教师、选择结果、评价轨迹和图件由运行生成，统一保存在 `outputs/`。各阶段共享这个目录，以便后续阶段读取前一阶段的产物。首次运行必须按顺序生成依赖。

```bash
python -m pip install -e ".[all]"
python -B scripts/audit_repository.py
python -B scripts/run_workflow.py inspect
```

`run_workflow.py run` 执行相应原算法入口，不复制旧模型。附加 `--show-commands` 只打印命令，不执行。大规模求解、训练与扫描可能耗时较长。

## DDP 与 SDP

DDP 在 EVAL 上直接求解，作为完美预见参考；SDP 先在 TRAIN 上生成策略，再评价：

```bash
python -B scripts/run_workflow.py run --algorithm ddp --stage solve
python -B scripts/run_workflow.py run --algorithm sdp --stage train
python -B scripts/run_workflow.py run --algorithm sdp --stage evaluate
```

## DQN

论文代表配置已保留。开发阶段用于验证，最终训练使用既定参数与预算，最后才读取独立 EVAL：

```bash
python -B scripts/run_workflow.py run --algorithm dqn --stage development
python -B scripts/run_workflow.py run --algorithm dqn --stage final
python -B scripts/run_workflow.py run --algorithm dqn --stage evaluate
```

种子为 1、42、100。具体网络、训练预算及惩罚系数见 `data/experiment_settings.yaml`（单库）或 `configs/paper/dqn5000*.yaml`（双库）。需要更改实验参数时先另存配置，避免将新实验误称为原论文实验。

## ISO-RF

先以 TRAIN 径流和 λ=500 生成 DDP 教师，再运行 ISO：

```bash
python -B scripts/run_workflow.py run --algorithm iso --stage teacher
python -B scripts/run_workflow.py run --algorithm iso --stage development
python -B scripts/run_workflow.py run --algorithm iso --stage final-train
python -B scripts/run_workflow.py run --algorithm iso --stage evaluate
```

教师保存在 `outputs/DDP/teachers/failure_penalty_500/`，配置见 `configs/paper/iso500.yaml`。保留历史 400 年教师后切分 320/80 年的实验，验证初值来自全期教师，因此不声称教师信息严格隔离。固定 RF 配置不等同于历史完整超参数搜索或十一组教师惩罚选择。

## 连续 PPO

```bash
python -B scripts/run_workflow.py run --algorithm ppo --stage development
python -B scripts/run_workflow.py run --algorithm ppo --stage final
python -B scripts/run_workflow.py run --algorithm ppo --stage evaluate
```

配置见 `configs/paper/ppo6000*.yaml`，种子为 1、42、100。

## 径流与绘图

正式 TRAIN/EVAL 径流已包含在 `data/`，复现调度实验无需重新生成。径流生成器及历史输入保留在 `hydrology/generator/`，其结果写入 `outputs/hydrology_generated/`。

绘图代码保留，但调度结果和扫描数据不随源码提交。先运行实验，然后将以下配置的输入路径改为自己生成的轨迹、JSON 和选择记录：

- `configs/comparison_ddp_sdp_iso_dqn_ppo.yaml`：双库评价轨迹。
- `configs/paper_results.json`：单库/双库年度与迁移统计。
- `configs/paper/reference_results.json`：可选结果校验清单。
- `scripts/plot_generation_reliability_tradeoff.py`：权衡扫描输入路径。

默认的 `outputs/paper_inputs/` 路径仅是输入模板；其中带散列目录名的路径沿用论文结果包的布局，不代表仓库已包含那些结果。也可以在本地将自己的数据组织到这个布局。多案例与权衡图还需要生成对应的单库结果、验证扫描和选择统计，仅运行一个代表策略不足以生成全部图。

```bash
python -B scripts/plot_final_figures.py --check-only
python -B scripts/plot_final_figures.py --figures all
```

检查需要上述全部结果齐备。绘图默认使用 Times New Roman，可通过 `--font-dir` 指定本地字体；字体不随源码分发。

## 校验与版本

```bash
python -B -m unittest discover -s tests -v
```

默认 `audit_repository.py` 检查两份正式径流与 `data/datasets_manifest.json` 的 SHA-256，不需要训练产物。

如需校验本地评价结果，先将 `configs/paper/reference_results.json` 的轨迹和汇总路径指向自己生成的文件，再运行：

```bash
python -B scripts/audit_repository.py --results
```

完整历史训练环境锁文件未全部留存；已知版本记录在 `environments/historical_versions.json`，不作为完整环境锁定文件。新训练的 RF 策略应在兼容的软件环境中评价，重训结果可能受随机性、库版本与硬件影响。

极端入流超过下泄能力且水库已满时，历史物理实现可能报告未解决库容超限和非零水量平衡残差；保留相应测试，不在本次仓库清理中改变论文算法行为。

## 仓库约定

`outputs/`、旧 `artifacts/`、`reference_results/`、`provenance/` 目录，以及模型权重、缓存、日志均已忽略。不要使用 `git add -f` 强行提交运行产物。输入径流、工程曲线、实验配置与测试源码属于复现必需内容，应保留在 Git 中。
