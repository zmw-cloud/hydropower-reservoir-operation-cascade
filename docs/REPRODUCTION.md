# Reproduction

## Three distinct operations

1. Read saved trajectories and verify their recorded statistics: audit_repository.py.
2. Load a frozen policy and execute a new evaluation: prepare then execute, stage=evaluate.
3. Train or solve again: prepare then execute with an explicit training/solve stage.

Integration verification uses operation1 and small synthetic invariant tests only. It does not execute operation2 or3, regenerate hydrology or render the paper figures.

## Isolated execution

run_workflow.py copies src, scripts, configs and data into a new outputs/runs/RUN/workspace. The original algorithm's relative output convention is retained inside this workspace. Frozen model bindings are copied only for their documented stages. Core algorithm functions are not rewritten to impose a common state, action or reward definition.

prepare refuses an existing directory. execute creates its marker and log exclusively and refuses a second execution. Failure requires a fresh preparation rather than silently reusing outputs. Audit reports also use exclusive creation. The original external projects are never runtime dependencies.

Example, replace the placeholder with the printed directory:

~~~text
python -B scripts/run_workflow.py prepare --algorithm sdp --stage evaluate
python -B scripts/run_workflow.py execute outputs/runs/RUN
~~~

## Single reservoir stages

| Algorithm | Stages |
|---|---|
| ddp | solve |
| sdp | train, evaluate |
| iso | development, lambda-scan, select-lambda, final-train, evaluate |
| dqn | development, final, evaluate |

Final DQN: lambda650,2300episodes/seed. Development is the saved3000episode budget. ISO development uses anchor1200; validation50 scan chooses700, then the final policy uses all400 training years. Stage-specific inputs are documented by configs/paper/workflows.json.

## Cascade stages

| Algorithm | Stages |
|---|---|
| ddp | solve |
| sdp | train, evaluate |
| iso | development, final-train, evaluate |
| dqn | development, final, evaluate |
| ppo | development, final, evaluate |

Cascade ISO500 development with a single fixed RF parameter set is not a rerun of the earlier RF hyperparameter search. The eleven-teacher penalty selection is preserved as evidence. Repeating the entire historical tuning sequence requires its recorded candidate list and teacher generation, not a claim that this one representative command repeats every historical search.

SDP preparation uses the saved physical/settings YAML bytes so policy provenance is preserved; the only difference from current physics YAML was a trailing blank line. ISO configuration adds the verified train_400y teacher guard. DQN/PPO use exact archived final configurations, including their own reward scales and action representations.

## Environments

Install the project with its documented extras. The data audit can run without Torch or loading pickle models. Use sklearn1.7.2 for the single RF artifact and1.9.1 for the cascade RF artifact according to saved software metadata. Keep those environments separate if necessary. No full historical lockfile is invented. New environment/version observations from integration are diagnostic evidence, not historical training versions.

## Output and source records

Each run has run_plan.json, execution_started.json, execution.log and execution_result.json. Historical absolute paths remain untouched inside frozen artifact metadata as source evidence. New run metadata records the new paths naturally.

provenance/source_manifest.json lists original paths and hashes. integration_changes.json explains adaptations. release_manifest.json hashes the delivered repository inputs; outputs are excluded. If later edits are authorized, preserve the prior manifest and issue a new release manifest rather than silently claiming old hashes still describe new code.
