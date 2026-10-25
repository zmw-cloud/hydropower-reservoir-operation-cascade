# Confirmed integration decisions

- The user confirmed on2026-10-22 that the formal single ISO is RF700.
- Single ISO-RF700:20.41323808751621TWh/yr and81.97916666666667% reliability.
- The manuscript's20.413TWh with79.98% combines the RF energy with the old ISO-SURF1100 reliability. Record the correction; this integration does not edit the manuscript file.
- Freeze effective settings from result packages rather than generic defaults.
- Keep the existing cascade ISO experimental protocol and disclose its teacher split.
- Include only previously requested redrawn/new paper figure tools in the cascade repository; no single-reservoir plotting scripts.
- Formal hydrology has TRAIN and EVAL only. Dry/wet are not formal experiments.
- Preserve original code/configuration/result sources and never write new outputs there.

## Why1223and812are not invalid water levels

They are physically valid levels at the start of the cascade ISO validation period. The issue concerns where they came from: the400-year DDP teacher uses the full realized TRAIN hydrology, including the last80years later called validation. Its first320year targets and the validation-start storage can depend on that last80year future.

The fitted RF itself receives first320year teacher rows, but that is not the same as solving a teacher solely on those320years. Thus the protocol is a teacher-trajectory320/80split, not strict teacher-information isolation. The independent400-year EVAL file remains separate. No new result or replacement teacher is generated.

## Other definitions retained

DDP is a perfect-foresight reference under its objective/grid/boundary conditions. SDP can request a target subsequently corrected by physics; actual continuous storage is carried forward. DQN/PPO mask/map before action execution. Feedback evaluation does not force final refill or reset annually.

The paper minimum-of-mean station reliability differs from the mean of per-seed minima. Total-power reliability and simultaneous station satisfaction are different again. Across-seed SD of long-run energy is not annual generation variability.

The synthetic extreme-inflow test deliberately reaches a conflicting full-storage/release-cap case. The historical simulator reports a nonzero water-balance residual after clipping; it is not universally feasible under arbitrary extreme forcing. The final recorded paper trajectories pass the0.001m3 residual criterion. No physical formula is silently changed to repair an unobserved experimental case.

## Remaining historical evidence limits

The source directories do not contain Git history. DQN/PPO metadata provides configuration, method and original paths, but not a complete hash list for all source executed during historical training. Formal DQN5000 reuses existing final runs; strict training/selection chronology is not fully established. Historical complete environment lockfiles are not fabricated.
