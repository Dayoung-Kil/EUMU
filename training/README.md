# EUMU Training Scripts

This directory contains the executable recipe scripts for rebuilding the final
EUMU tagging checkpoint.

| Stage | Script |
|---|---|
| Prepare tagging rows | `run_prepare_tagging_data.sh` |
| Train quality-specialized head | `run_train_quality_tagging.sh` |
| Train main scene/event head | `run_train_main_tagging.sh` |
| Search scene/event thresholds | `run_search_tagging_thresholds.sh` |
| Assemble final tagging checkpoint | `run_assemble_tagging_heads.sh` |

See the English recipe docs:

```text
docs/dataset.md
docs/env.md
docs/training.md
docs/eval.md
docs/checkpoints.md
```
