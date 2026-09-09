# Evaluation

EUMU evaluation uses the packaged `model/` directory and the assembled tagging
checkpoint installed at `model/task_a_heads.pt`.

## Tagging Threshold Search

After training and assembling the initial final tagging checkpoint, run:

```bash
VAL_ROOT=data/val_data \
bash training/run_search_tagging_thresholds.sh
```

Outputs:

```text
training_runs/threshold_search/scene_summary.json
training_runs/threshold_search/event_summary.json
```

Then rebuild the final tagging checkpoint with the searched thresholds:

```bash
SCENE_SUMMARY=training_runs/threshold_search/scene_summary.json \
EVENT_SUMMARY=training_runs/threshold_search/event_summary.json \
bash training/run_assemble_tagging_heads.sh --install-to-model
```

The assembly script also contains the final threshold defaults, so the packaged
checkpoint can be rebuilt even if the intermediate threshold summaries are not
available.

## Prediction Generation

Generate evaluation predictions for an image folder:

```bash
python evaluation/generate_predictions.py \
  --image-dir /path/to/eval/images \
  --ids-file /path/to/image_ids.txt \
  --out predictions.generated.json
```

If no `--ids-file` is provided, image ids are taken from image filename stems
and processed in sorted order.

The output schema is:

```json
{
  "schema_version": "1.0",
  "model_info": {
    "parameters_m": 239.169,
    "gflops_224": 23.947,
    "peak_memory_gb": 4.5
  },
  "predictions": []
}
```

The repository root `predictions.json` is the submitted reference output.
Detection and captioning are reproduced through prompt-based Florence inference
and config-driven post-processing; no additional detection or captioning
training checkpoint is required.
