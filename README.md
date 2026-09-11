# Efficient Unified Multimodal Understanding (EUMU)

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-EUMU-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/Dayoung-space/EUMU)

This repository contains the official implementation of **Efficient Unified
Multimodal Understanding (EUMU)**, our first-place solution for the **Mobile
Unified Multimodal Understanding (MUMU) Track** of the 8th Large-scale Video
Object Segmentation (LSVOS) Challenge, held with ECCV 2026.

MUMU asks a single efficient model to jointly solve three image understanding
problems:

| Output | Prediction Target | Metric |
|---|---|---|
| Tagging | Quality, scene, and event concepts | Macro-F1 |
| Detection | Open-vocabulary object labels, confidence scores, and `bbox_xyxy` boxes | Novel-aware mAP |
| Captioning | One concise English sentence | CIDEr, SPICE, and CLIPScore mixture |

The challenge also constrains the model to one integrated architecture with no
separate task-specific models, at most 0.5B parameters, and at most 8 GB peak
inference memory.

## Key Results

| Item | Value |
|---|---:|
| Final challenge score | `17.3409` |
| Parameters | `239.169M` |
| GFLOPs @224 | `23.947` |
| Peak inference memory | `4.5 GB` |
| Base model | `microsoft/Florence-2-base` |

## Method Overview

EUMU uses one pretrained Florence-2-base model for all three outputs. Detection
and captioning use Florence's prompt-based capabilities directly, while
tagging is handled by lightweight attention-pooling heads trained on shared
visual features.

```text
image
  |
  v
shared Florence-2-base visual-language model
  |             |                |
  v             v                v
tagging heads   prompt detection prompt captioning
  |             |                |
  +-------------+----------------+
                |
                v
task-aware inference refinement
                |
                v
tags + detections + caption
```

The refinement stage reuses outputs across tasks:

| Component | Refinement Signal |
|---|---|
| Tagging | Image statistics refine quality; captions and detections refine scene/event labels |
| Detection | Caption cues recover likely missed objects and normalize labels |
| Captioning | Detection cues help select and clean the final caption |

This keeps the system unified while letting each output benefit from the other
outputs at inference time.

## What Is Included

This repository contains the submitted inference package, packaged checkpoints,
evaluation script, and training recipe needed to reproduce the final EUMU
submission when the official datasets are available.

| Area | Files |
|---|---|
| Inference API | [`inference.py`](inference.py) |
| Batch prediction | [`evaluation/generate_predictions.py`](evaluation/generate_predictions.py) |
| Packaged model | [`model/`](model/) |
| Training scripts | [`training/`](training/) |
| Recipe docs | [`docs/`](docs/) |
| Submitted predictions | [`predictions.json`](predictions.json) |

The packaged checkpoint is also mirrored on Hugging Face:

```text
https://huggingface.co/Dayoung-space/EUMU
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run single-image inference:

```python
from inference import EUMUPredictor

predictor = EUMUPredictor(weight_path="model", config_path="model/config.yaml")
result = predictor.predict("/path/to/image.jpg")
```

Generate a submission-style prediction file:

```bash
python evaluation/generate_predictions.py \
  --image-dir /path/to/eval/images \
  --ids-file /path/to/image_ids.txt \
  --out predictions.generated.json
```

If `--ids-file` is omitted, image ids are taken from filename stems in sorted
order.

## Reproducing the Final Submission

The final EUMU submission can be reproduced from the packaged checkpoints, the
training recipe, and the evaluation script. The detailed instructions are split
into focused English docs:

| Topic | File |
|---|---|
| Dataset structure and pseudo-data generation | [`docs/dataset.md`](docs/dataset.md) |
| Environment and hardware | [`docs/env.md`](docs/env.md) |
| Tagging-head training and final checkpoint assembly | [`docs/training.md`](docs/training.md) |
| Threshold search and prediction generation | [`docs/eval.md`](docs/eval.md) |
| Checkpoint files and Hugging Face mirror | [`docs/checkpoints.md`](docs/checkpoints.md) |

Minimal command flow:

```bash
# 1. Prepare official training rows and derived tagging pseudo-data.
SOURCE_DIR=data/train_data \
CAPTION_JSONL=data/train_data/train_caption.jsonl \
IMAGE_ROOTS=/path/to/train/images \
bash training/run_prepare_tagging_data.sh

# 2. Train the quality-specialized tagging head.
NPROC=4 WEIGHT_PATH=model bash training/run_train_quality_tagging.sh

# 3. Train the main scene/event tagging head.
NPROC=4 WEIGHT_PATH=model bash training/run_train_main_tagging.sh

# 4. Assemble the final tagging checkpoint.
bash training/run_assemble_tagging_heads.sh

# 5. Search scene/event thresholds on local validation data.
VAL_ROOT=data/val_data bash training/run_search_tagging_thresholds.sh

# 6. Install the final assembled checkpoint into the inference package.
SCENE_SUMMARY=training_runs/threshold_search/scene_summary.json \
EVENT_SUMMARY=training_runs/threshold_search/event_summary.json \
bash training/run_assemble_tagging_heads.sh --install-to-model
```

Detection and captioning do not require additional training checkpoints in this
release. They are reproduced through the packaged Florence-2-base model and
`model/config.yaml`.

## Repository Structure

```text
predictions.json
inference.py
requirements.txt
model/
  config.yaml
  config.json
  model.safetensors
  task_a_heads.pt
  task_a_vocab.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  preprocessor_config.json
  detection_eval_aliases.json
  configuration_florence2.py
  modeling_florence2.py
  processing_florence2.py
training/
  prepare_tagging_data.py
  train_tagging_heads.py
  search_tagging_thresholds.py
  assemble_tagging_heads.py
evaluation/
  generate_predictions.py
docs/
  dataset.md
  env.md
  training.md
  eval.md
  checkpoints.md
```

The filenames `model/task_a_heads.pt`, `model/task_a_vocab.json`, and the
`task_a` config key are retained for loader and checkpoint compatibility.
