# Training

The final EUMU checkpoint is assembled from two tagging-head training profiles:
a quality-specialized profile and a main scene/event profile. Florence-2-base
is frozen; only the lightweight attention-pooling tagging heads are trained.

## 1. Prepare Data

```bash
SOURCE_DIR=data/train_data \
CAPTION_JSONL=data/train_data/train_caption.jsonl \
IMAGE_ROOTS=/path/to/train/images \
bash training/run_prepare_tagging_data.sh
```

## 2. Train Quality Tagging Head

```bash
NPROC=4 \
WEIGHT_PATH=model \
bash training/run_train_quality_tagging.sh
```

Hyperparameters:

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `2e-4` |
| Batch size | `64` per GPU |
| Effective batch size | `256` |
| Epochs | `6` |
| GPUs | `4` |
| Warmup ratio | `0.08` |
| Weight decay | `0.04` |
| Dropout | `0.25` |
| Trunk dimension | `768` |
| Focal gamma | `1.25` |
| Label smoothing | `0.02` |
| CE label smoothing | `0.05` |
| Quality loss weight | `2.0` |
| Scene loss weight | `0.8` |
| Event loss weight | `1.8` |
| Synthetic quality probability | `0.0` |
| Max positive weight | `8.0` |
| Max class weight | `6.0` |
| Workers per process | `4` |

Output:

```text
training_runs/tagging_quality/tagging_heads.pt
```

## 3. Train Main Tagging Head

```bash
NPROC=4 \
WEIGHT_PATH=model \
bash training/run_train_main_tagging.sh
```

Hyperparameters:

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Batch size | `96` per GPU |
| Effective batch size | `384` |
| Epochs | `10` |
| GPUs | `4` |
| Warmup ratio | `0.08` |
| Weight decay | `0.04` |
| Dropout | `0.25` |
| Trunk dimension | `768` |
| Focal gamma | `1.0` |
| Label smoothing | `0.01` |
| Quality loss weight | `1.8` |
| Scene loss weight | `1.0` |
| Event loss weight | `1.7` |
| Synthetic quality probability | `0.35` |
| Max positive weight | `12.0` |
| Workers per process | `6` |

Output:

```text
training_runs/tagging_main/tagging_heads.pt
```

## 4. Assemble Final Tagging Checkpoint

```bash
bash training/run_assemble_tagging_heads.sh
```

This copies the quality head from `training_runs/tagging_quality/`, keeps the
attention pool plus scene/event heads from `training_runs/tagging_main/`, and
applies the final threshold defaults.

Output:

```text
training_runs/final_tagging/tagging_heads.pt
```

To install the assembled checkpoint into the packaged inference model:

```bash
bash training/run_assemble_tagging_heads.sh --install-to-model
```

Installed file:

```text
model/task_a_heads.pt
```

The `task_a` filename is kept for compatibility with the packaged loader and
checkpoint metadata.
