# Checkpoints

EUMU uses one packaged model directory for inference and three tagging
checkpoint stages during training.

## Packaged Inference Checkpoints

Required files:

```text
model/
  model.safetensors
  config.json
  config.yaml
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  configuration_florence2.py
  modeling_florence2.py
  processing_florence2.py
  detection_eval_aliases.json
  task_a_vocab.json
  task_a_heads.pt
```

`task_a_heads.pt` is the final assembled tagging checkpoint used by
`EUMUPredictor`. The filename is retained for loader compatibility.

## Training Checkpoints

```text
training_runs/tagging_quality/tagging_heads.pt
training_runs/tagging_main/tagging_heads.pt
training_runs/final_tagging/tagging_heads.pt
```

`training_runs/final_tagging/tagging_heads.pt` is copied to
`model/task_a_heads.pt` by:

```bash
bash training/run_assemble_tagging_heads.sh --install-to-model
```

## Hugging Face

The evaluation checkpoint package is mirrored at:

```text
https://huggingface.co/Dayoung-space/EUMU
```
