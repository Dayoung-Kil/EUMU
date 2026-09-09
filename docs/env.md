# Environment

EUMU was packaged as a self-contained Florence-2-base inference model plus
tagging heads. The same environment is used for training the tagging heads and
generating evaluation predictions.

## Python

Use Python 3.10.

```bash
conda create -n eumu python=3.10 -y
conda activate eumu
pip install -r requirements.txt
```

Pinned and required packages are listed in `requirements.txt`:

```text
torch==2.9.0
transformers==4.41.2
tokenizers>=0.19,<0.20
huggingface-hub>=0.23
safetensors>=0.4
timm>=1.0
einops>=0.7
pillow>=10.0
pyyaml>=6.0
```

## Hardware

The final tagging recipe uses 4 GPUs with PyTorch distributed training.

Inference resource declaration:

```json
{
  "parameters_m": 239.169,
  "gflops_224": 23.947,
  "peak_memory_gb": 4.5
}
```

The packaged inference config uses CUDA when available and falls back to CPU
only if CUDA is unavailable.
