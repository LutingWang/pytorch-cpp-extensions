# PyTorch Cpp Extensions

## Installation

```bash
conda create -n todd python=3.9
conda activate todd
```

Install `PyTorch`.

```bash
pip config set global.find-links https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1
```

```bash
pip install .\[optional\]
pip install . --no-build-isolation
```

```bash
pytest tests --durations=0 -vv
```
