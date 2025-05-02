# TimeSeries‑Foundation

A modular PyTorch implementation of a **Masked‑patch Transformer** pre‑training pipeline for generic multivariate
time‑series data, plus *supervised fine‑tuning* and *inference* scripts.

<p align="center">
  <img src="https://img.shields.io/badge/Transforms-Time&nbsp;Series-success?style=flat"/>
  <img src="https://img.shields.io/badge/PyTorch‑>=2.2-blue"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blueviolet"/>
</p>

## Features
* **Self‑supervised pre‑training** on huge CSV archives such as the [TimeSeries‑PILE].
* **Supervised fine‑tuning** on downstream KPIs with optional layer un‑freezing.
* **Inference & evaluation** helpers that restore the full preprocessing pipeline.
* Lightweight **package layout** (`src/tsfound`) ready for `pip install -e .`.
* **W&B** logging hooks, HF snapshot download helper and reproducible seed utils.


## Quick start

```bash
# 1 : clone & install
git clone https://github.com/frezazadeh/Time-Series-Foundation-Model.git
cd timeseries-foundation
pip install -e .

# 2 : pre‑train
python scripts/pretrain.py --hf_token your_hf_token

# 3 : fine‑tune
python scripts/finetune.py --ckpt checkpoints/model_epoch_50.pt \
                           --x /dataset/x.npy --y /dataset/y.npy

# 4 : run inference
python scripts/infer.py --ckpt supervised_model.pt --x /dataset/x.npy
```

---

© 2025 Farhad Rezazadeh· MIT License
