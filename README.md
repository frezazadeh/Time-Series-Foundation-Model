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


### Authentication

Create a `.env` file in the project root (or copy `.env.example`) and paste your
[Hugging Face access token](https://huggingface.co/settings/tokens):

```bash
cp .env.example .env
echo "HF_TOKEN=<your_token>" >> .env
```

All scripts automatically load environment variables at runtime via `python‑dotenv`,
so there’s no need to export the token manually.


## Quick start

```bash
# 1 : clone & install
git clone https://github.com/frezazadeh/Time-Series-Foundation-Model.git
cd timeseries-foundation
pip install -r requirements.txt

# 2 : pre‑train
python scripts/pretrain.py --data_dir /path/to/tspile

# 3 : fine‑tune
python scripts/finetune.py --ckpt checkpoints/model_epoch_50.pt \
                           --x /content/x.npy --y /content/y.npy

# 4 : run inference
python scripts/infer.py --ckpt supervised_model.pt --x /content/x.npy
```

See the **`examples/`** folder for notebook demos.

---

© 2025 Farhad Rezazadeh· MIT License
