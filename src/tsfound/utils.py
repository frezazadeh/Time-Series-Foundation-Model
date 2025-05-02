"""Utility helpers (device selection, seeding, etc.)."""
import torch, random, os


# Auto‑load secrets from a .env file at project root
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()

def get_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda"), torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps"), torch.float32
    else:
        return torch.device("cpu"), torch.float32

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
