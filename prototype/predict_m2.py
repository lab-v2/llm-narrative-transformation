"""
LLM-free survey with a trained M2 model: story text in, 20 ratings out.

For each story, end to end:
    read story -> OpenAI embedding -> cosine distances to the saved prototypes
    (sorted within each rating) -> scaling -> 20 per-question networks -> ratings

Same output files, timing and options as predict_m1.py.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/predict_m2.py \
        --stories-dir data/test-prototype \
        --model-dir output_test/prototype/models/m2_h32-16 \
        --output-dir output_test/prototype/predictions/m2_h32-16/test
"""

from pathlib import Path

import torch

from predict_m1 import run
from train_m1 import PerQuestionMLP


def load_model(model_dir: Path, args=None):
    model_file = model_dir / "m2_final.pt"
    ckpt = torch.load(model_file, weights_only=True)
    cfg = ckpt["config"]
    n_features = cfg["n_classes"] * cfg["k_per_rating"]
    model = PerQuestionMLP(len(cfg["question_ids"]), n_features, tuple(cfg["hidden"]), cfg["n_classes"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tensors = [ckpt["prototypes"], ckpt["scaler_mean"], ckpt["scaler_std"], *ckpt["state_dict"].values()]
    info = {
        "name": cfg["model"],
        "file": str(model_file),
        "n_parameters": sum(t.numel() for t in tensors),
        "size_bytes": sum(t.numel() * t.element_size() for t in tensors),
    }
    return model, ckpt, cfg, info


def predict_one(model, ckpt, vector: torch.Tensor) -> torch.Tensor:
    """embedding (D,) -> probabilities (Q, C). Same features as train_m2.distances + scaling."""
    cfg = ckpt["config"]
    k = cfg["k_per_rating"]
    with torch.no_grad():
        v = vector / vector.norm().clamp_min(1e-12)
        d = 1.0 - ckpt["prototypes"] @ v                                          # (Q, C*k)
        d = torch.where(ckpt["prototype_mask"], d, torch.tensor(cfg["empty_distance"]))
        Q, S = d.shape
        d = d.view(Q, S // k, k).sort(dim=-1).values.view(Q, 1, S)               # nearest-first per rating
        z = (d - ckpt["scaler_mean"]) / ckpt["scaler_std"]
        return torch.softmax(model(z), dim=-1)[:, 0, :]


def main():
    run(load_model, predict_one, "M2")


if __name__ == "__main__":
    main()
