"""
LLM-free survey with a trained M4 model: story text in, answers to all questions out.

For each story, end to end:
    read story -> OpenAI embedding -> PCA + scaling -> paired with each question's saved
    features -> one shared network -> answers

Same output files, timing and options as predict_m1.py.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/predict_m4.py \
        --stories-dir data/test-prototype \
        --model-dir output_test/prototype/models/m4_binary_h16 \
        --output-dir output_test/prototype/predictions/m4_binary_h16/test
"""

from pathlib import Path

import torch

from predict_m1 import run
from train_m4 import SharedMLP


def load_model(model_dir: Path, args=None):
    model_file = model_dir / "m4_final.pt"
    ckpt = torch.load(model_file, weights_only=True)
    cfg = ckpt["config"]
    in_dim = cfg["pca_k"] + ckpt["question_features"].shape[1]
    model = SharedMLP(in_dim, tuple(cfg["hidden"]), cfg["n_classes"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tensors = [ckpt["pca_components"], ckpt["pca_mean"], ckpt["scaler_mean"], ckpt["scaler_scale"],
               ckpt["question_features"], *ckpt["state_dict"].values()]
    info = {
        "name": cfg["model"],
        "file": str(model_file),
        "n_parameters": sum(t.numel() for t in tensors),
        "size_bytes": sum(t.numel() * t.element_size() for t in tensors),
    }
    return model, ckpt, cfg, info


def predict_one(model, ckpt, vector: torch.Tensor) -> torch.Tensor:
    """embedding (D,) -> probabilities (Q, C). Same features as train_m4.pair_features."""
    with torch.no_grad():
        zs = (vector - ckpt["pca_mean"]) @ ckpt["pca_components"].T
        zs = (zs - ckpt["scaler_mean"]) / ckpt["scaler_scale"]                       # (Fs,)
        zq = ckpt["question_features"]                                                # (Q, Fq)
        x = torch.cat([zs.expand(len(zq), -1), zq], dim=-1).unsqueeze(1)             # (Q, 1, Fs + Fq)
        return torch.softmax(model(x), dim=-1)[:, 0, :]


def main():
    run(load_model, predict_one, "M4")


if __name__ == "__main__":
    main()
