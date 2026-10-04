"""
LLM-free survey with a trained M3 model: story text in, 20 ratings out.

For each story, end to end:
    read story -> OpenAI embedding -> PCA + scaling -> encoder (M1 without its
    output layer, per question) -> Euclidean distances to the saved prototypes
    (sorted within each rating) -> readout -> ratings

Readout (--readout):
    head     (default) scaling -> 20 per-question head networks
    nearest  rating of the closest prototype, no network (probabilities are one-hot)

Same output files, timing and options as predict_m1.py.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/predict_m3.py \
        --stories-dir data/test-prototype \
        --model-dir output_test/prototype/models/m3_h32-16 \
        --output-dir output_test/prototype/predictions/m3_h32-16/test
    python prototype/predict_m3.py --readout nearest \
        --stories-dir data/test-prototype \
        --model-dir output_test/prototype/models/m3_h32-16 \
        --output-dir output_test/prototype/predictions/m3_nearest/test
"""

from pathlib import Path

import torch

from predict_m1 import run
from train_m1 import PerQuestionMLP
from train_m3 import hidden_output


def add_args(parser):
    parser.add_argument("--readout", choices=["head", "nearest"], default="head",
                        help="head = M3's networks on the distances (default); "
                             "nearest = rating of the closest prototype, no network")


def load_model(model_dir: Path, args=None):
    readout = getattr(args, "readout", "head")
    model_file = model_dir / "m3_final.pt"
    ckpt = torch.load(model_file, weights_only=True)
    cfg = ckpt["config"]
    Q, C, k = len(cfg["question_ids"]), cfg["n_classes"], cfg["k_per_rating"]

    encoder = PerQuestionMLP(Q, cfg["enc_pca_k"], tuple(cfg["enc_hidden"]), C, cfg["enc_dropout"])
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    head = PerQuestionMLP(Q, C * k, tuple(cfg["hidden"]), C, cfg["dropout"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    tensors = [ckpt["pca_components"], ckpt["pca_mean"], ckpt["scaler_mean"], ckpt["scaler_scale"],
               *ckpt["encoder_state_dict"].values(), ckpt["prototypes"]]
    if readout == "head":
        tensors += [ckpt["dist_mean"], ckpt["dist_std"], *ckpt["state_dict"].values()]
    info = {
        "name": cfg["model"] + (" (nearest-prototype readout)" if readout == "nearest" else ""),
        "readout": readout,
        "file": str(model_file),
        "n_parameters": sum(t.numel() for t in tensors),
        "size_bytes": sum(t.numel() * t.element_size() for t in tensors),
    }
    return (encoder, head, readout), ckpt, cfg, info


def predict_one(model, ckpt, vector: torch.Tensor) -> torch.Tensor:
    """embedding (D,) -> probabilities (Q, C). Same features as train_m3 (encode + euclidean distances)."""
    encoder, head, readout = model
    cfg = ckpt["config"]
    C, k = cfg["n_classes"], cfg["k_per_rating"]
    with torch.no_grad():
        z = (vector - ckpt["pca_mean"]) @ ckpt["pca_components"].T
        z = (z - ckpt["scaler_mean"]) / ckpt["scaler_scale"]
        h = hidden_output(encoder, z.unsqueeze(0))                                       # (Q, 1, H)
        d = ((h[:, :, None, :] - ckpt["prototypes"][:, None, :, :]) ** 2).sum(-1).sqrt()  # (Q, 1, C*k)
        mask = ckpt["prototype_mask"][:, None, :]
        far = torch.where(mask, d, torch.tensor(float("-inf"))).max(-1, keepdim=True).values
        d = torch.where(mask, d, far)
        Q, _, S = d.shape
        d = d.view(Q, 1, S // k, k).sort(dim=-1).values.view(Q, 1, S)                   # nearest-first per rating

        if readout == "nearest":
            nearest = d.view(Q, C, k)[..., 0]                                            # (Q, C)
            has_proto = ckpt["prototype_mask"].view(Q, C, k)[..., 0]
            nearest = torch.where(has_proto, nearest, torch.tensor(float("inf")))
            return torch.nn.functional.one_hot(nearest.argmin(-1), C).float()

        z = (d - ckpt["dist_mean"]) / ckpt["dist_std"]
        return torch.softmax(head(z), dim=-1)[:, 0, :]


def main():
    run(load_model, predict_one, "M3", add_args=add_args)


if __name__ == "__main__":
    main()
