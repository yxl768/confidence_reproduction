"""Check the two published checkpoint/config pairs before ESM inference."""

from argparse import Namespace
from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "official"))

import torch  # noqa: E402
import yaml  # noqa: E402
from utils.diffusion_utils import t_to_sigma  # noqa: E402
from utils.utils import get_model  # noqa: E402


def check(kind: str, ckpt_name: str, confidence_mode: bool) -> None:
    folder = ROOT / "official" / "workdir" / kind
    with (folder / "model_parameters.yml").open(encoding="utf-8") as stream:
        args = Namespace(**yaml.full_load(stream))
    device = torch.device("cpu")
    model = get_model(
        args,
        device,
        t_to_sigma=partial(t_to_sigma, args=args),
        no_parallel=True,
        confidence_mode=confidence_mode,
    )
    weights = torch.load(folder / ckpt_name, map_location=device)
    model.load_state_dict(weights, strict=True)
    print(f"{kind}: loaded {len(weights)} tensors; "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")


if __name__ == "__main__":
    check("pretrained_score", "best_ema_inference_epoch_model.pt", False)
    check("pretrained_confidence", "best_model.pt", True)
