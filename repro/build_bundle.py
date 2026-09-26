"""Build the small AutoDL upload bundle from verified source and repro files."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "1a0q_repro_bundle.zip"
PREFIX = Path("confidence_reproduction")


def included_files():
    yield ROOT / "README.md"
    yield ROOT / "AUTODL_CHECKLIST.md"
    for folder in (ROOT / "official", ROOT / "repro"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                yield path


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in included_files():
        archive.write(path, (PREFIX / path.relative_to(ROOT)).as_posix())

with zipfile.ZipFile(OUTPUT) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupt member: {bad}")
    members = archive.namelist()
    required = {
        "confidence_reproduction/AUTODL_CHECKLIST.md",
        "confidence_reproduction/README.md",
        "confidence_reproduction/repro/paper_experiments.py",
        "confidence_reproduction/repro/run_autodl_1a0q.sh",
        "confidence_reproduction/repro/setup_linux.sh",
        "confidence_reproduction/repro/run_1a0q.sh",
        "confidence_reproduction/repro/setup_autodl_vgpu32.sh",
        "confidence_reproduction/repro/run_1a0q_vgpu32.sh",
        "confidence_reproduction/repro/patch_official_inference.py",
        "confidence_reproduction/official/data/1a0q/1a0q_ligand.sdf",
        "confidence_reproduction/official/data/1a0q/1a0q_protein_processed.pdb",
        "confidence_reproduction/official/workdir/pretrained_score/best_ema_inference_epoch_model.pt",
        "confidence_reproduction/official/workdir/pretrained_confidence/best_model.pt",
    }
    missing = sorted(required.difference(members))
    if missing:
        raise RuntimeError(f"Missing required members: {missing}")

digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
print(f"{OUTPUT}\nfiles={len(members)}\nbytes={OUTPUT.stat().st_size}\nsha256={digest}")
