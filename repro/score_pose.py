"""Score DiffDock-S SDF poses against a crystal ligand without realignment.

Run from anywhere with Python, RDKit, NumPy, NetworkX and the bundled official
spyrmsd package present in ../official. The reference and predictions must use
the original receptor coordinate frame. Hydrogens are removed before scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "official"))

from rdkit import Chem  # noqa: E402
from spyrmsd import molecule, rmsd  # noqa: E402


def read_sdf(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)
    if len(supplier) != 1 or supplier[0] is None:
        raise ValueError(f"Expected one readable molecule in {path}")
    mol = supplier[0]
    if mol.GetNumConformers() != 1:
        raise ValueError(f"Expected one 3D conformer in {path}")
    return molecule.Molecule.from_rdkit(mol)


def symmetry_rmsd(reference, predicted) -> float:
    if reference.natoms != predicted.natoms:
        raise ValueError(
            f"Heavy-atom count differs: reference={reference.natoms}, "
            f"prediction={predicted.natoms}"
        )
    value = rmsd.symmrmsd(
        reference.coordinates,
        predicted.coordinates,
        reference.atomicnums,
        predicted.atomicnums,
        reference.adjacency_matrix,
        predicted.adjacency_matrix,
        center=False,
        minimize=False,
    )
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("RMSD is not finite")
    return value


def list_ranked_poses(pred_dir: Path):
    if not pred_dir.is_dir():
        raise NotADirectoryError(pred_dir)
    ranked = {}
    confidences = {}
    for path in pred_dir.glob("rank*.sdf"):
        plain = re.fullmatch(r"rank([1-9]\d*)\.sdf", path.name)
        scored = re.fullmatch(
            r"rank([1-9]\d*)_confidence([-+]?\d+(?:\.\d+)?)\.sdf",
            path.name,
        )
        if plain:
            ranked[int(plain.group(1))] = path
        elif scored:
            rank = int(scored.group(1))
            ranked.setdefault(rank, path)
            confidences[rank] = float(scored.group(2))
    if 1 not in ranked:
        raise ValueError(f"No rank1 pose found in {pred_dir}")
    return ranked, confidences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-poses", type=int)
    parser.add_argument("--require-confidence", action="store_true")
    args = parser.parse_args()

    reference = read_sdf(args.reference)
    ranked, confidences = list_ranked_poses(args.pred_dir)
    expected_ranks = list(range(1, len(ranked) + 1))
    if sorted(ranked) != expected_ranks:
        raise ValueError(f"Pose ranks are not contiguous: {sorted(ranked)}")
    if args.expect_poses is not None and len(ranked) != args.expect_poses:
        raise ValueError(f"Expected {args.expect_poses} poses, found {len(ranked)}")
    if args.require_confidence:
        if sorted(confidences) != expected_ranks:
            raise ValueError(f"Confidence is missing for some ranks: {sorted(confidences)}")
        scores = [confidences[rank] for rank in expected_ranks]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Confidence contains a non-finite value")
        if any(left < right for left, right in zip(scores, scores[1:])):
            raise ValueError("Poses are not sorted by descending confidence")
    rows = []
    for rank, path in sorted(ranked.items()):
        value = symmetry_rmsd(reference, read_sdf(path))
        rows.append(
            {
                "rank": rank,
                "pose": str(path.resolve()),
                "confidence": confidences.get(rank),
                "symmetry_rmsd_angstrom": round(value, 4),
            }
        )
    result = {
        "reference": str(args.reference.resolve()),
        "n_poses": len(rows),
        "top1_rmsd_angstrom": rows[0]["symmetry_rmsd_angstrom"],
        "top1_under_2_angstrom": rows[0]["symmetry_rmsd_angstrom"] < 2.0,
        "scoring": "heavy atoms; graph symmetry correction; no centering or alignment",
        "poses": rows,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
