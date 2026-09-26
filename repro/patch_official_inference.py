"""Apply the two-line upstream unpacking fix required by the pinned commit.

`extract_receptor_structure` returns ten values, including chain IDs, while
`InferenceDataset` in this commit unpacks only nine. Keep the unused tenth
value explicit so the published model and preprocessing remain unchanged.
"""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
path = root / "official" / "utils" / "inference_utils.py"
source = path.read_text(encoding="utf-8")
old = "misc_features_array, lm_embeddings, sequences \\"
new = "misc_features_array, lm_embeddings, sequences, _chain_ids \\"

if source.count(old) == 2 and source.count(new) == 0:
    path.write_text(source.replace(old, new), encoding="utf-8")
    print("Patched both inference call sites")
elif source.count(old) == 0 and source.count(new) == 2:
    print("Inference call sites already patched")
else:
    raise RuntimeError("Unexpected upstream file; inspect before patching")
