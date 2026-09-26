import importlib.util
import io
import json
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "repro" / "paper_experiments.py"
spec = importlib.util.spec_from_file_location("paper_experiments", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class PaperExperimentScoringTests(unittest.TestCase):
    def test_confidence_top1_is_not_oracle_best_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            np.save(folder / "complex_names.npy", np.array(["complex-A", "complex-B"]))
            np.save(folder / "rmsds.npy", np.array([[3.0, 1.0], [1.5, 4.0]]))
            np.save(folder / "confidences.npy", np.array([[9.0, 1.0], [8.0, 2.0]]))
            actual = experiment.score_directory(folder, ["complex-B", "complex-A"], 2)
            self.assertEqual(actual, {"complex-A": 3.0, "complex-B": 1.5})
            self.assertEqual(sum(value < 2 for value in actual.values()), 1)

    def test_missing_names_are_rejected_instead_of_inflating_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            np.save(folder / "complex_names.npy", np.array(["complex-A"]))
            np.save(folder / "rmsds.npy", np.ones((1, 8)))
            np.save(folder / "confidences.npy", np.ones((1, 8)))
            with self.assertRaises(ValueError):
                experiment.score_directory(folder, ["complex-A", "complex-B"], 8)

    def test_symmetry_fallback_is_not_counted_as_paper_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "run.log").write_text("Using non corrected RMSD because of the error: x\n")
            with self.assertRaisesRegex(ValueError, "Symmetry-corrected RMSD failed"):
                experiment.score_directory(folder, ["complex-A"], 8)

    def test_eight_cluster_summary_weights_by_complex_and_repeat(self):
        counts = [8, 9, 9, 10, 11, 12, 13, 13]
        groups = {f"cluster-{i}": [f"c{i}-{j}" for j in range(n)]
                  for i, n in enumerate(counts, 1)}
        all_names = [name for names in groups.values() for name in names]
        manifest = {"groups": groups, "subset": all_names}

        def write_scores(folder, names, hits):
            folder.mkdir(parents=True)
            rmsds = np.full((len(names), 8), 3.0)
            confidences = np.zeros((len(names), 8))
            rmsds[:hits, 0] = 1.0
            confidences[:, 0] = 1.0
            np.save(folder / "complex_names.npy", np.asarray(names))
            np.save(folder / "rmsds.npy", rmsds)
            np.save(folder / "confidences.npy", confidences)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (101, 202):
                for repeat in (1, 2):
                    base_hits = {names[0] for names in groups.values()}
                    folder = root / "baseline" / f"clusters85_poses8_seed{seed}_rep{repeat}"
                    folder.mkdir(parents=True)
                    rmsds = np.full((85, 8), 3.0)
                    rmsds[:, 0] = [1.0 if name in base_hits else 3.0 for name in all_names]
                    confidences = np.zeros((85, 8))
                    confidences[:, 0] = 1.0
                    np.save(folder / "complex_names.npy", np.asarray(all_names))
                    np.save(folder / "rmsds.npy", rmsds)
                    np.save(folder / "confidences.npy", confidences)
                for index, (cluster, names) in enumerate(groups.items(), 1):
                    model = root / "models" / f"cluster{index:02d}_{experiment.slug(cluster)}_seed{seed}"
                    model.mkdir(parents=True)
                    with (model / "training_metrics.pkl").open("wb") as handle:
                        pickle.dump({"train_optimizer_steps": [40] * 300}, handle)
                    for repeat in (1, 2):
                        folder = root / "evaluation" / f"cluster{index:02d}_{experiment.slug(cluster)}_poses8_seed{seed}_rep{repeat}"
                        write_scores(folder, names, 2)
            with redirect_stdout(io.StringIO()):
                experiment.summarize(manifest, root, (101, 202), 8, 2)
            result = json.loads((root / "summary" / "results.json").read_text())
            self.assertEqual(result["denominator"], 340)
            self.assertEqual(result["baseline_hits"], 32)
            self.assertEqual(result["trained_hits"], 64)


if __name__ == "__main__":
    unittest.main()
