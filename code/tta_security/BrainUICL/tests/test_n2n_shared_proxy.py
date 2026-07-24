from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.n2n_shared_proxy import (
    make_task_entry,
    resolve_task,
    validate_manifest,
    write_manifest,
)
from experiments.generate_n2n_shared_proxy_manifest import parse_affected_tasks
from experiments.summarize_canonical_n2n_matrix import validate_paired_prefix


class CanonicalN2NManifestTests(unittest.TestCase):
    def test_multi_user_task_parser_is_sorted_unique_and_backward_compatible(self):
        self.assertEqual(parse_affected_tasks("31,26,31,49", 10), (26, 31, 49))
        self.assertEqual(parse_affected_tasks("", 26), (26,))
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_affected_tasks("0,26", 10)

    def test_paired_prefix_validation_ignores_metadata_but_rejects_drift(self):
        clean = {
            "initial": {"old": 0.5},
            "tasks": [
                {"task": 1, "value": 0.4, "attack": None},
                {"task": 2, "value": 0.3, "attack": None},
                {"task": 3, "value": 0.2, "attack": None},
            ],
        }
        shared = {
            "initial": {"old": 0.5},
            "tasks": [
                {"task": 1, "value": 0.4, "noise": {"proxy_sequences": 0}},
                {"task": 2, "value": 0.3, "noise": {"proxy_sequences": 0}},
                {"task": 3, "value": 0.1, "noise": {"proxy_sequences": 1}},
            ],
        }

        report = validate_paired_prefix(
            clean,
            shared,
            affected_task=3,
            expected_tasks=3,
            label="fixture",
        )
        self.assertEqual(report["tasks_compared"], 2)

        shared["tasks"][0]["value"] = 0.39
        with self.assertRaisesRegex(ValueError, "diverge before affected task"):
            validate_paired_prefix(
                clean,
                shared,
                affected_task=3,
                expected_tasks=3,
                label="fixture",
            )

    def _fixture(self, root: Path):
        clean_tasks: dict[int, tuple[int, list[Path]]] = {}
        for task, subject in ((1, 64), (2, 89)):
            paths = []
            directory = root / "clean" / str(subject)
            directory.mkdir(parents=True)
            for index in range(3):
                path = directory / f"{index}.npy"
                value = np.full((2, 3, 4), task + index + 1, dtype=np.float32)
                value[:, :, 1::2] *= -1
                np.save(path, value)
                paths.append(path)
            clean_tasks[task] = subject, paths

        proxy_path = root / "payload" / "task_2" / "1.npy"
        proxy_path.parent.mkdir(parents=True)
        proxy = np.load(clean_tasks[2][1][1]).copy()
        proxy += 0.05
        np.save(proxy_path, proxy)

        entries = [
            make_task_entry(
                task=1,
                subject=64,
                clean_paths=clean_tasks[1][1],
                proxy_paths={},
            ),
            make_task_entry(
                task=2,
                subject=89,
                clean_paths=clean_tasks[2][1],
                proxy_paths={1: proxy_path},
            ),
        ]
        manifest_path = root / "manifest.json"
        write_manifest(
            manifest_path,
            tasks=entries,
            split={"new_order": [64, 89]},
            constraints={
                "upload_multiplier": 1,
                "repeat": 0,
                "max_relative_l2": 0.1,
                "max_linf_over_std": 0.1,
            },
            provenance={"surrogate": "unit-test"},
        )
        return manifest_path, clean_tasks, proxy_path

    def test_partial_resolver_preserves_cardinality_order_and_clean_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, tasks, proxy_path = self._fixture(Path(directory))
            clean_paths = tasks[2][1]
            label_paths = [Path(f"label-{index}.npy") for index in range(3)]
            labels_before = tuple(label_paths)

            resolved = resolve_task(
                manifest,
                task=2,
                subject=89,
                clean_data_paths=clean_paths,
                verify="selected",
            )

            self.assertEqual(len(resolved.data_paths), len(clean_paths))
            self.assertEqual(resolved.proxy_indices, (1,))
            self.assertEqual(resolved.data_paths[0], clean_paths[0].resolve())
            self.assertEqual(resolved.data_paths[1], proxy_path.resolve())
            self.assertEqual(resolved.data_paths[2], clean_paths[2].resolve())
            self.assertEqual(tuple(label_paths), labels_before)
            self.assertTrue(resolved.diagnostics["n_to_n"])
            self.assertEqual(resolved.diagnostics["repeat"], 0)

    def test_resolver_interface_cannot_receive_target_labels(self):
        parameters = set(inspect.signature(resolve_task).parameters)
        self.assertNotIn("labels", parameters)
        self.assertNotIn("label_paths", parameters)
        self.assertNotIn("true_labels", parameters)

    def test_full_validation_covers_every_task_and_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, tasks, _proxy_path = self._fixture(Path(directory))
            report = validate_manifest(manifest, tasks)
            self.assertEqual(report["tasks"], 2)
            self.assertEqual(report["uploaded_sequences"], 6)
            self.assertEqual(report["proxy_sequences"], 1)
            self.assertEqual(report["proxy_tasks"], [2])
            self.assertTrue(report["all_slots_validated"])

    def test_selected_payload_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, tasks, proxy_path = self._fixture(Path(directory))
            proxy = np.load(proxy_path)
            proxy[0, 0, 0] += 0.01
            np.save(proxy_path, proxy)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                resolve_task(
                    manifest,
                    task=2,
                    subject=89,
                    clean_data_paths=tasks[2][1],
                    verify="selected",
                )

    def test_wrong_subject_and_slot_order_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, tasks, _proxy_path = self._fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "subject mismatch"):
                resolve_task(
                    manifest,
                    task=2,
                    subject=90,
                    clean_data_paths=tasks[2][1],
                )
            payload = json.loads(manifest.read_text())
            payload["tasks"][1]["slots"][0]["slot"] = 1
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "slot ordering"):
                resolve_task(
                    manifest,
                    task=2,
                    subject=89,
                    clean_data_paths=tasks[2][1],
                    verify="none",
                )

    def test_recorded_budget_is_enforced_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, tasks, _proxy_path = self._fixture(Path(directory))
            payload = json.loads(manifest.read_text())
            payload["constraints"]["max_relative_l2"] = 1e-5
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "exceeds canonical budget"):
                resolve_task(
                    manifest,
                    task=2,
                    subject=89,
                    clean_data_paths=tasks[2][1],
                    verify="none",
                )


if __name__ == "__main__":
    unittest.main()
