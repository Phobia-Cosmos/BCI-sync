from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.replay_cl_eeg import (
    ReplayRecord,
    ReservoirReplayMemory,
    UnlabeledSequenceDataset,
    build_memory_records,
    load_replay_batch,
    resolve_n2n_replay_uploads,
)
from experiments.n2n_shared_proxy import make_task_entry, write_manifest


def record(index: int, poisoned: bool = False) -> ReplayRecord:
    return ReplayRecord(
        data_path=Path(f"{index}.npy"),
        pseudo_labels=np.full(20, index % 5, dtype=np.int64),
        task=1,
        subject=64,
        sequence_index=index,
        poisoned=poisoned,
        repeated_upload=False,
    )


class ReservoirReplayMemoryTests(unittest.TestCase):
    def test_canonical_n2n_marks_only_selected_signal_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_paths = []
            for index in range(3):
                path = root / "clean" / f"{index}.npy"
                path.parent.mkdir(exist_ok=True)
                np.save(path, np.full((2, 3, 4), index + 1, dtype=np.float32))
                clean_paths.append(path)
            proxy_path = root / "payload" / "different-name.npy"
            proxy_path.parent.mkdir()
            np.save(proxy_path, np.load(clean_paths[2]) + np.float32(0.01))
            manifest = root / "manifest.json"
            write_manifest(
                manifest,
                tasks=[
                    make_task_entry(
                        task=1,
                        subject=64,
                        clean_paths=clean_paths,
                        proxy_paths={2: proxy_path},
                    )
                ],
                split={"new_order": [64]},
                constraints={"repeat": 0, "upload_multiplier": 1},
                provenance={"surrogate": "unit-test"},
            )
            args = SimpleNamespace(n2n_manifest=manifest, n2n_verify="selected")

            paths, tracked, indices, diagnostics = resolve_n2n_replay_uploads(
                args, 1, 64, clean_paths
            )

            self.assertEqual(indices, (2,))
            self.assertEqual(tracked, {str(proxy_path.resolve())})
            self.assertEqual(paths[:2], [path.resolve() for path in clean_paths[:2]])
            self.assertEqual(paths[2], proxy_path.resolve())
            self.assertTrue(diagnostics["n_to_n"])

    def test_explicit_sequence_indices_do_not_depend_on_proxy_basename(self):
        paths = [Path("proxy-a.npy"), Path("proxy-b.npy")]
        labels = [np.zeros(20, dtype=np.int64) for _ in paths]
        records = build_memory_records(
            paths,
            labels,
            task_index=1,
            subject=64,
            original_count=2,
            poisoned_paths={"proxy-b.npy"},
            sequence_indices=[7, 11],
        )
        self.assertEqual([item.sequence_index for item in records], [7, 11])
        self.assertEqual([item.poisoned for item in records], [False, True])

    def test_reservoir_is_fixed_capacity_and_tracks_poisoned_replay(self):
        memory = ReservoirReplayMemory(capacity=3, seed=7)
        update = memory.add([record(index, poisoned=index % 2 == 0) for index in range(10)])
        self.assertEqual(len(memory), 3)
        self.assertEqual(update["total_seen"], 10)
        sampled = memory.sample(6)
        self.assertEqual(len(sampled), 6)
        self.assertEqual(memory.stats()["total_replay_draws"], 6)
        self.assertEqual(
            memory.stats()["poisoned_replay_draws"],
            sum(int(item.poisoned) for item in sampled),
        )

    def test_volume_matched_records_preserve_repeated_and_poison_flags(self):
        paths = [Path("0.npy"), Path("1.npy"), Path("0.npy"), Path("1.npy")]
        labels = [np.zeros(20, dtype=np.int64) for _ in paths]
        records = build_memory_records(
            paths,
            labels,
            task_index=3,
            subject=1,
            original_count=2,
            poisoned_paths={"0.npy", "1.npy"},
        )
        self.assertEqual(len(records), 4)
        self.assertEqual(sum(item.repeated_upload for item in records), 2)
        self.assertTrue(all(item.poisoned for item in records))

    def test_unlabeled_loader_and_replay_batch_use_signal_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.zeros((20, 8, 3000), dtype=np.float32)
            data_path = root / "0.npy"
            np.save(data_path, values)
            dataset = UnlabeledSequenceDataset([data_path], sequence_length=20)
            eog, eeg, dummy = dataset[0]
            self.assertEqual(tuple(eog.shape), (20, 2, 3000))
            self.assertEqual(tuple(eeg.shape), (20, 6, 3000))
            self.assertTrue(dummy.eq(0).all())

            replay = ReplayRecord(
                data_path=data_path,
                pseudo_labels=np.arange(20, dtype=np.int64) % 5,
                task=1,
                subject=64,
                sequence_index=0,
                poisoned=False,
                repeated_upload=False,
            )
            replay_eog, replay_eeg, replay_labels = load_replay_batch([replay])
            self.assertEqual(tuple(replay_eog.shape), (1, 20, 2, 3000))
            self.assertEqual(tuple(replay_eeg.shape), (1, 20, 6, 3000))
            self.assertEqual(tuple(replay_labels.shape), (1, 20))


if __name__ == "__main__":
    unittest.main()
