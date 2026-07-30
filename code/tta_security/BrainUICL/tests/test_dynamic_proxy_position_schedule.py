import unittest

from experiments.dynamic_proxy_position_schedule import (
    DATASET_TASKS,
    clean_tasks,
    manifest,
    proxy_tasks,
    schedule,
)


class DynamicProxyPositionScheduleTest(unittest.TestCase):
    def test_front_middle_tail_k10_isruc(self) -> None:
        self.assertEqual(proxy_tasks(49, 10, "front"), list(range(1, 20, 2)))
        self.assertEqual(proxy_tasks(49, 10, "middle"), list(range(16, 35, 2)))
        self.assertEqual(proxy_tasks(49, 10, "tail"), list(range(31, 50, 2)))

    def test_front_middle_tail_k20_faced(self) -> None:
        self.assertEqual(proxy_tasks(61, 20, "front"), list(range(1, 40, 2)))
        self.assertEqual(proxy_tasks(61, 20, "middle"), list(range(12, 51, 2)))
        self.assertEqual(proxy_tasks(61, 20, "tail"), list(range(23, 62, 2)))

    def test_every_schedule_is_disjoint_complete_and_interleaved(self) -> None:
        for dataset, total_tasks in DATASET_TASKS.items():
            for strength in (10, 20):
                for placement in ("front", "middle", "tail"):
                    with self.subTest(
                        dataset=dataset, strength=strength, placement=placement
                    ):
                        row = schedule(dataset, strength, placement)
                        selected = row["proxy_tasks"]
                        clean = row["clean_feedback_tasks"]
                        self.assertEqual(len(selected), strength)
                        self.assertTrue(set(selected).isdisjoint(clean))
                        self.assertEqual(
                            sorted(selected + clean),
                            list(range(1, total_tasks + 1)),
                        )
                        self.assertTrue(
                            all(
                                right - left == 2
                                for left, right in zip(selected, selected[1:])
                            )
                        )
                        self.assertEqual(clean, clean_tasks(total_tasks, selected))

    def test_manifest_contains_all_twelve_dataset_schedules(self) -> None:
        payload = manifest()
        self.assertEqual(set(payload["datasets"]), {"ISRUC", "FACED"})
        self.assertTrue(
            all(len(rows) == 6 for rows in payload["datasets"].values())
        )


if __name__ == "__main__":
    unittest.main()
