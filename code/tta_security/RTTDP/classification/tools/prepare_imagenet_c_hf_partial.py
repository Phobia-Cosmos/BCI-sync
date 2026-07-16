#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a partial ImageNet-C split from WNJXYK/TTA-ImageNet-C for RTTDP."
    )
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep one streamed example every N examples to cover more classes from class-sorted parquet files.",
    )
    parser.add_argument("--repo", default="WNJXYK/TTA-ImageNet-C")
    parser.add_argument("--data-root", default="data/ImageNet-C")
    parser.add_argument(
        "--class-map",
        default="robustbench/data/imagenet_class_to_id_map.json",
    )
    parser.add_argument(
        "--list-path",
        default=None,
        help="Output image-id list path. Defaults to robustbench/data/imagenet_test_image_ids_hf_<corruption>_<severity>_<num>.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_root = Path(__file__).resolve().parents[1]
    sys.path = [
        path for path in sys.path
        if Path(path or os.getcwd()).resolve() != script_root
    ]
    from datasets import load_dataset

    with open(args.class_map, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    list_path = args.list_path
    if list_path is None:
        list_path = (
            f"robustbench/data/imagenet_test_image_ids_hf_"
            f"{args.corruption}_{args.severity}_{args.num_examples}.txt"
        )

    out_root = Path(args.data_root) / args.corruption / str(args.severity)
    Path(list_path).parent.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        args.repo,
        data_dir=f"data/{args.corruption}/severity_{args.severity}",
        split="train",
        streaming=True,
    )

    rel_paths = []
    per_class_counts = {}
    for stream_idx, example in enumerate(ds):
        if len(rel_paths) >= args.num_examples:
            break
        if stream_idx % args.stride != 0:
            continue

        label = int(example["label"])
        wnid = idx_to_class[label]
        per_class_counts[wnid] = per_class_counts.get(wnid, 0) + 1
        rel_path = f"{wnid}/hf_{args.corruption}_{args.severity}_{len(rel_paths):06d}.JPEG"
        abs_path = out_root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        image = example["image"].convert("RGB")
        image.save(abs_path, format="JPEG", quality=95)
        rel_paths.append(rel_path)

        if len(rel_paths) % 100 == 0:
            print(f"saved {len(rel_paths)} images", flush=True)

    with open(list_path, "w") as f:
        for rel_path in rel_paths:
            f.write(rel_path + "\n")

    print(f"saved_images={len(rel_paths)}")
    print(f"data_dir={out_root}")
    print(f"list_path={list_path}")
    sys.stdout.flush()

    # Avoid a pyarrow/datasets shutdown crash observed in this environment.
    os._exit(0)


if __name__ == "__main__":
    main()
