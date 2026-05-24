#!/usr/bin/env python3
"""Export a CAREamics N2V2 checkpoint to the fixed-tile TorchScript model used by C++."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from careamics import CAREamist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="CAREamics .ckpt file.")
    parser.add_argument("output", type=Path, help="TorchScript .pt output path.")
    parser.add_argument(
        "--tile-size",
        nargs=2,
        type=int,
        default=(256, 256),
        metavar=("HEIGHT", "WIDTH"),
        help="Fixed inference tile size used by the C++ prototype.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    careamist = CAREamist(checkpoint_path=args.checkpoint, enable_progress_bar=False)
    model = careamist.model.model.eval().cpu()
    example = torch.zeros(1, 1, args.tile_size[0], args.tile_size[1], dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=False)
        frozen = torch.jit.freeze(traced.eval())

    frozen.save(args.output)

    normalization = careamist.config.data_config.normalization
    metadata = {
        "checkpoint": str(args.checkpoint),
        "torchscript": str(args.output),
        "tile_size": list(args.tile_size),
        "careamics_mean": float(normalization.input_means[0]),
        "careamics_std": float(normalization.input_stds[0]),
        "model": dict(careamist.config.algorithm_config.model),
        "note": "The C++ prototype feeds fixed-size tiles and applies CAREamics mean/std normalization outside the model.",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote TorchScript model: {args.output}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
