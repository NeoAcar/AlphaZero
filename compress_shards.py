"""
Convert sharded supervised data from float32 boards to uint8.

For each shard_*.pt found under --input-dir (recursively), reads the shard,
clamps board values to [0, 1], scales by 255, casts to uint8, and writes
the converted shard atomically to the same path (or to --output-dir if
given).

Storage shrinks 4x. ChessDataset/SelfPlayDataset detect uint8 boards at
__getitem__ time and convert back via `.float() / 255.0` -- no model
changes needed.

Values that the encoder maps to > 1.0 (e.g. very long games where
move_counter/500 exceeds 1) are clipped. The lost precision is the
0.4% quantization step (1/256) per cell, which has no measurable
training impact.

Usage:
    uv run python compress_shards.py --input-dir data/sf_shards
    uv run python compress_shards.py --input-dir data/sf_shards --output-dir data/sf_shards_u8
"""
import argparse
import os
import shutil
from pathlib import Path

import torch
from tqdm import tqdm  # type: ignore


def convert_shard(in_path: Path, out_path: Path) -> tuple[int, int]:
    """Returns (bytes_before, bytes_after)."""
    bytes_before = in_path.stat().st_size

    d = torch.load(in_path, map_location="cpu", weights_only=False)
    boards = d["boards"]
    if boards.dtype != torch.float32:
        # Already converted; copy if writing to different dir, else skip.
        if in_path != out_path:
            shutil.copy(in_path, out_path)
        return bytes_before, out_path.stat().st_size

    # Clamp + scale + cast.
    boards_u8 = torch.clamp(boards, 0.0, 1.0).mul(255.0).round().to(torch.uint8)
    d["boards"] = boards_u8

    # Mark the new format so downstream code / future tools know.
    meta = d.get("meta", {}) or {}
    meta["boards_dtype"] = "uint8"
    meta["boards_scale"] = 255
    d["meta"] = meta

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, out_path)
    return bytes_before, out_path.stat().st_size


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True,
                   help="directory to scan recursively for shard_*.pt files")
    p.add_argument("--output-dir", default=None,
                   help="if given, write converted shards here (preserving subdir "
                        "structure); otherwise overwrite in place")
    cli = p.parse_args()

    in_root = Path(cli.input_dir)
    out_root = Path(cli.output_dir) if cli.output_dir else in_root

    shard_paths = sorted(in_root.rglob("shard_*.pt"))
    if not shard_paths:
        raise SystemExit(f"No shard_*.pt files found under {in_root}")
    print(f"Found {len(shard_paths)} shards under {in_root}; "
          f"writing to {out_root}")

    total_before = 0
    total_after = 0
    pbar = tqdm(shard_paths, unit="shard")
    for shard in pbar:
        rel = shard.relative_to(in_root)
        out_path = out_root / rel
        before, after = convert_shard(shard, out_path)
        total_before += before
        total_after += after
        pbar.set_postfix(saved_gb=f"{(total_before - total_after) / 1e9:.1f}",
                         ratio=f"{total_after / max(total_before, 1):.2f}")
    pbar.close()

    print(f"\nDone: {total_before / 1e9:.1f} GB -> {total_after / 1e9:.1f} GB "
          f"({(1 - total_after / max(total_before, 1)) * 100:.0f}% reduction)")


if __name__ == "__main__":
    main()
