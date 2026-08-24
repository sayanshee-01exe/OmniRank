#!/usr/bin/env python
"""Download the PixelRec50K dataset.

    python scripts/download_pixelrec50k.py                  # 51 MB, the two CSVs
    python scripts/download_pixelrec50k.py --with-features   # + 17.3 GB of vectors
    python scripts/download_pixelrec50k.py --dry-run         # show plan, download nothing

Source: https://github.com/westlake-repl/PixelRec
PixelRec50K folder:
https://drive.google.com/drive/folders/1bQPgM-6yAnzcD0jKBoUUheA9LL5xnCHG

LICENCE - read before running. The dataset is provided by the Westlake
Representation Learning Lab **exclusively for non-commercial research and
educational purposes**. No rights are granted to copy, modify, publish,
distribute, or commercialise it, and offering modified secondary downloads is
explicitly prohibited. Running this script downloads it to your machine only;
never commit it, and never redistribute it.

This script never downloads the full PixelRec dataset. Only the four file ids
below are reachable from it, and the two large feature files require an explicit
flag.

If Google Drive rate-limits the download - which it does for large files - the
script says so and prints the manual steps rather than retrying silently.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_TARGET = Path("data/raw/pixelrec50k")

MISSING_DEPENDENCY_EXIT = 2
DOWNLOAD_FAILED_EXIT = 4


@dataclass(frozen=True)
class RemoteFile:
    """One downloadable artifact."""

    name: str
    drive_id: str
    approx_bytes: int
    description: str
    optional: bool = False


#: Verified against the official folder on 2026-08-24. Sizes for the two CSVs
#: are exact; the feature files are as reported by Drive.
FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        name="interaction.csv",
        drive_id="1JvYdOBP76J6oymsV395z94JVcTx_PrDQ",
        approx_bytes=28_124_439,
        description="989,494 interactions - item_id, user_id, timestamp",
    ),
    RemoteFile(
        name="item_info.csv",
        drive_id="1hEBbBOq-3FACcZN_Tzu7RiLguZFhYROJ",
        approx_bytes=24_973_166,
        description="82,865 items - engagement counters, title, tag, description",
    ),
    RemoteFile(
        name="text_feature.json",
        drive_id="1t1ZknzSY-8KxhhfTWMORh66BOdV7qmCj",
        approx_bytes=9_290_203_646,
        description="1024-d text vectors for ALL 408,374 full-PixelRec items",
        optional=True,
    ),
    RemoteFile(
        name="image_feature.json",
        drive_id="12VW6o5AToMFWLbSILi5_c6tlSXS43qm6",
        approx_bytes=9_235_894_623,
        description="1024-d image vectors for ALL 408,374 full-PixelRec items",
        optional=True,
    ),
)

MANUAL_INSTRUCTIONS = """
Automatic download failed. Google Drive rate-limits large files and sometimes
requires an interactive confirmation. Download manually instead:

  1. Open https://drive.google.com/drive/folders/1bQPgM-6yAnzcD0jKBoUUheA9LL5xnCHG
  2. Download `interaction.csv` and `item_info.csv`.
  3. Place both in {target}/
  4. Re-run this script to verify sizes and write checksums, or go straight to:
       python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml

Optional multimodal vectors (17.3 GB total, only if you need them):
  text  https://drive.google.com/file/d/1t1ZknzSY-8KxhhfTWMORh66BOdV7qmCj/view
  image https://drive.google.com/file/d/12VW6o5AToMFWLbSILi5_c6tlSXS43qm6/view
"""


def human_bytes(count: int) -> str:
    """Format a byte count for a human reader."""
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def sha256_of(path: Path) -> str:
    """Checksum a downloaded file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download PixelRec50K into data/raw/pixelrec50k.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--target", type=Path, default=DEFAULT_TARGET, help="Destination directory."
    )
    parser.add_argument(
        "--with-features",
        action="store_true",
        help="Also download the two ~8.6 GB multimodal feature files (17.3 GB total).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist. Off by default.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan and exit without downloading."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    target: Path = args.target
    selected = [file for file in FILES if not file.optional or args.with_features]
    total = sum(file.approx_bytes for file in selected)

    print("PixelRec50K download")
    print("  source : https://github.com/westlake-repl/PixelRec")
    print(f"  target : {target}")
    print("  licence: non-commercial research/education only; do not redistribute.")
    print()
    print(f"  {'file':<22} {'size':>10}  description")
    for file in selected:
        print(f"  {file.name:<22} {human_bytes(file.approx_bytes):>10}  {file.description}")
    print(f"  {'TOTAL':<22} {human_bytes(total):>10}")
    print()

    if not args.with_features:
        print("  Multimodal vectors are NOT included (add --with-features for 17.3 GB more).")
        print("  The pipeline runs without them and reports feature coverage as 0.0.")
        print()

    if args.dry_run:
        print("Dry run: nothing downloaded.")
        return 0

    try:
        import gdown
    except ImportError:
        print(
            "gdown is required for automatic download. Install it with:\n"
            "  uv pip install gdown\n"
            f"{MANUAL_INSTRUCTIONS.format(target=target)}",
            file=sys.stderr,
        )
        return MISSING_DEPENDENCY_EXIT

    target.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for file in selected:
        destination = target / file.name
        if destination.exists() and not args.overwrite:
            size = human_bytes(destination.stat().st_size)
            print(f"  {file.name}: already present ({size}) - skipping")
            continue
        print(f"  {file.name}: downloading {human_bytes(file.approx_bytes)} …")
        try:
            # gdown resumes partial downloads when the destination exists.
            gdown.download(id=file.drive_id, output=str(destination), quiet=False, resume=True)
        except Exception as exc:
            failures.append(f"{file.name}: {type(exc).__name__}: {exc}")
            continue

        if not destination.exists() or destination.stat().st_size == 0:
            failures.append(f"{file.name}: download produced no data")
            continue

        actual = destination.stat().st_size
        # A Drive rate-limit page is a few KB of HTML saved under the right name;
        # a size sanity check is what catches that.
        if actual < file.approx_bytes * 0.5:
            failures.append(
                f"{file.name}: got {human_bytes(actual)}, expected about "
                f"{human_bytes(file.approx_bytes)} - likely a rate-limit page, not the file"
            )
            continue
        print(f"    ok: {human_bytes(actual)}  sha256={sha256_of(destination)[:16]}…")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(MANUAL_INSTRUCTIONS.format(target=target), file=sys.stderr)
        return DOWNLOAD_FAILED_EXIT

    print("\nDone. Next:")
    print("  python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
