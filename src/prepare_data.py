"""
prepare_data.py
================
Copies a balanced subset of the "140K Real and Fake Faces" dataset into the
project's data/ directory, ready for training.

Source layout (Kaggle archive):
    <source_dir>/
        real_vs_fake/
            real-vs-fake/
                train/  real/ fake/
                valid/  real/ fake/
                test/   real/ fake/

Destination layout:
    data/
        train/  real/ fake/
        val/    real/ fake/
        test/   real/ fake/

Split ratios  : 70 % train  /  15 % val  /  15 % test
Default total : 30 000 images  (15 000 real + 15 000 fake)

Usage
-----
    # From the project root:
    python src/prepare_data.py

    # Custom source / destination:
    python src/prepare_data.py --source "C:/path/to/archive" --dest data

    # Different subset size:
    python src/prepare_data.py --total 20000
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────

# Override with --source flag, or set this to your local Kaggle archive path.
# e.g. Path(r"C:\Users\YourName\Downloads\archive") or Path("/home/user/archive")
DEFAULT_SOURCE = None
DEFAULT_DEST   = Path(__file__).resolve().parent.parent / "data"

# Inner path inside the archive where the images live
INNER_PATH     = Path("real_vs_fake") / "real-vs-fake"

# Kaggle source split names -> our destination split names
SOURCE_SPLITS  = ["train", "valid", "test"]      # all three are pooled together
DEST_SPLITS    = ["train", "val", "test"]
SPLIT_RATIOS   = [0.70, 0.15, 0.15]

VALID_EXTS     = {".jpg", ".jpeg", ".png", ".webp"}


# ── Helpers ────────────────────────────────────────────────────────────────

def collect_images(base: Path, class_name: str) -> list[Path]:
    """
    Gather all image paths under every '<split>/<class_name>/' folder
    inside `base`.  Returns a flat, shuffled list.
    """
    images: list[Path] = []
    for split in SOURCE_SPLITS:
        folder = base / split / class_name
        if not folder.is_dir():
            print(f"  [warn] Folder not found, skipping: {folder}")
            continue
        found = [p for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() in VALID_EXTS]
        images.extend(found)
    return images


def split_list(items: list, ratios: list[float]) -> list[list]:
    """
    Partition `items` into len(ratios) groups according to the given ratios.
    The last group absorbs any rounding remainder.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    n      = len(items)
    groups = []
    start  = 0
    for i, r in enumerate(ratios[:-1]):
        end = start + round(n * r)
        groups.append(items[start:end])
        start = end
    groups.append(items[start:])   # last group gets the remainder
    return groups


def copy_file(src: Path, dst: Path) -> bool:
    """
    Copy `src` to `dst`.  Returns True if the file was newly copied,
    False if it already existed (idempotent — skips duplicates).
    """
    if dst.exists():
        return False          # already there from a previous run
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)    # copy2 preserves file metadata
    return True


# ── Main logic ─────────────────────────────────────────────────────────────

def prepare(source_dir: Path,
            dest_dir:   Path,
            total:      int  = 30_000,
            seed:       int  = 42) -> None:
    """
    Core preparation routine — additive strategy.

    If dest folders already contain images from a previous run (e.g. the old
    12 000-image setup), only the *missing* images are added to reach the new
    per-folder targets.  No existing file is ever deleted or moved between
    splits, so train / val / test remain permanently disjoint.

    Why additive?
    -------------
    A naive re-split (e.g. bumping per_class from 6 000 to 15 000 and
    reapplying 70/15/15) would silently reassign ~1 800 images per class from
    val/test into train, causing data leakage — those files already live in
    val/ but would also be copied into train/.  The additive approach avoids
    this: each folder only ever grows, never shrinks or swaps images.

    Args:
        source_dir : Root of the Kaggle archive (contains real_vs_fake/)
        dest_dir   : Destination data/ folder for the project
        total      : Total images on disk when done (balanced: total//2 per class)
        seed       : Random seed for reproducible sub-sampling
    """
    rng = random.Random(seed)

    inner = source_dir / INNER_PATH
    if not inner.is_dir():
        print(f"\n[ERROR] Expected folder not found:\n  {inner}")
        print("Please verify --source points to the extracted Kaggle archive.")
        sys.exit(1)

    if total % 2 != 0:
        total -= 1
        print(f"[info] total adjusted to {total} to keep classes balanced.")

    per_class = total // 2    # 15 000

    print(f"\n{'='*55}")
    print(f"  Source : {source_dir}")
    print(f"  Dest   : {dest_dir}")
    print(f"  Total  : {total}  ({per_class} real + {per_class} fake)")
    print(f"  Split  : train {SPLIT_RATIOS[0]:.0%}  "
          f"val {SPLIT_RATIOS[1]:.0%}  test {SPLIT_RATIOS[2]:.0%}")
    print(f"  Seed   : {seed}")
    print(f"{'='*55}\n")

    # ── Step 1: Scan source + audit what is already in dest ────────────────
    print("[1/3] Scanning source folders and auditing existing data ...")

    # Stores all per-class data needed for steps 2 and 3
    per_cls: dict[str, dict] = {}

    for cls in ["real", "fake"]:
        # Deterministic shuffle — same seed produces the same order every run
        all_imgs = collect_images(inner, cls)
        print(f"  [{cls}] {len(all_imgs):>6} source images found.")
        if len(all_imgs) < per_class:
            print(f"  [ERROR] Not enough {cls} images "
                  f"(need {per_class}, found {len(all_imgs)}).")
            sys.exit(1)
        rng.shuffle(all_imgs)
        selected = all_imgs[:per_class]   # always the same 15 000 images

        # Per-folder target counts derived from the full 15 000 split
        target_groups = split_list(selected, SPLIT_RATIOS)
        targets = {s: len(g) for s, g in zip(DEST_SPLITS, target_groups)}
        # e.g. {train: 10500, val: 2250, test: 2250}

        # Snapshot of filenames already on disk in every dest folder
        existing_by_folder: dict[str, set[str]] = {}
        all_existing_names: set[str] = set()
        for split_name in DEST_SPLITS:
            folder = dest_dir / split_name / cls
            names: set[str] = set()
            if folder.is_dir():
                names = {
                    p.name for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in VALID_EXTS
                }
            existing_by_folder[split_name] = names
            all_existing_names |= names   # union across ALL splits

        # How many more images each folder still needs
        deficits = {
            s: max(0, targets[s] - len(existing_by_folder[s]))
            for s in DEST_SPLITS
        }
        total_deficit = sum(deficits.values())

        # Images from the selected pool not yet present in any dest folder.
        # Using the union set ensures we never place the same image in two
        # different splits — train/val/test stay disjoint.
        new_pool = [p for p in selected if p.name not in all_existing_names]

        if len(new_pool) < total_deficit:
            print(
                f"  [ERROR] Only {len(new_pool)} unused {cls} images available "
                f"but {total_deficit} are needed.  "
                f"Try reducing --total or check source data."
            )
            sys.exit(1)

        per_cls[cls] = {
            "targets"            : targets,
            "existing_by_folder" : existing_by_folder,
            "deficits"           : deficits,
            "new_pool"           : new_pool,
        }

    # ── Step 2: Show what will be added ───────────────────────────────────
    print("\n[2/3] Additions needed per folder ...")
    for split_name in DEST_SPLITS:
        for cls in ["real", "fake"]:
            d       = per_cls[cls]
            existed = len(d["existing_by_folder"][split_name])
            to_add  = d["deficits"][split_name]
            print(
                f"  {split_name:5} / {cls:4}  "
                f"target={d['targets'][split_name]:6}  "
                f"existing={existed:6}  "
                f"to_add={to_add:6}"
            )

    # ── Step 3: Copy only the new images ──────────────────────────────────
    print("\n[3/3] Copying new images ...")
    grand_total_copied  = 0
    grand_total_existed = 0
    summary: dict[str, dict[str, dict]] = {}

    for cls in ["real", "fake"]:
        d        = per_cls[cls]
        pool_idx = 0   # walks through new_pool in order: train → val → test

        for split_name in DEST_SPLITS:
            need        = d["deficits"][split_name]
            batch       = d["new_pool"][pool_idx : pool_idx + need]
            pool_idx   += need
            dest_folder = dest_dir / split_name / cls
            copied = skipped = 0

            for src_path in batch:
                dst_path = dest_folder / src_path.name
                if copy_file(src_path, dst_path):
                    copied  += 1
                else:
                    # Safety-net: copy_file returns False only if dst already
                    # exists.  Should never happen here (new_pool excludes them).
                    skipped += 1

            existed = len(d["existing_by_folder"][split_name])
            on_disk = existed + copied
            summary.setdefault(split_name, {})[cls] = {
                "target" : d["targets"][split_name],
                "existed": existed,
                "copied" : copied,
                "on_disk": on_disk,
            }
            grand_total_copied  += copied
            grand_total_existed += existed

            tag = f"added {copied}" if copied > 0 else "already at target"
            print(
                f"  {split_name:5} / {cls:4}  "
                f"added={copied:6}  on_disk={on_disk:6}  [{tag}]"
            )

    # ── Final summary ──────────────────────────────────────────────────────
    grand_total = grand_total_copied + grand_total_existed
    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Newly added    : {grand_total_copied}")
    print(f"  Already existed: {grand_total_existed}")
    print(f"  Total on disk  : {grand_total}")
    print(f"{'='*55}")

    print("\n  Final per-folder counts:")
    print(f"  {'Split':<6}  {'Class':<4}  {'Target':>7}  {'On disk':>7}")
    print(f"  {'-'*35}")
    for split_name in DEST_SPLITS:
        for cls in ["real", "fake"]:
            s = summary[split_name][cls]
            ok = "OK" if s["on_disk"] == s["target"] else "WARN"
            print(
                f"  {split_name:<6}  {cls:<4}  "
                f"{s['target']:>7}  {s['on_disk']:>7}  [{ok}]"
            )

    print(f"\n  Data is ready in:  {dest_dir}\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare a balanced subset of the 140K Deepfake dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the extracted Kaggle archive folder.",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Destination data/ folder for the project.",
    )
    p.add_argument(
        "--total",
        type=int,
        default=30_000,
        help="Total images on disk when done (balanced: total//2 per class).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sub-sampling.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.source is None:
        print("\n[ERROR] --source is required.")
        print("  Provide the path to the extracted Kaggle archive, e.g.:")
        print("    python src/prepare_data.py --source /path/to/archive\n")
        sys.exit(1)
    prepare(
        source_dir=args.source.resolve(),
        dest_dir=args.dest.resolve(),
        total=args.total,
        seed=args.seed,
    )
