import glob
import os
import re
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import List, Optional
from typing_extensions import assert_never

from .colmap import Dataset
from .colmap_dataparser import ColmapParser, CONSOLE

# ===== Optional hardcoded overrides =====
# Set these to a list of indices (relative to images_test[_factor])
# to directly control which images are evaluated, bypassing test_every/hold and files.
# Example:
# EVAL_IDS_TEST = [5, 15, 25]
# EVAL_IDS_VAL = None
EVAL_IDS_TEST: Optional[List[int]] = [5, 15, 25]
EVAL_IDS_VAL: Optional[List[int]] = None


def _find_files(directory: Path, exts: List[str]) -> List[Path]:
    """Find all files in a directory that have a certain file extension.

    Args:
        directory : The directory to search for files.
        exts :  A list of file extensions to search for. Each file extension should be in the form '*.ext'.

    Returns:
        A list of file paths for all the files that were found. The list is sorted alphabetically.
    """
    if directory.exists() and os.path.isdir(directory):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(directory, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(list(set(files_grabbed)))
        files_grabbed = [Path(f) for f in files_grabbed]
        return files_grabbed
    return []


class DeblurNerfDataset(Dataset):
    """DeblurNerf dataset class."""

    def __init__(
        self,
        parser: ColmapParser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        # find the file named `hold=n` , n is the eval_interval to be recognized
        hold_file = [f for f in os.listdir(parser.data_dir) if f.startswith("hold=")]
        if len(hold_file) == 0:
            print(f"[INFO] defaulting hold={parser.test_every}")
        else:
            parser.test_every = int(hold_file[0].split("=")[-1])
            print(f"[INFO] found hold={parser.test_every}")
        if split == "train" and parser.test_every < 1:
            split = "all"

        super().__init__(parser, split, patch_size, load_depths)

        # "test" for deblur, "val" for novel-view
        if split == "val" and parser.test_every < 1:
            self.indices = []
            return

        if split == "val" or split == "test":
            self.parser = deepcopy(parser)
            if self.parser.factor > 1:
                image_dir_suffix = f"_{self.parser.factor}"
            else:
                image_dir_suffix = ""
            gt_dir = parser.data_dir / ("images_test" + image_dir_suffix)

            if gt_dir.exists():
                # Will find both deblurring & NVS eval image files in `images_test` folder.
                gt_image_paths = _find_files(gt_dir, ["*.png", "*.jpg", "*.JPG", "*.PNG"])
                num_gt_images = len(gt_image_paths)
                indices = np.arange(num_gt_images)
                # Code-level overrides (hardcoded lists at top of file)
                code_test_override = np.array(EVAL_IDS_TEST, dtype=int) if EVAL_IDS_TEST is not None else None
                code_val_override = np.array(EVAL_IDS_VAL, dtype=int) if EVAL_IDS_VAL is not None else None
                # Optional override: user-specified eval IDs via text files
                def _read_eval_ids_from_files(files: List[Path]) -> Optional[np.ndarray]:
                    for fp in files:
                        try:
                            if fp.exists():
                                text = fp.read_text()
                                ids = [int(x) for x in re.findall(r"\d+", text)]
                                if len(ids) > 0:
                                    return np.array(ids, dtype=int)
                        except Exception:
                            # ignore malformed files and continue
                            pass
                    return None

                test_override = _read_eval_ids_from_files([
                    parser.data_dir / "eval_ids_test.txt",
                    gt_dir / "eval_ids_test.txt",
                    parser.data_dir / "eval_ids.txt",
                    gt_dir / "eval_ids.txt",
                ])
                val_override = _read_eval_ids_from_files([
                    parser.data_dir / "eval_ids_val.txt",
                    gt_dir / "eval_ids_val.txt",
                ])
                # clamp overrides to valid range and unique order (preserve input order)
                def _sanitize(ids: Optional[np.ndarray]) -> Optional[np.ndarray]:
                    if ids is None:
                        return None
                    if num_gt_images == 0:
                        return np.array([], dtype=int)
                    ids = np.array([i for i in ids if 0 <= i < num_gt_images], dtype=int)
                    # de-duplicate while preserving order
                    seen = set()
                    clean = []
                    for i in ids.tolist():
                        if i not in seen:
                            seen.add(i)
                            clean.append(i)
                    return np.array(clean, dtype=int)

                # Sanitize overrides
                test_override = _sanitize(test_override)
                val_override = _sanitize(val_override)
                code_test_override = _sanitize(code_test_override)
                code_val_override = _sanitize(code_val_override)

                # Choose indices with precedence:
                # 1) Code-level override EVAL_IDS_TEST/EVAL_IDS_VAL
                # 2) File-based override eval_ids_test.txt/eval_ids_val.txt
                # 3) Default modulo split by test_every
                if split == "test":
                    if code_test_override is not None:
                        self.indices = code_test_override
                        CONSOLE.log(f"[bold yellow]Using hardcoded TEST eval IDs: {self.indices.tolist()}[/bold yellow]")
                    elif test_override is not None:
                        self.indices = test_override
                        CONSOLE.log(f"[bold yellow]Using file-based TEST eval IDs: {self.indices.tolist()}[/bold yellow]")
                    else:
                        self.indices = (
                            indices if parser.test_every < 1 else indices[indices % self.parser.test_every != 0]
                        )
                else:  # split == "val"
                    if code_val_override is not None:
                        self.indices = code_val_override
                        CONSOLE.log(f"[bold yellow]Using hardcoded VAL eval IDs: {self.indices.tolist()}[/bold yellow]")
                    elif val_override is not None:
                        self.indices = val_override
                        CONSOLE.log(f"[bold yellow]Using file-based VAL eval IDs: {self.indices.tolist()}[/bold yellow]")
                    else:
                        self.indices = (
                            indices if parser.test_every < 1 else indices[indices % self.parser.test_every == 0]
                        )
                assert num_gt_images == 0 or num_gt_images == len(self.parser.image_names)
                self.parser.image_paths = gt_image_paths
                self.parser.image_names = [image_path.stem for image_path in gt_image_paths]
            else:
                CONSOLE.log(f"[bold red][WARN] No images found in {gt_dir}.[/bold red]")
                if split == "test":
                    # No deblurring eval images found
                    self.indices = []
                    return
                elif split == "val":
                    # Fallback to original Dataset split. Will find NVS eval image files in `images` folder.
                    self.parser = parser
