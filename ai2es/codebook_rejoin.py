"""
Remerge sidelined data from one labeler into
dataset from 4 labelers after being certified
"""

import os
import shutil
from typing import Optional

from joblib import Parallel, delayed


def find_file(filename: str, search_path: str) -> bool:
    """Return True if the given filename is found in the given search path

    Args:
        filename: The name of the file
        search_path: The absolute path
    """
    return any(filename in filenames for _, _, filenames in os.walk(search_path))


def main(class_: str) -> None:
    """Copy the originally labeled image from src into codebook dataset if not already there

    Args:
        class_: The name of the class
    """
    print(class_)
    # original labeled images
    src = "/home/vanessa/hulk/ai2es/night_precip_hand_labeled/2017"
    # where to merge the labeled images
    dest = "/home/vanessa/hulk/ai2es/codebook_dataset/combined_extra"
    files = os.listdir(os.path.join(src, class_))
    for filename in files:
        if not find_file(filename, dest):
            shutil.copy(
                os.path.join(src, class_, filename),
                os.path.join(dest, class_, filename),
            )


if __name__ == "__main__":
    Parallel(n_jobs=3)(
        delayed(main)(class_) for class_ in ["no_precip", "obstructed", "precip"]
    )
