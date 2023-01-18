"""
Remerge sidelined data from one labeler into 
dataset from 4 labelers after being certified
"""

import os
import shutil
from joblib import Parallel, delayed


def find_file(filename, search_path):
    return any(
        filename in filenames for _, _, filenames in os.walk(search_path)
    )


def main(class_: str):
    print(class_)
    src = "/ai2es/night_precip_hand_labeled/2017"
    dest = "/ai2es/codebook_dataset/combined_extra"
    files = os.listdir(os.path.join(src, class_))
    for filename in files:
        if not find_file(filename, dest):
            shutil.copy(
                os.path.join(src, class_, filename),
                os.path.join(dest, class_, filename),
            )


if __name__ == "__main__":
    Parallel(n_jobs=3)(
        delayed(main)(class_)
        for class_ in ["no_precip", "obstructed", "precip"]
    )
