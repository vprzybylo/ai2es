import os
import cocpit.config as config
from collections import defaultdict
from PIL import Image
import random
import numpy as np


def main(x=150):
    """make sure there are no more than x images per station/class
    from already lableled samples"""

    for class_name in config.CLASS_NAMES:

        files = os.listdir(
            os.path.join(config.DATA_DIR, config.CLASS_NAME_MAP[class_name])
        )
        stids = [file.split("_")[1].split(".")[0] for file in files]
        counts = defaultdict(int)
        savepath = os.path.join(
            "/ai2es/codebook_dataset/", config.CLASS_NAME_MAP[class_name]
        )
        os.makedirs(savepath, exist_ok=True)
        for stn, file in zip(stids, files):
            if counts[stn] < x:
                im = Image.open(
                    os.path.join(
                        config.DATA_DIR,
                        config.CLASS_NAME_MAP[class_name],
                        file,
                    )
                )
                print(os.path.join(savepath, file))
                im.save(os.path.join(savepath, file))
            counts[stn] += 1


def split():
    for class_name in config.CLASS_NAMES:
        files = os.listdir(
            os.path.join(
                "/ai2es/codebook_dataset/", config.CLASS_NAME_MAP[class_name]
            )
        )
        print(class_name)
        random.shuffle(files)
        sub_lists = np.array_split(files, 4)
        names = ["carly", "chris", "mariana", "vanessa"]
        for name, subset in zip(names, sub_lists):
            for file in subset:

                savepath = os.path.join(
                    "/ai2es/codebook_dataset/",
                    name,
                    config.CLASS_NAME_MAP[class_name],
                )
                os.makedirs(savepath, exist_ok=True)
                im = Image.open(
                    os.path.join(
                        config.DATA_DIR,
                        config.CLASS_NAME_MAP[class_name],
                        file,
                    )
                )
                im.save(os.path.join(savepath, file))


split()
