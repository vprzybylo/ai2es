"""
Codebook image generation
"""
import os
import random
from collections import defaultdict

import cocpit.config as config
import numpy as np
from PIL import Image


def parse_by_station(x: int = 150) -> None:
    """
    For each class, save up to x images from each site

    Args:
        x (int): The maximum number of images per station to save for labeling
    """

    for class_name in config.CLASS_NAMES:

        files = os.listdir(
            os.path.join(config.DATA_DIR, config.CLASS_NAME_MAP[class_name])
        )
        stids = [file.split("_")[1].split(".")[0] for file in files]
        counts = defaultdict(int)
        savepath = os.path.join(
            "/home/vanessa/hulk/ai2es/codebook_dataset/",
            config.CLASS_NAME_MAP[class_name],
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


def split() -> None:
    """Split the images evenly across coders and classes"""

    for class_name in config.CLASS_NAMES:
        print(class_name)

        files = os.listdir(
            os.path.join(
                "/home/vanessa/hulk/ai2es/codebook_dataset/",
                config.CLASS_NAME_MAP[class_name],
            )
        )
        # each coder gets a random assortment across sites and classes
        random.shuffle(files)
        sub_lists = np.array_split(files, 4)
        names = ["carly", "chris", "mariana", "vanessa"]
        for name, subset in zip(names, sub_lists):
            for file in subset:

                savepath = os.path.join(
                    "/home/vanessa/hulk/ai2es/codebook_dataset/",
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


if __name__ == "__main__":
    parse_by_station()
    split()
