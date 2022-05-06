import os
import cocpit.config as config
from collections import defaultdict
from PIL import Image
import random
import numpy as np

stnid = [
    "ADDI",
    "ANDE",
    "BATA",
    "BEAC",
    "BELD",
    "BELL",
    "BELM",
    "BERK",
    "BING",
    "BKLN",
    "BRAN",
    "BREW",
    "BROC",
    "BRON",
    "BROO",
    "BSPA",
    "BUFF",
    "BURD",
    "BURT",
    "CAMD",
    "CAPE",
    "CHAZ",
    "CHES",
    "CINC",
    "CLAR",
    "CLIF",
    "CLYM",
    "COBL",
    "COHO",
    "COLD",
    "COPA",
    "COPE",
    "CROG",
    "CSQR",
    "DELE",
    "DEPO",
    "DOVE",
    "DUAN",
    "EAUR",
    "EDIN",
    "EDWA",
    "ELDR",
    "ELLE",
    "ELMI",
    "ESSX",
    "FAYE",
    "FRED",
    "GABR",
    "GFAL",
    "GFLD",
    "GROT",
    "GROV",
    "HAMM",
    "HARP",
    "HARR",
    "HART",
    "HERK",
    "HFAL",
    "ILAK",
    "JOHN",
    "JORD",
    "KIND",
    "LAUR",
    "LOUI",
    "MALO",
    "MANH",
    "MEDI",
    "MEDU",
    "MORR",
    "NBRA",
    "NEWC",
    "NHUD",
    "OLDF",
    "OLEA",
    "ONTA",
    "OPPE",
    "OSCE",
    "OSWE",
    "OTIS",
    "OWEG",
    "PENN",
    "PHIL",
    "PISE",
    "POTS",
    "QUEE",
    "RAND",
    "RAQU",
    "REDF",
    "REDH",
    "ROXB",
    "RUSH",
    "SARA",
    "SBRI",
    "SCHA",
    "SCHO",
    "SCHU",
    "SCIP",
    "SHER",
    "SOME",
    "SOUT",
    "SPRA",
    "SPRI",
    "STAT",
    "STEP",
    "SUFF",
    "TANN",
    "TICO",
    "TULL",
    "TUPP",
    "TYRO",
    "VOOR",
    "WALL",
    "WALT",
    "WANT",
    "WARS",
    "WARW",
    "WATE",
    "WBOU",
    "WELL",
    "WEST",
    "WFMB",
    "WGAT",
    "WHIT",
    "WOLC",
    "YORK",
]


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
                        config.DATA_DIR, config.CLASS_NAME_MAP[class_name], file
                    )
                )
                print(os.path.join(savepath, file))
                im.save(os.path.join(savepath, file))
            counts[stn] += 1


def split():
    for class_name in config.CLASS_NAMES:
        files = os.listdir(
            os.path.join("/ai2es/codebook_dataset/", config.CLASS_NAME_MAP[class_name])
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
                        config.DATA_DIR, config.CLASS_NAME_MAP[class_name], file
                    )
                )
                im.save(os.path.join(savepath, file))


split()
