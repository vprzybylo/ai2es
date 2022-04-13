"""output statistics on the training dataset such as total images belong to 
a specific class at a particular station"""

import cocpit.config as config
import numpy as np
import pandas as pd
import os

pd.options.mode.chained_assignment = None  # default='warn'


def all_station_df():
    """make a df with all NYSM stations and 0s for classes len of stations"""
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
    return pd.DataFrame(
        {
            "stnid": stnid,
            "precip": np.zeros(len(stnid)),
            "no precip": np.zeros(len(stnid)),
            "obstructed": np.zeros(len(stnid)),
        }
    )


def dataset_stats(class_):
    """print # of images for a given class, unique stations for that class, and top station

    Args:
        class_ (str): class name
    Returns:
        df (pd.DataFrame): dataframe of stations present for a specific class
    """

    print(f"CLASS: {class_}")
    files = os.listdir(os.path.join(config.DATA_DIR, config.CLASS_NAME_MAP[class_]))
    stids = [file.split("_")[1].split(".")[0] for file in files]
    df = pd.DataFrame({"stids": stids})
    print(df.stids.describe())
    return df


def main():
    """create df highlighted by image count for each station/class"""
    df_precip = dataset_stats(class_="precipitation")
    df_no_precip = dataset_stats(class_="no precipitation")
    df_obstructed = dataset_stats(class_="obstructed")
    df_all_stns = all_station_df()

    for stn in df_all_stns["stnid"]:
        df_all_stns["precip"][df_all_stns["stnid"] == stn] = len(
            df_precip[df_precip["stids"] == stn]
        )
        df_all_stns["no precip"][df_all_stns["stnid"] == stn] = len(
            df_no_precip[df_no_precip["stids"] == stn]
        )
        df_all_stns["obstructed"][df_all_stns["stnid"] == stn] = len(
            df_obstructed[df_obstructed["stids"] == stn]
        )
    df_all_stns = df_all_stns.astype(
        {"precip": int, "no precip": int, "obstructed": int}
    )

    return df_all_stns.style.background_gradient()


if __name__ == "__main__":
    display = main()
