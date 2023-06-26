"""output statistics on the training dataset such as total images belong to
a specific class at a particular station"""

import os

import cocpit.config as config
import numpy as np
import pandas as pd
from cocpit import config as config

pd.options.mode.chained_assignment = None  # default='warn'


def all_station_df() -> pd.DataFrame:
    """make a df with all NYSM stations and 0s for classes len of stations"""

    return pd.DataFrame(
        {
            "stnid": config.STNID,
            "precip": np.zeros(len(config.STNID)),
            "no precip": np.zeros(len(config.STNID)),
            "obstructed": np.zeros(len(config.STNID)),
        }
    )


def dataset_stats(class_: str) -> pd.DataFrame:
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


def main() -> pd.DataFrame:
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

    return df_all_stns.style.background_gradient(
        subset=["precip", "no precip", "obstructed"], axis=None
    )


if __name__ == "__main__":
    display = main()
