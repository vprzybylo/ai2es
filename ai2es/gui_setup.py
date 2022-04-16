"""initialize gui by reading in images specified by user inputs in notebooks/label.ipynb"""

import cocpit
import os
import pandas as pd
from typing import Tuple
import itertools


def read_parquet(
    year: int, time_of_day: str, precip_threshold: float, precip: str
) -> Tuple[pd.DataFrame, str]:
    """
    Read a time matched parquet file for a given year between camera images and observations.
    Filter based on year, precip or no precip, precip threshold, and time of day (day or night).

    Args:
        year (int): user-specified year to label images from
        time_of_day (str): 'day' or 'night'
        precip_threshold (float): only grab images above this threshold
        precip (str): 'precip' or 'no precip'
    """
    df = pd.read_parquet(f"/ai2es/matched_parquet/{year}_timeofday.parquet")
    df = df[df["night"] == True if time_of_day == "night" else df["night"] == False]
    df = df[
        df["precip_accum_1min [mm]"] > precip_threshold
        if precip == "precip"
        else df["precip_accum_1min [mm]"] == 0.0
    ]
    return (df, f"/ai2es/{time_of_day}_{precip}_hand_labeled/{year}")


def shuffle_df(df: pd.DataFrame) -> pd.DataFrame:
    """shuffle df paths such that there is station diversity in training dataset"""
    return df.sample(frac=1)


def make_folders(folder_dest) -> None:
    """
    Make folders in training dir to save to if they don't exist

    Args:
        folder_dest (str): folder to save images to
    """

    for label in cocpit.config.CLASS_NAME_MAP.values():
        save_path = os.path.join(folder_dest, label)
        if not os.path.exists(save_path):
            os.makedirs(save_path)


def show_new_images(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure images shown haven't already been labeled

    Args:
        df (pd.DataFrame): input df with all paths from read parquet
    Returns:
        df (pd.DataFrame): df where paths are removed if already labeled
    """

    all_classes = [
        os.listdir(
            os.path.join(cocpit.config.DATA_DIR, cocpit.config.CLASS_NAME_MAP[class_])
        )
        for class_ in cocpit.config.CLASS_NAMES
    ]

    all_classes = list(itertools.chain.from_iterable(all_classes))
    already_labeled = pd.DataFrame({"path": all_classes})
    print("Already labeled: ", len(already_labeled))
    df_paths = pd.DataFrame({"paths": df["path"].str.split("/").str[-1]})
    print("Observations in parquet file: ", len(df_paths))
    len_before = len(df)
    df = df[~df_paths["paths"].isin(already_labeled["path"])]
    print(
        f"Removing {len_before-len(df)} images that have already been labeled. {len(df)} remain."
    )
    return df
