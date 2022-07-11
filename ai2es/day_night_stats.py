import numpy as np
import pandas as pd


def precip_stats(year: int) -> None:
    df = pd.read_parquet(f"/ai2es/matched_parquet/{year}.parquet")
    print(year)
    print("precip images: ", len(df[df["precip_accum_1min [mm]"] > 0.001]))
    print("no precip images: ", len(df[df["precip_accum_1min [mm]"] == 0.00]))


def yearly_stats(year: int) -> None:
    df = pd.read_parquet(f"/ai2es/matched_parquet/{year}_timeofday.parquet")
    print(year)
    print("night images: ", len(df[df["night"] == True]))
    print("day images: ", len(df[df["night"] == False]))
    print(
        "night images precip > 0.001 mm: ",
        len(df[(df["night"] == True) & (df["precip_accum_1min [mm]"] > 0.001)]),
    )
    print(
        "night images precip == 0.0 mm: ",
        len(df[(df["night"] == True) & (df["precip_accum_1min [mm]"] == 0.0)]),
    )
    print(
        "day images precip > 0.001 mm: ",
        len(df[(df["night"] == False) & (df["precip_accum_1min [mm]"] > 0.001)]),
    )
    print(
        "day images precip = 0.0 mm: ",
        len(df[(df["night"] == False) & (df["precip_accum_1min [mm]"] == 0.0)]),
    )
    print(
        "all images precip > 0.001 mm: ",
        len(df[df["precip_accum_1min [mm]"] > 0.001]),
    )
    print(
        "all images precip = 0.0 mm: ",
        len(df[df["precip_accum_1min [mm]"] == 0.00]),
    )


for year in np.arange(2017, 2022):
    # yearly_stats(year)
    precip_stats(year)
