"""
Make directories for precip and no precip based on rain gauge and mesonet data
that corresponds to 2017 and 2022 images in hand labeled training dataset
for comparison

Run for each root dir subfolder/class at a time - not parallelized - takes awhile
"""
import pandas as pd
import os
import shutil


if __name__ == "__main__":

    df_2017 = pd.read_parquet("/ai2es/matched_parquet/2017_timeofday.parquet")
    df_2017["image_path"] = df_2017["path"].str.split("/").str[-1]

    df_2022 = pd.read_parquet(
        "/ai2es/matched_parquet/2022_timeofday.parquet"
    ).drop(columns=["tair", "ta9m"])
    df_2022["image_path"] = df_2022["path"].str.split("/").str[-1]
    print("done reading dfs")

    root = "/ai2es/codebook_dataset/combined_extra/obstructed"
    for filename in os.listdir(root):
        if "2017" in filename:
            precip = df_2017["precip_accum_1min [mm]"][
                df_2017["image_path"] == filename
            ]
        else:
            precip = df_2022["precip_5min"][df_2022["image_path"] == filename]
        if precip.item() > 0.0:
            shutil.copy(
                os.path.join(root, filename),
                "/ai2es/unsupervised/precip",
            )
        else:
            shutil.copy(
                os.path.join(root, filename),
                "/ai2es/unsupervised/no_precip",
            )
