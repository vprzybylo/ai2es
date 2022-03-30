"""apply filters to NYSM parquet files by year"""
import pandas as pd
import os
import numpy as np
from datetime import datetime
import warnings
import pandas as pd
import time
from typing import List
from glob import glob

warnings.filterwarnings("ignore")


class DateMatch:
    """match camera photos to closest mesonet observation in time for each station"""

    def __init__(self, year: int, parquet_dir: str = "../mesonet_parquet_1M") -> None:
        self.year = year
        self.parquet_dir = parquet_dir
        self.camera_df: pd.DataFrame = None
        self.df: pd.DataFrame = None

    def read_parquet(self) -> None:
        """read parquet file holding mesonet data for a specified year (all stations)"""
        self.df = (
            pd.read_parquet(f"{self.parquet_dir}/{self.year}.parquet")
            .set_index(["datetime"])
            .sort_values(by=["datetime"])
        )

    def camera_photo_paths(
        self, photo_dir: str = "../NYSM/archive/nysm/cam_photos/"
    ) -> None:
        """create dataframe of camera images including station ids, paths, and datetimes

        Args:
            photo_dir (str, optional): directory of camera photos. Defaults to "../cam_photos".
        """

        photo_files = []
        # only get files for a specific year
        for path, subdirs, files in os.walk(os.path.join(photo_dir, str(self.year))):
            for name in files:
                if name.lower().endswith(".jpg"):
                    photo_files.append(os.path.join(path, name))

        self.camera_df = pd.DataFrame({"path": photo_files}).astype(str)

        self.camera_df["stnid"] = (
            self.camera_df["path"]
            .str.split("/")[-1]
            .str.split("_")[1]
            .str.split(".")[0]
        )

        self.camera_df["datetime"] = datetime.strptime(
            self.camera_df["path"]
            .str.split("/")[-1]
            .str.split("_")[0]
            .replace("T", ""),
            "%Y%m%d%H%M%S",
        )
        self.camera_df = self.camera_df.set_index(["datetime"])
        print(self.camera_df)

    def check_time_diff(
        self, nysm_time: datetime, image_time: datetime, time_diff: int = 5
    ) -> bool:
        """ensure that the time matching between the camera time
        (image_time) and the nysm obs time (nysm_time) is less than time_diff minutes

        Args:
            nysm_time (pandas._libs.tslibs.timestamps): time of nysm observation
            image_time (pandas._libs.tslibs.timestamps): time image was taken
            time_diff (int, optional): time difference allowed in minutes between when
                                       the image was taken and when the mesonet observation was recorded.
                                       Defaults to 5.
        """
        time_delta_mins = divmod((image_time - nysm_time).total_seconds(), 60)[0]
        return time_delta_mins < time_diff

    def concat_stn_data(self, all_stn_groups: List[pd.DataFrame], timer: float) -> None:
        print(type(all_stn_groups))
        """
        Concatenate all stations that have time matched data
        Remove rows of obs that dont have a match

        Args:
            all_stn_groups (List[pd.DataFrame]): list of image matched dfs from each station
            timer (float): how long it took to group and match obs with images
        """
        len_before = len(self.df)
        self.df = pd.concat(all_stn_groups).dropna(subset=["camera path"])
        print(
            f"[INFO] {len(self.df)} observations were matched with images in {round(timer, 2)} seconds"
        )
        print(
            f"[INFO] {len_before - len(self.df)} rows of data were removed that don't have a corresponding image."
        )

    def find_closest_date(
        self, group_nysm: pd.DataFrame, group_cam: pd.DataFrame
    ) -> pd.DataFrame:
        """for each datetime in the camera photo df, find the closest date in the mesonet df
        - vectorized

        Args:
            group_nysm (pd.DataFrame): station grouped dataframe of nysm observations
            group_cam (pd.DataFrame): station grouped dataframe of camera images

        Returns:
            group_cp (pd.DataFrame): time matched df for a station

        """

        # get the indices of the nysm datetimes at the closest match to the image datetimes
        matched_date_indices = group_nysm.index.get_indexer(
            group_cam.index, method="nearest"
        )

        # check that there is less than a 5 min time diff between when the image
        # was taken and the matched nysm observation timestamp
        time_diff = self.check_time_diff(
            group_nysm.index[matched_date_indices],
            group_cam.index,
        )

        group_cp = (
            group_nysm.copy()
        )  # to get rid of warning: "A value is trying to be set on a copy of a slice from a DataFrame"
        # remove any indices where time match is too far apart
        matched_date_indices = matched_date_indices[time_diff]

        # assign datetime that image was taken to nysm df where time_diff returns true (i.e., close enough match)
        group_cp["camera path"].iloc[matched_date_indices] = group_cam.index[time_diff]
        return group_cp

    def group_by_stations(self, time_diff=5) -> None:
        """
        First group camera df by station id since
        there can be multiple identical times of images at diff stations.
        Check that the stations in the obs df are also in the camera df
        so that when we groupby stations the stations align

        Args:
            time_diff (int): time difference allowed between image and obs time. Defaults to 5 mins
        """
        # eventually will hold the matched camera paths based on time and station
        self.df["camera path"] = np.nan
        self.df = self.df.loc[self.df["stid"].isin(self.camera_df["stid"])]

        start_time = time.time()
        all_stn_groups = []
        for (cam_stn, group_cam), (ny_stn, group_nysm) in zip(
            self.camera_df.groupby("stid"), self.df.groupby("stid")
        ):
            group_cp = self.find_closest_date(group_nysm, group_cam)
            all_stn_groups.append(group_cp)
        self.concat_stn_data(all_stn_groups, time.time() - start_time)

        # station_cams = []
        # station_obs = []
        # for (cam_stn, group_cam), (ny_stn, group_nysm) in zip(
        #     self.camera_df.groupby("stid"), self.df.groupby("stid")
        # ):
        #     station_cams.append(group_cam)
        #     station_obs.append(group_nysm)

        # len_before = len(self.df)
        # self.df = ray.get(
        #     [
        #         self.find_closest_date(station_obs, station_cams)
        #         for station_obs, station_cams in zip(station_obs, station_cams)
        #     ]
        # ).dropna(subset=["camera path"])
        # print("[INFO] Length of observations and matched images: ", len(self.df))
        # print(
        #     f"[INFO] Removed {len_before - len(self.df)} rows of data that don't have a corresponding image"
        # )


def main() -> None:

    for year in range(2017, 2021):
        print(f"[INFO] processing {year}..")
        start_time = time.time()
        match = DateMatch(year)
        match.read_parquet()
        match.camera_photo_paths()
        match.group_by_stations()
        print(
            f"[INFO] Year {year} completed in {round(time.time()-start_time, 2)} seconds"
        )


if __name__ == "__main__":
    main()
