"""apply filters to NYSM parquet files by year"""
import pandas as pd
import os
import numpy as np
from datetime import datetime
import warnings
import pandas as pd
import time
from typing import List
from dataclasses import dataclass
from multiprocessing import Pool

warnings.filterwarnings("ignore")


@dataclass
class Images:
    """create df for camera images"""

    year: int
    camera_df: pd.DataFrame = None

    def create_station_col(self) -> None:
        """use filename to parse out station id"""
        self.camera_df["stid"] = (
            self.camera_df["path"]
            .str.split("/")
            .str[-1]
            .str.split("_")
            .str[1]
            .str.split(".")
            .str[0]
        )

    def create_datetime_col(self) -> None:
        """use filename to parse out datetime"""
        self.camera_df["datetime"] = pd.to_datetime(
            self.camera_df["path"].str.split("/").str[-1].str.split("_").str[0],
            format="%Y%m%dT%H%M%S",
        )

    def camera_photo_paths(
        self, photo_dir: str = "../cam_photos/"
    ) -> None:  # sourcery skip: for-append-to-extend
        """create dataframe of camera images including station ids, paths, and datetimes

        Args:
            photo_dir (str, optional): directory of camera photos (before each year).
        """

        photo_files = []
        # only get files for a specific year
        for path, subdirs, files in os.walk(os.path.join(photo_dir, str(self.year))):
            for name in files:
                [
                    photo_files.append(os.path.join(path, name))
                    for name in files
                    if name.lower().endswith(".jpg")
                ]

        self.camera_df = pd.DataFrame({"path": photo_files}).astype(str)
        self.create_station_col()
        self.create_datetime_col()
        self.camera_df = self.camera_df.set_index(["datetime"])
        print(f"[INFO] There are {len(self.camera_df)} images taken in {self.year}")


@dataclass
class DateMatch(Images):
    """match camera images to closest mesonet observation in time for each station"""

    parquet_dir: str = "../mesonet_parquet_1M"
    df: pd.DataFrame = None  # mesonet obs df

    def read_parquet(self) -> None:
        """read parquet file holding mesonet data for a specified year (all stations)"""
        self.df = (
            pd.read_parquet(f"{self.parquet_dir}/{self.year}.parquet")
            .set_index(["datetime"])
            .sort_values(by=["datetime"])
        )

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

    def group_by_stations(self, time_diff=5) -> List[pd.DataFrame]:
        """
        First group camera df by station id since there can be multiple
        identical times of images at diff stations.
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
        return all_stn_groups

    def write_df_per_year(self, all_stn_groups: pd.DataFrame) -> None:
        """output time matched df's per year for all stations"""
        all_stn_groups.to_parquet(f"../matched_parquet/{self.year}.parquet")


def process_years(year: int) -> None:
    """match dates between obs and images for a given year"""
    print(f"[INFO] processing {year}..")
    start_time = time.time()
    match = DateMatch(year=year)
    match.camera_photo_paths()
    match.read_parquet()
    all_stn_groups = match.group_by_stations()
    all_stn_groups = match.concat_stn_data(all_stn_groups, time.time() - start_time)
    match.write_df_per_year(all_stn_groups)
    print(f"[INFO] Year {year} completed in {round(time.time()-start_time, 2)} seconds")


def main() -> None:
    years = np.arange(2017, 2022)
    pool = Pool(len(years))
    pool.map(process_years, years)
    pool.close()


if __name__ == "__main__":
    main()
