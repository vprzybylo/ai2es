"""match NYSM camera images to precip observations for all stations by year"""
import pandas as pd
import os
import numpy as np
from datetime import datetime
import warnings
import pandas as pd
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from datetime import timedelta
import cocpit.config as config

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

    def camera_photo_paths(self) -> None:
        """create dataframe of camera images including station ids, paths, and datetimes"""

        photo_files = Path(os.path.join(config.photo_dir, str(self.year))).rglob(
            "*.jpg"
        )

        self.camera_df = pd.DataFrame({"path": photo_files}).astype(str)
        self.create_station_col()
        self.create_datetime_col()
        self.camera_df = self.camera_df.set_index(["datetime"])
        print(f"[INFO] {self.year}: There were {len(self.camera_df)} images taken.")


@dataclass
class DateMatch(Images):
    """match camera images to closest mesonet observation in time for each station"""

    df: pd.DataFrame = None  # mesonet obs df
    all_station_groups = (
        None  # list of pd.Dataframe of time matched stations to be concatenated
    )

    def read_parquet(self) -> None:
        """read parquet file holding mesonet data for a specified year (all stations)"""
        self.df = (
            pd.read_parquet(f"{config.parquet_dir}/{self.year}.parquet")
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
        # time_delta_mins = divmod((image_time - nysm_time).total_seconds(), 60)[0]
        actual_time_delta = np.abs(image_time - nysm_time)
        return actual_time_delta < timedelta(minutes=time_diff)

    def concat_stn_data(self) -> None:
        """
        Concatenate all stations that have time matched data
        Remove rows of obs that dont have a match

        Args:
            timer (float): how long it took to group and match obs with images
        """
        len_before = len(self.df)
        self.df = pd.concat(self.all_stn_groups).dropna()
        print(
            f"[INFO] {self.year}: {len(self.df)} observations were matched with images that have precip data."
        )
        print(
            f"[INFO] {self.year}: {len_before - len(self.df)} rows of data were removed that don't have a corresponding image or precip data."
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
        group_nysm = group_nysm[~group_nysm.index.duplicated()].sort_index()
        group_nysm = group_nysm[group_nysm.index.notnull()]
        group_cam = group_cam[~group_cam.index.duplicated()].sort_index()

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
        group_cp["camera time"].iloc[matched_date_indices] = group_cam.index[time_diff]
        group_cp["path"].iloc[matched_date_indices] = group_cam["path"][time_diff]
        return group_cp

    def group_by_stations(self, time_diff=5) -> None:
        """
        First group camera df by station id since there can be multiple
        identical times of images at diff stations.
        Check that the stations in the obs df are also in the camera df
        so that when we groupby stations the stations align

        Args:
            time_diff (int): time difference allowed between image and obs time. Defaults to 5 mins
        """
        # eventually will hold the matched camera times based on time and station
        self.df["camera time"] = np.nan
        self.df["path"] = np.nan
        self.df = self.df.loc[self.df["stid"].isin(self.camera_df["stid"])]

        start_time = time.time()
        self.all_stn_groups = []
        for (cam_stn, group_cam), (ny_stn, group_nysm) in zip(
            self.camera_df.groupby("stid"), self.df.groupby("stid")
        ):
            group_cp = self.find_closest_date(group_nysm, group_cam)
            self.all_stn_groups.append(group_cp)

    def time_diff_average(self) -> None:
        """print stats on how far apart obs are from image timestamps (<time_diff mins)"""
        actual_time_diff = np.abs(
            pd.to_datetime(self.df["camera time"]) - self.df.index
        )
        print(
            f"[INFO] {self.year}: Distribution stats for how far apart obs and image time stamps are: {actual_time_diff.describe()}"
        )

    def write_df_per_year(self) -> None:
        """output time matched df's per year for all stations"""
        self.df.to_parquet(f"{config.write_path}/{self.year}.parquet")


def process_years(year: int) -> None:
    """match dates between obs and images for a given year"""
    print(f"[INFO] processing {year}..")
    start_time = time.time()
    match = DateMatch(year=year)
    match.camera_photo_paths()
    match.read_parquet()
    match.group_by_stations()
    match.concat_stn_data()
    match.time_diff_average()
    match.write_df_per_year()
    print(f"[INFO] Year {year} completed in {round(time.time()-start_time, 2)} seconds")


def main() -> None:
    years = np.arange(2017, 2022)
    pool = Pool(len(years))
    pool.map(process_years, years)
    pool.close()
    end_time = time.time()


if __name__ == "__main__":
    main()
