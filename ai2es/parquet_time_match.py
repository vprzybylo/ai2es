"""apply filters to NYSM parquet files by year"""
import pandas as pd
import os
import numpy as np
from datetime import datetime
import warnings
from dask.distributed import Client, LocalCluster
import dask


warnings.filterwarnings("ignore")


def start_client(num_workers):
    """
    initialize dask client
    """
    cluster = LocalCluster(
        n_workers=4, threads_per_worker=1, memory_limit=None, silence_logs=False
    )
    print("dashboard link: ", cluster.dashboard_link)

    # cluster.scale(num_workers)
    client = Client(cluster)
    print(client)
    return client


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

    def camera_photo_paths(self, photo_dir: str = "../cam_photos") -> None:
        """create dataframe of camera images including station ids, paths, and datetimes

        Args:
            photo_dir (str, optional): directory of camera photos. Defaults to "../cam_photos".
        """
        photo_files = os.listdir(photo_dir)
        stations = [photo.split("_")[1].split(".")[0] for photo in photo_files]
        dates = [
            datetime.strptime(photo.split("_")[0].replace("T", ""), "%Y%m%d%H%M%S")
            for photo in photo_files
        ]
        self.camera_df = pd.DataFrame({"stid": stations, "datetime": dates})
        # truncate for year
        self.camera_df = self.camera_df[
            self.camera_df["datetime"].dt.year == self.year
        ].set_index(["datetime"])

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

    def match_dates_dask(self, group_nysm, group_cam):
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

    def concat_station_data(self, matched_groups) -> None:
        len_before = len(self.df)
        print(type(matched_groups))
        # remove rows in nysm df that don't have corresponding image
        self.df = pd.concat(matched_groups).dropna(subset=["camera path"])
        print("[INFO] Length of observations and matched images: ", len(self.df))
        print(
            f"[INFO] Removed {len_before - len(self.df)} rows of data that don't have a corresponding image"
        )

    def find_closest_date(self, time_diff=5) -> None:
        """
        for each datetime in the camera photo df, find the closest date in the mesonet df
        - vectorized
        """
        # eventually will hold the matched camera paths based on time and station
        self.df["camera path"] = np.nan

        # First group camera df by station id since
        # there can be multiple identical times of images at diff stations.
        # Check that the stations in the obs df are also in the camera df
        # so that when we groupby stations the stations align
        self.df = self.df.loc[self.df["stid"].isin(self.camera_df["stid"])]
        # parallelize finding close dates between images and obs across station id groups
        all_stn_groups = [
            dask.delayed(self.match_dates_dask)(group_nysm, group_cam)
            for (cam_stn, group_cam), (ny_stn, group_nysm) in zip(
                self.camera_df.groupby("stid"), self.df.groupby("stid")
            )
        ]
        print("[INFO] Starting dask client")
        client = start_client(4)
        print(
            "[INFO] Matching dates between images and observations for all stations..",
        )
        matched_groups = client.compute(all_stn_groups)
        matched_groups.dask.visualize()
        print("matched_groups", matched_groups)
        # matched_groups = client.gather(matched_groups)

        self.concat_station_data(matched_groups)


def main() -> None:

    for year in range(2017, 2021):
        print(f"[INFO] processing {year}..")
        match = DateMatch(year)
        match.read_parquet()
        match.camera_photo_paths()
        match.find_closest_date()


if __name__ == "__main__":
    main()
