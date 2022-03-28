"""apply filters to NYSM parquet files by year"""
import pandas as pd
import cv2
from multiprocessing import Pool
import os
import numpy as np
from datetime import datetime
from numpy import ndarray
from typing import Optional
import warnings

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
        self.camera_df = self.camera_df[self.camera_df["datetime"].dt.year == self.year]

    def check_time_diff(
        self, nysm_time: datetime, image_time: datetime, time_diff: int = 5
    ):
        """ensure that the time matching between the camera time
        (image_time) and the nysm obs time (nysm_time) is less than time_diff minutes

        Args:
            nysm_time (pandas._libs.tslibs.timestamps): time of nysm observation
            image_time (pandas._libs.tslibs.timestamps): time image was taken
            time_diff (int, optional): time difference allowed in minutes between when
                                       the image was taken and when the mesonet observation was recorded.
                                       Defaults to 5.
        """
        return np.abs(image_time.minute - nysm_time.minute) < time_diff

    def find_closest_date(self) -> None:
        """
        for each datetime in the camera photo df, find the closest date (down to sec) in the mesonet df
        """
        # eventually will hold the matched camera paths based on time and station
        self.df["camera path"] = np.nan

        # first group camera df by station id
        # there can be multiple identical times of images at diff stations
        camera_stn = self.camera_df.groupby("stid")
        nysm_stn = self.df.groupby("stid")
        all_stn_groups = []
        for (stn, group_cam), (_, group_nysm) in zip(camera_stn, nysm_stn):
            # iterate through camera datetimes and find the nearest mesonet datetime
            for idx, row in group_cam.iterrows():

                # get the index of the nysm datetime at the closest match to the image datetime
                matched_date_idx = group_nysm.index.get_loc(
                    row["datetime"].to_pydatetime(), method="nearest"
                )

                # check that there is less than a 5 min time diff between when the image
                # was taken and the matched nysm observation timestamp
                time_diff = self.check_time_diff(
                    group_nysm.index[matched_date_idx], row["datetime"]
                )

                # insert camera date in mesonet df at closest index
                group_cp = (
                    group_nysm.copy()
                )  # to get rid of 'A value is trying to be set on a copy of a slice from a DataFrame'
                group_cp["camera path"].iloc[matched_date_idx] = (
                    row["datetime"] if time_diff else np.nan
                )

                all_stn_groups.append(group_cp)
        len_before = len(self.df)
        # remove rows in nysm df that don't have corresponding image
        self.df = pd.concat(all_stn_groups).dropna(subset=["camera path"])
        print("[INFO] Length of observations and matched images: ", len(self.df))
        print(
            f"[INFO] Removed {len_before - len(self.df)} rows of data that don't have a corresponding image"
        )


# class ImageFilter(DateMatch):
#     def day(self, img_path: str) -> Optional[bool]:
#         """find images taken during the day"""
#         image = cv2.imread(img_path)
#         b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
#         return None if (b == g).all() and (b == r).all() else True

#     def night(self):
#         """find images taken at night"""

#     def precip(self) -> None:
#         """find images where there IS precipitation occuring according to 5 min difference in mesonet data"""
#         self.df = self.df[self.df["precip_5min"] > 0.0]

#     def no_precip(self) -> None:
#         """find images where there is NO precipitation occuring"""
#         self.df = self.df[self.df["precip_5min"] == 0.0]

#     def station_filter(self, station: str) -> None:
#         """find images from a specific station id

#         Args:
#             station (str): station id
#         """
#         self.df = self.df[self.df["station"] == station]

#     def grayscale(self, img_path: str) -> ndarray[int, int]:
#         """convert image to grayscale

#         Args:
#             img_path (str): full path to image

#         Returns:
#             ndarray: converted image to gray scale
#         """
#         image = cv2.imread(img_path)
#         return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def main() -> None:

    for year in range(2017, 2021):
        print(f"[INFO] processing {year}..")
        match = DateMatch(year)
        match.read_parquet()
        match.camera_photo_paths()
        match.find_closest_date()

        # filt = ImageFilter(match)

        # filt.check_camera_path_exists()

    # pool = Pool(processes=8)
    # time_of_day = pool.map(filt.day, filt.df['img_paths']) # or night()
    # time_of_day = pool.map(filt.precip, filt.df['img_paths']) # or night()

    # filt.precip()  # of no_precip
    # filt.station_filter("TANN")
    # filt.grayscale()

    # print("[INFO] waiting for processes to finish...")
    # pool.close()
    # pool.join()


if __name__ == "__main__":
    main()
