"""apply filters to NYSM parquet files by year"""
import pandas as pd
import cv2
from multiprocessing import Pool
import os
import time


class ImageFilter:
    def __init__(self, year: int, parquet_dir: str = "../mesonet_parquet"):
        self.year = year
        self.parquet_dir = parquet_dir
        self.df = None

    def read_parquet(self):
        """read parquet file for a specified year"""
        self.df = pd.read_parquet(
            f"{self.parquet_dir}/{self.year}.parquet"
        ).reset_index()

    def grouped_df_year(self):
        """return dataframe of NYSM files by year"""
        self.df["year"] = self.df["filename"][:4]
        return self.df[self.df["year"] == self.year]

    def camera_photo_paths(self, photo_dir: str = "../cam_photos"):
        """camera photo paths down to every 5 minutes (ignore seconds to line up with NYSM data)"""
        photo_files = os.listdir(photo_dir)
        photo_files = [photo[:13] + photo[15:] for photo in photo_files]
        print("len camera photo paths", len(photo_files))
        return photo_files

    def mesonet_paths(self):
        """
        Create mesonet data path filenames from time stamps in same format as camera images.
        To be used to see if a camera image exists for the given 5 min interval
        """
        date = self.df["time_5M"].dt.strftime("%Y%m%d")
        time_fmt = self.df["time_5M"].dt.strftime("%H%M")  # ignore seconds
        self.df["mesonet_path"] = (
            date + "T" + time_fmt + "_" + self.df["station"] + ".jpg"
        )

    def check_camera_path_exists(self):
        """keep NYSM row if same camera filename exists"""
        print("len df before check", len(self.df))
        self.df = self.df[self.df["mesonet_path"].isin(self.camera_photo_paths())]
        print("len after check", len(self.df))

    def day(self, img_path: str):
        """find images taken during the day"""
        image = cv2.imread(img_path)
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        return None if (b == g).all() and (b == r).all() else True

    def night(self):
        """find images taken at night"""

    def precip(self):
        """find images where there IS precipitation occuring according to 5 min difference in mesonet data"""
        self.df = self.df[self.df["precip_5min"] > 0.0]

    def no_precip(self):
        """find images where there is NO precipitation occuring"""
        self.df = self.df[self.df["precip_5min"] == 0.0]

    def station_filter(self, station):
        """find images from a specific station id"""
        self.df = self.df[self.df["station"] == station]

    def grayscale(self, img_path):
        image = cv2.imread(img_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def main():
    for year in range(2015, 2021):
        print(f"[INFO] processing {year}..")
        filt = ImageFilter(year)
        filt.read_parquet()
        filt.mesonet_paths()
        filt.check_camera_path_exists()

    # pool = Pool(processes=8)
    # time_of_day = pool.map(filt.day, filt.df['img_paths']) # or night()
    # time_of_day = pool.map(filt.precip, filt.df['img_paths']) # or night()

    # filt.precip()  # of no_precip
    # filt.station_filter("TANN")
    # filt.grayscale()

    # print("[INFO] waiting for processes to finish...")
    # pool.close()
    # pool.join()


# print("[INFO] multiprocessing complete")


if __name__ == "__main__":
    main()
