"""apply filters to NYSM/image parquet files"""
import pandas as pd
import cv2
from multiprocessing import Pool
import os
import time


class ImageFilter:
    def __init__(self, parquet_dir="../NYSM/"):
        self.parquet_dir = parquet_dir
        self.df = None

    def read_parquet(self, year: int):
        """read parquet file for a specified year"""
        self.df = pd.read_parquet(f"{self.parquet_dir}/{year}.parquet").reset_index()

    def image_path(self, photo_dir="../cam_photos"):
        """convert timestamp into image filename and append to df if there is a corresponding image"""

        date = self.df["time_5M"].dt.strftime("%Y%m%d")
        time = self.df["time_5M"].dt.strftime("%H%M%S")

        date_path = [
            photo_dir + "/" + date + "T" + time + "_" + self.df["station"] + ".jpg"
        ]
        start = time.process_time()
        photo_files = os.listdir(photo_dir)
        present_paths = [
            present_paths.append(path) for path in date_path if path in photo_files
        ]
        print(time.process_time() - start)
        print(len(present_paths))

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
    filt = ImageFilter()
    filt.read_parquet(2021)
    filt.image_path()

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
