"""Apply filters to image in df for a given year"""
import cv2
import pandas as pd
from dataclasses import dataclass


@dataclass
class ImageFilter:
    """partition yearly df of images for day, night, precip, no precip, etc."""

    year: int

    def read_df(self):
        """read parquet file that has yearly df of images"""
        self.df = pd.read_parquet(f"/ai2es/matched_parquet/{self.year}.parquet")

    def day(self, img_path: str):
        """find images taken during the day"""
        image = cv2.imread(img_path)
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        return None if (b == g).all() and (b == r).all() else True

    def night(self):
        """find images taken at night"""

    def precip(self) -> None:
        # CHANGE VAR FOR 1M
        """find images where there IS precipitation occuring according to 5 min difference in mesonet data"""
        self.df = self.df[self.df["precip_5min"] > 0.0]

    def no_precip(self) -> None:
        """find images where there is NO precipitation occuring"""
        self.df = self.df[self.df["precip_5min"] == 0.0]

    def station_filter(self, station: str) -> None:
        """find images from a specific station id

        Args:
            station (str): station id
        """
        self.df = self.df[self.df["stid"] == station]

    def grayscale(self, img_path: str):  # -> ndarray[int, int]:
        """convert image to grayscale

        Args:
            img_path (str): full path to image

        Returns:
            ndarray: converted image to gray scale
        """
        image = cv2.imread(img_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def main() -> None:
    filt = ImageFilter(year)

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
