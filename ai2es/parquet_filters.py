"""apply filters to NYSM/image parquet files"""
import pandas as pd
import opencv as cv2
from skimage import color
from skimage import io


class Filter:
    def __init__(self, parquet_dir="../NYSM/"):
        self.parquet_dir = parquet_dir
        self.df = None

    def read_parquet(self, year):
        """read parquet file for a specified year"""
        self.df = pd.read_parquet(f"{self.parquet_dir}/{year}.parquet")

    def day(self):
        """find images taken during the day"""
        image = cv2.imread(self.df["image_path"])
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

    def grayscale(self):
        img = io.imread(self.df["img_path"])
        return color.rgb2gray(img)


def main():
    options = Filter()
    options.read_parquet(2021)
    options.day()  # or night()
    options.precip()  # of no_precip
    options.station_filter("TANN")
    options.grayscale()


if __name__ == "__main__":
    main()
