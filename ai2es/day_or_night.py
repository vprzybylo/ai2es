"""
Determine if camera in nightmode.
Processes in parallel across years.
"""
import cv2
import pandas as pd
import numpy as np
from dataclasses import dataclass
import time
from PIL import Image
from parallelbar import progress_map


@dataclass
class ImageFilter:
    """
    Determine if camera in night/IR mode and append day or night to df

    Args:
        year (int): year of data to process
        df (pd.DataFrame): df with time matched images and observations
    """

    year: int
    df: pd.DataFrame = None

    def read_parquet(self):
        """
        Read parquet file that has yearly df of images
        """
        start_time = time.time()
        self.df = pd.read_parquet(f"/ai2es/matched_parquet/{str(self.year)}.parquet")
        print(
            f"[INFO] Read {self.year} parquet file in {time.time()-start_time} seconds."
        )

    def time_of_day(self, file: str) -> bool:
        """
        Find if images were taken during the day or night

        Args:
            file (str): image filename to open
        """

        image = np.asarray(Image.open(file))
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        return bool((b == g).all() and (b == r).all())


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

    year = 2022
    filt = ImageFilter(year)
    filt.read_parquet()
    print(f'{len(filt.df["path"])} files to process')
    is_night = progress_map(
        filt.time_of_day,
        filt.df["path"],
        chunk_size=158433,
        n_cpu=10,
        core_progress=True,
    )

    filt.df["night"] = is_night
    filt.df.to_parquet(f"/ai2es/matched_parquet/{year}_timeofday.parquet")


if __name__ == "__main__":
    main()
