"""
create yearly parquet files of mesonet data for every station for either 1 min or 5 min observations
"""

import os
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from cocpit import config as config


@dataclass
class NYSM:
    """
    Base class to drop unused vars and convert to parquet files by year

    Args:
        df: pd.DataFrame = None
    """

    def drop_unused_vars(self) -> None:
        """Only keep certain variables"""
        keep_vars = [
            "tair",
            "ta9m",
            "precip",
            "precip_total",
            "precip_max_intensity",
        ]
        self.df: pd.DataFrame = self.df[keep_vars]

    def write_parquet(self, temporal_resolution: str, filename: str) -> None:
        """
        Write combined dataframe from multiple years/stations to parquet

        Args:
            temporal_resolution (str): time frequency of 1M or 5M
            filename (str): filename to write to parquet
        """

        if temporal_resolution == "1M":
            self.df["datetime"] = pd.to_datetime(self.df["datetime"], errors="coerce")
            path = "../mesonet_parquet_1M"
        elif temporal_resolution == "5M":
            path = "../mesonet_parquet_5M"
        else:
            print("temporal resolution must be either 1M or 5M")
            return

        pq.write_table(pa.Table.from_pandas(self.df), f"{path}/{filename}.parquet")


@dataclass
class NYSM_1M(NYSM):
    """
    Convert csv's to parquet for 1 min observations from 2017-2021
    - Manually removed the few rows with 7 columns instead of 5 and values that were not floats (e.g., 0.00.00)
    - Mounted from /raid/lgaudet/precip/Precip/NYSM_1min_data to /home/vanessa/hulk/ai2es/1_min_obs
    - No snow measurements or T, RH, etc. in these csvs

    Args:
        df (pd.DataFrame): Concatenated df for 2017-2021 and NYSM observations at 1 min observations.
                           Holds station, datetime, precip accumulation, and total precip
        df_years (pd.DataFrame): grouped by by yearl of NYSM observations at 1 min observations.
        filelist (List[str]): List of csv files of 1 min observations from csv_file_dir
    """

    df: pd.DataFrame = None
    df_years: pd.DataFrame = None
    filelist: List[str] = field(default_factory=list, init=False)

    def csv_file_list(self) -> None:
        """
        Generate list of csv files of 1 min observations from csv_file_dir
        """
        self.filelist = []
        for root, dirs, files in os.walk(config.CSV_FILE_DIR):
            self.filelist.extend(
                os.path.join(root, file)
                for file in files
                if "checkpoint" not in file and file.endswith("csv")
            )

    def grouped_df_year(self) -> pd.DataFrame:
        """
        Group dataframe of NYSM files by year
        """
        self.df = pd.DataFrame(self.filelist, columns=["filename"])
        self.df["year"] = self.df["filename"].str.split("/").str[4].str[:4]
        self.df_years = self.df.groupby("year")

    def read_data(self, group: pd.DataFrame) -> None:
        """
        - Read csv's for each day for a given year and concatenate into df

        - For reference on local machine:
            - 2017: 25.83 sec
            - 2018: 28.43 sec
            - 2019: 27.28 sec
            - 2020: 28.32 sec
            - 2021: 27.41 sec

        Args:
            group (pd.DataFrame): yearly df
        """
        # sourcery skip: for-append-to-extend, list-comprehension
        yearly_files = []

        for filename in group["filename"]:
            yearly_files.append(
                pd.read_csv(
                    filename,
                    header=0,
                    on_bad_lines="skip",  # skip some rows that have 7 columns instead of 5..
                    dtype={
                        "stid": "str",
                        "datetime": "str",
                        "intensity [mm/min]": "float64",
                        "precip_accum_1min [mm]": "float64",
                        "precip_total [mm]": "float64",
                    },
                )
            )

        self.df = pd.concat(yearly_files, axis=0, ignore_index=True)


@dataclass
class NYSM_5M(NYSM):
    """
    Convert netcdf's to parquet for 5 min observations from 2015-2021
    - Mounted in container from /raid/NYSM/archive/nysm/netcdf/proc/:/home/vanessa/hulk/ai2es/5_min_obs/

    Args:
        df (pd.DataFrame): df of 5 min observations from 2015-2021
        df_years (pd.DataFrame): grouped by by yearl of NYSM observations at 1 min observations.
        self.filelist (List[str]): list of netcdf files
    """

    df: pd.DataFrame = None
    df_years: pd.DataFrame = None
    filelist: List[str] = field(default_factory=list, init=False)

    def nc_file_list(self) -> None:
        """
        Generate list of nc files in directory - encompasses all years/months/days available
        """
        self.filelist = []
        for root, _, files in os.walk(config.NC_FILE_DIR):
            self.filelist.extend(
                os.path.join(root, file) for file in files if file.endswith(".nc")
            )

    def grouped_df_year(self) -> pd.DataFrame:
        """
        Group dataframe of 5M NYSM files by year
        """
        self.df = pd.DataFrame(self.filelist, columns=["filename"])
        self.df["year"] = self.df["filename"].str.split("/").str[3]
        self.df_years = self.df.groupby("year")

    def read_data(self, group: pd.DataFrame) -> None:
        """
        Use xarray to open netcdf's in parallel and convert to pd.DataFrame

        - Timing on local machine for reference:
            2015: 13 sec
            2016: 84 sec
            2017: 47 sec
            2018: 34 sec
            2019: 29 sec
            2020: 29 sec
            2021: 29 sec
            2022: 5 sec (partial year)

        Args:
            group (pd.DataFrame): yearly df
        """
        self.df = xr.open_mfdataset(group["filename"], parallel=True).to_dataframe()

    def precip_diff(self) -> None:
        """
        Calculate precip over 5 min observations
        precip - since 00UTC every 5 mins
            resets to 0 at 00:05:00 or 00:00:00
        precip-total - Total Accumulated NRT (mm):
            Accumulated total non real time precipitation
            since the Pluvio started operation with a fixed delay of 5 minutes
        """
        self.df = self.df.reset_index()
        self.df = self.df.sort_values(by=["station", "time_5M"])
        self.df["precip_diff"] = self.df["precip"] - self.df["precip"].shift(1)

        # check when negative precip occurs
        neg_precip = self.df["time_5M"].dt.time[(self.df["precip_diff"] < 0.0)]
        print("negative precip occuring at minutes: ", neg_precip.unique())
        # mask and drop negative precip diff values from gauge resetting for new day
        self.df = self.df.mask(self.df["precip_diff"] < 0, np.nan).dropna(
            subset=["precip_diff"]
        )


def iterate_years_5M(nysm: NYSM_5M) -> None:
    """
    Loop over years of data and convert to parquet
    after dropping unused vars and calculating precip diff
    between times/rows

    Args:
        nysm (NYSM_5M_type): class instance for 5 min observations
    """
    nysm.nc_file_list()
    nysm.grouped_df_year()
    for year, group in nysm.df_years:
        start_time = time.time()
        print(f"[INFO] Reading files for {year}...")
        nysm.read_data(group)
        nysm.drop_unused_vars()
        nysm.precip_diff()  # 1M already has accumulation over the minute
        # nysm.write_parquet("5M", f"{year}")
        print_log(year, start_time)


def iterate_years_1M(nysm: NYSM_1M) -> None:
    """
    Loop over years of data and convert to parquet

    Args:
        nysm (NYSM_1M_type): class instance for 1 min observations
    """
    nysm.csv_file_list()
    print(nysm.filelist)
    nysm.grouped_df_year()
    for year, group in nysm.df_years:
        start_time = time.time()
        print(f"[INFO] Reading files for {year}...")
        nysm.read_data(group)
        nysm.write_parquet("1M", f"{year}")
        print_log(year, start_time)


def print_log(year: int, start_time: float) -> None:
    """
    Print when each year is done and how long the conversion took
    """
    print(
        "[INFO] Done reading %s files in %.2f seconds"
        % (year, time.time() - start_time)
    )


def main(temporal_resolution="5M"):
    """
    Read and convert all NYSM files to pandas dataframes in parallel based on year.
    Drop unused vars, calculate precip diff based on temporal resolution (1 or 5 min),
    and write df to parquet.
    """
    if temporal_resolution == "5M":
        nysm = NYSM_5M()
        iterate_years_5M(nysm)
    elif temporal_resolution == "1M":
        nysm = NYSM_1M()
        iterate_years_1M(nysm)
    else:
        print("temporal resolution needs to be either 1 minute or 5 minute")


if __name__ == "__main__":
    main()
