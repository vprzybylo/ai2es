"""
create dataframes of mesonet data for every station for 5 min observations
"""

from re import T
import pandas as pd
import numpy as np
import xarray as xr
import time
import pyarrow.parquet as pq
import pyarrow as pa
import os
from dask.distributed import Client, LocalCluster


def start_client():
    cluster = LocalCluster(n_workers=4)  # http://127.0.0.1:8787/status for dashboard
    client = Client(cluster)


class NYSM:
    """base class to drop unused vars, calculate precip diff based on
    temporal resolution, and convert to parquet files by year"""

    def drop_unused_vars(self):
        """only keep certain important variables"""
        keep_vars = [
            "tair",
            "ta9m",
            "precip",
            "precip_total",
            "precip_max_intensity",
        ]
        self.df = self.df[keep_vars]

    def write_parquet(self, temporal_resolution: str, filename: str):
        """write combined dataframe from multiple years/stations to parquet"""

        if temporal_resolution == "1M":
            self.df["datetime"] = pd.to_datetime(self.df["datetime"], errors="coerce")
            path = "../mesonet_parquet_1M"
        elif temporal_resolution == "5M":
            path = "../mesonet_parquet_5M"
        else:
            print("temporal resolution must be either 1M or 5M")

        pq.write_table(pa.Table.from_pandas(self.df), f"{path}/{filename}.parquet")


class NYSM_1M(NYSM):
    """convert csv's to parquet for 1 min observations from 2017-2021
    Copied from /raid/lgaudet/precip/Precip/NYSM_1min_data to local machine in /NYSM/1_min_obs
    Manually removed the few rows with 7 columns instead of 5 and values that were not floats (e.g., 0.00.00)"""

    def __init__(self, csv_file_dir="../NYSM/1_min_obs"):
        self.csv_file_dir = csv_file_dir
        self.df = None

    def read_data(self, group: pd.DataFrame):
        """read csv's for each day and concatenate into df
        for reference on local machine:
            2017: 25.83 sec
            2018: 28.43 sec
            2019: 27.28 sec
            2020: 28.32 sec
            2021: 27.41 sec
        """
        files_in_year = []

        for filename in group["filename"]:
            df = pd.read_csv(
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
            files_in_year.append(df)
        self.df = pd.concat(files_in_year, axis=0, ignore_index=True)

    def csv_file_list(self):
        """generate list of csv files of 1 min observations from csv_file_dir"""
        filelist = []
        for root, dirs, files in os.walk(self.csv_file_dir):
            filelist.extend(os.path.join(root, file) for file in files)
        return filelist

    def grouped_df_year(self):
        """group dataframe of NYSM files by year"""
        df = pd.DataFrame(self.csv_file_list(), columns=["filename"])
        df["year"] = df["filename"].str.split("/").str[4].str[:4]
        df = df.groupby("year")
        return df


class NYSM_5M(NYSM):
    """convert netcdf's to parquet for 5 min observations from 2015-2021"""

    def __init__(self, nc_file_dir="../NYSM/archive/nysm/netcdf/proc/"):
        self.nc_file_dir = nc_file_dir
        self.df = None

    def read_data(self, group: pd.DataFrame):
        """use xarray to open netcdf's in parallel and convert to pd.DataFrame
        timing on local machine for reference:
            2015: 13 sec
            2016: 84 sec
            2017:  sec
            2018: 37.31 sec
            2019: 30.78 sec
            2020: 31.30 sec
            2021: 30.31 sec
            2022: 6.25 sec (partial year)
        """
        self.df = xr.open_mfdataset(group["filename"], parallel=True).to_dataframe()

    def nc_file_list(self):
        """generate list of nc files in directory - encompasses all years/months/days available"""
        filelist = []
        for root, dirs, files in os.walk(self.nc_file_dir):
            filelist.extend(os.path.join(root, file) for file in files)
        return filelist

    def grouped_df_year(self):
        """group dataframe of NYSM files by year"""
        df = pd.DataFrame(self.nc_file_list(), columns=["filename"])
        df["year"] = df["filename"].str.split("/").str[6]
        df = df.groupby("year")
        return df

    def precip_diff(self):
        """calculate precip over 5 min observations
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


def iterate_years(nysm, temporal_resolution: str):
    """
    read data for each year, drop unused vars,
    calculate precip diff based on temporal resolution (1 or 5 min),
    write df to parquet
    """
    df = nysm.grouped_df_year()
    for year, group in df:
        start_time = time.time()
        print(f"[INFO] Reading files for {year}...")
        nysm.read_data(group)
        if temporal_resolution == "5M":
            nysm.drop_unused_vars()
            nysm.precip_diff()  # 1M already has accumulation over the minute
        nysm.write_parquet(temporal_resolution, f"{year}")
        print(
            "[INFO] Done reading %s files in %.2f seconds"
            % (year, time.time() - start_time)
        )


def main(temporal_resolution: str = "5M"):
    """read and convert all NYSM files to pandas dataframes in parallel based on year
    pass in '5M' for netcdf 5 minute resolution"""
    try:
        nysm = (
            NYSM_1M()
            if temporal_resolution == "1M"
            else NYSM_5M()
            if temporal_resolution == "5M"
            else None
        )
    except ValueError:
        if nysm is None:
            print("temporal resolution needs to be either 1 minute or 5 minute")

    # start_client()  # start dask client
    iterate_years(nysm, temporal_resolution)


if __name__ == "__main__":
    main()
