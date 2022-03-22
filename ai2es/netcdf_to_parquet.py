"""create dataframes of mesonet data for every station between 2015 and 2020
two separate dataframes saved based on whether precip or no precip from rain gauge"""

import pandas as pd
import numpy as np
import xarray as xr
import time
from fastparquet import write
import os
from dask.distributed import Client, LocalCluster
import datetime


def start_client():
    cluster = LocalCluster(n_workers=8)  # http://127.0.0.1:8787/status for dashboard
    client = Client(cluster)


class NYSM:
    def __init__(self, nc_file_dir="../NYSM/archive/nysm/netcdf/proc/"):
        self.nc_file_dir = nc_file_dir
        self.df = None

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

    def five_min_precip(self):
        """calculate precip over 5 mins
        precip - since 00UTC every 5 mins
            resets to 0 at 00:05:00
        precip-total - Total Accumulated NRT (mm):
            Accumulated total non real time precipitation
            since the Pluvio started operation with a fixed delay of 5-minutes
        """
        self.df = self.df.reset_index()
        self.df = self.df.sort_values(by=["station", "time_5M"])
        self.df["precip_5min"] = self.df["precip"] - self.df["precip"].shift(1)

        # check that negative precip only occurs at minute 05 from reset
        print(
            self.df["time_5M"].dt.time[
                (self.df["precip_5min"] < 0.0) & (self.df["time_5M"].dt.minute != 5)
            ]
        )
        # mask and drop negative 5 min precip values at 00:05:00 from gauge resetting for new day
        self.df = self.df.mask(self.df["precip_5min"] < 0, np.nan).dropna(
            subset=["precip_5min"]
        )

    def image_paths(self, photo_dir="../cam_photos"):
        """convert timestamp into image filename and append to df if there is a corresponding image"""
        timestamp = self.df["time_5M"]
        date_path = f"{photo_dir}/{timestamp.strftime('%Y')}/{time.strftime('%m')}/{time.strftime('%d')}"

        # file_path = date_path+'/'+x['station']+'/'+time.strftime('%Y%m%dT%H%M')+'*'
        # if(os.path.exists(site_path) and len(glob.glob(file_path))>0):
        #     return glob.glob(file_path)[0]
        # else: return None

    def write_parquet(self, path: str, filename: str):
        """write combined dataframe from multiple years/stations to parquet"""
        write(f"{path}/{filename}.parquet", self.df)


def main():
    """read and convert all netcdf NYSM files to pandas dataframes in parallel based on year"""
    nysm = NYSM()
    start_client()  # start dask client

    df = nysm.grouped_df_year()
    for year, group in df:
        start_time = time.time()
        print(f"[INFO] Reading netcdf files for {year}...")

        nysm.df = xr.open_mfdataset(group["filename"], parallel=True).to_dataframe()
        nysm.drop_unused_vars()
        nysm.five_min_precip()
        # self.write_parquet("../NYSM/", f"{year}")
        print(
            "[INFO] Done reading %s files in %.2f seconds"
            % (year, time.time() - start_time)
        )


if __name__ == "__main__":
    main()
