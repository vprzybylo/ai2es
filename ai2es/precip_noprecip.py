"""create dataframes of mesonet data for every station between 2015 and 2020
two separate dataframes saved based on whether precip or no precip from rain gauge"""

import pandas as pd
import numpy as np
import xarray as xr
import time
from fastparquet import write


class NYSM:
    def __init__(self):
        self.df_2021 = None
        self.df_2015_2020 = None
        self.df = None

    def create_2015_2020_df(self, filename: str):
        """read parquet file from 2015 to 2020 and reset index based on station and time"""
        self.df_2015_2020 = (
            pd.read_parquet(f"../NYSM/mesonet_parquet/{filename}.parquet")
            .drop(columns="index")
            .reset_index()
        )
        self.df_2015_2020["station"] = self.df_2015_2020["station"].str.decode(
            encoding="UTF-8"
        )
        self.df_2015_2020 = self.df_2015_2020.set_index(["station", "time_5M"])

    def netcdf_to_df(self, indir="../NYSM/archive/nysm/netcdf/proc/", year=2021):
        """load mesonet netcdf from 2021"""
        self.df_2021 = pd.concat(
            [
                xr.open_mfdataset(
                    f"{indir}/{year}/{str(month).zfill(2)}/*.nc"
                ).to_dataframe()
                for month in range(1, 13)
            ]
        )
        print(xr.open_dataset(f"{indir}/{year}/01/20210101.nc"))

    def concat_years(self):
        """concatenate 2015-2021 NYSM data"""
        self.df = pd.concat([self.df_2015_2020, self.df_2021])

    def drop_unused_vars(self):
        """drop unused variables"""
        keep_vars = [
            "station",
            "time_5M",
            "tair",
            "ta9m",
            "precip",
            "precip_total",
            "precip_max_intensity",
            "snow_depth",
        ]
        self.df = self.df[keep_vars]

    def five_min_precip(self):
        """calculate precip over 5 mins
        precip - since 00UTC every 5 mins
        precip-total - Total Accumulated NRT (mm):
            Accumulated total non real time precipitation
            since the Pluvio started operation with a fixed delay of 5-minutes
        """
        self.df = self.df.set_index("station").sort_values(by="time_5M")
        self.df["precip_shifted"] = df.groupby(["station"])["precip"].shift(1)
        self.df["precip_5min"] = df["precip"] - df["precip_shifted"]
        self.df = self.df.drop(columns="precip_shifted").dropna(subset=["precip_5min"])

    def image_paths(self, photo_dir="../cam_photos"):
        """convert timestamp into image filename and append to df"""
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
    start_time = time.time()

    nysm = NYSM()
    nysm.create_2015_2020_df("201508-202012")
    print(f"done reading 2015-2020 in {time.time()-start_time} seconds")
    print("2015-2020", nysm.df_2015_2020)

    nysm.netcdf_to_df()
    print(f"done reading 2021 in {time.time()-start_time} seconds")
    print("2021", nysm.df_2021)

    nysm.concat_years()
    nysm.drop_unused_vars()
    nysm.five_min_precip()
    nysm.write_parquet("../NYSM/mesonet_parquet/", "201508_202112")


if __name__ == "__main__":
    main()
