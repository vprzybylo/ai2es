"""
- THIS FILE SHOUlD BE ALTERED AND RENAMED config.py FOR EACH USER
- config.py in .gitignore to avoid version changes upon specifications
- holds all user-defined variables
- treated as global variables that do not change in any module
- used in each module through 'import config as config'
- call using config.VARIABLE_NAME
isort:skip_file
"""

# /raid/NYSM/archive/nysm/netcdf/proc/ on hulk
nc_file_dir = "/ai2es/5_min_obs"

# /raid/lgaudet/precip/Precip/NYSM_1min_data on hulk
csv_file_dir = "/ai2es/1_min_obs"

# where to write time  matched data
write_path = f"/ai2es/matched_parquet/"

# root dir to raw images (before each year subdir)
photo_dir = "/ai2es/cam_photos/"

# where the mesonet obs live in parquet format
# output from nysm_obs_to_parquet
parquet_dir = "/ai2es/mesonet_parquet_1M"
