import os
from zipfile import ZipFile

import pandas as pd


def read_zipped_data(src_filename: str, dest_extention: str) -> pd.DataFrame:
    data_dir_path = "../data/"

    with ZipFile(data_dir_path + src_filename, "r") as zFile:
        zFile.extractall(path=data_dir_path)

    unzipped_filename, _ = os.path.splitext(src_filename)

    df = pd.DataFrame()

    unzipped_filename_full = data_dir_path + unzipped_filename + "." + dest_extention

    if dest_extention == "csv":
        df = pd.read_csv(unzipped_filename_full)
    elif dest_extention == "json":
        try:
            df = pd.read_json(unzipped_filename_full, lines=True)
        except:
            df = pd.read_json(unzipped_filename_full)

    os.remove(unzipped_filename_full)

    return df
