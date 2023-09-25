import os
import shutil
import time
import pickle
import datetime
import subprocess
from pathlib import Path
import ast
import pandas as pd

import open3d as o3d
import numpy as np
from typing import Generator


def read_point_cloud(filename: str) -> np.ndarray:
    # Load point cloud from file using Open3D
    point_cloud = o3d.io.read_point_cloud(filename)

    # Extract xyz coordinates as numpy array
    points = np.asarray(point_cloud.points)

    return points


def read_files(path: str) -> Generator:
    """
    :path: str path to the 3d point cloud files (.ply)
    :returns: Generator with two return values (name of parent folder / ego + frame and point clouds as Dict with the filename as key)
    """
    # loop through all the folders and files in the directory
    for par_folder in os.listdir(path):
        pt_clouds = {}
        dir_path = os.path.join(path, par_folder)
        
        for name in os.listdir(dir_path):
            # get the full path of the file
            file_path = os.path.join(dir_path, name)
            # check if the file is a PLY file
            if not os.path.splitext(file_path)[1] == '.ply':
                continue
                
            pt_cloud = read_point_cloud(file_path)

            obj_type = name.split('_')[0]
            if obj_type not in pt_clouds:
                pt_clouds[obj_type] = {}
            
            pt_clouds[obj_type][name] = pt_cloud

        yield par_folder, pt_clouds


def delete_all_items_in_dir(directory: str) -> None:
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)  # Remove the file
        elif os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
            except:
                time.sleep(5)
                shutil.rmtree(item_path)


def get_filesize(file: str) -> None:
    file_size = os.path.getsize(f'{file}.pkl')

    # print the size of the file in a human-readable format
    if file_size < 1024:
        print(f"{file_size} bytes")
    elif file_size < 1024 * 1024:
        print(f"{file_size / 1024:.1f} KB")
    elif file_size < 1024 * 1024 * 1024:
        print(f"{file_size / 1024 / 1024:.1f} MB")
    else:
        print(f"{file_size / 1024 / 1024 / 1024:.1f} GB")


def load_dataframes_from_pickle(filename: str) -> pd.DataFrame:
    dataframes = list()
    with open(filename, 'rb') as file:
        while True:
            try:
                df = pickle.load(file)
                dataframes.append(df)
            except EOFError:
                break
    return dataframes
