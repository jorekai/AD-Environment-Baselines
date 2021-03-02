import _pickle as pickle
import os
import time
import warnings
from typing import List, Tuple, Any, Union

import numpy as np
import pandas as pd
import sklearn.preprocessing
from pandas import DataFrame

import config as config
from src.Evaluator import Stats

scaler = sklearn.preprocessing.MinMaxScaler()


def get_file_list_from_directory(directory_path: str) -> List[str]:
    """
    Returns a List of filename paths
    :param directory_path: string of a path to recursively search through
    :return: List(filename_paths)
    """
    datasets = []
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            if file.find('.csv') != -1:
                datasets.append(os.path.join(subdir, file))
    return datasets


def get_frames_from_file_list(filelist: List[str], columns: Union[List[int], None], name: Union[str, None],
                              seperator: str = ",") -> List[
    DataFrame]:
    """
    This method returns a List of DataFrames and eventually stores it as pickle file
    :param filelist: List(paths)
    :param columns: List(indices)
    :param name: filename to be, if None the dataframes will not be stored
    :param seperator: ; or ,
    :return:
    """
    dataframes = []
    scaler = sklearn.preprocessing.MinMaxScaler()
    for i in range(len(filelist)):
        if not columns:
            series = pd.read_csv(filelist[i], header=0, sep=seperator)
            # comment this line to get first dataset column as feature
            series = series.drop(series.columns[0], axis=1)
        else:
            # series = pd.read_csv(filelist[i], usecols=columns, header=0, names=['value', 'anomaly'], sep=seperator)
            # scaler.fit(np.array(series['value']).reshape(-1, 1))
            # series['value'] = scaler.fit_transform(series[["value"]])
            series = pd.read_csv(filelist[i], usecols=columns, header=0, sep=seperator)
        dataframes.append(series)
    dataframes = fit_min_max_frames(dataframes)
    if not columns:
        dataframes = fit_min_max_frames(dataframes)
    if name is not None:
        store_object(dataframes, name)
    return dataframes


def get_bivariate_from_file_list(filelist: List[str], columns: List[int], name: str, seperator: str = ",") -> List[
    DataFrame]:
    """
    Get a List of bivariate DataFrames DEPRECATED
    """
    warnings.warn(
        "get_bivariate_from_file_list is deprecated, use get_frames_from_file_list instead",
        DeprecationWarning
    )
    dataframes = []
    for i in range(len(filelist)):
        series = pd.read_csv(filelist[i], usecols=columns, header=0, names=['sin', 'cos', 'anomaly'], sep=seperator)
        scaler.fit(np.array(series['sin']).reshape(-1, 1))
        series['sin'] = scaler.fit_transform(series[["sin"]])
        scaler.fit(np.array(series['cos']).reshape(-1, 1))
        series['cos'] = scaler.fit_transform(series[["cos"]])
        dataframes.append(series)
    store_object(dataframes, name)
    return dataframes


def init_dataframes(train_files: List[str], test_files: List[str], columns: List[int] = None) -> Tuple[
    List[DataFrame], List[DataFrame]]:
    """
    The initialization returns a Tuple of Lists of pandas Dataframes
    :param train_files: List of Strings (.csv) representing the path of each training file
    :param test_files: List of Strings (.csv) representing the path of each testing file
    :return: Tuple(List(pandas.DataFrame), List(pandas.DataFrame))
    """
    # if not already stored as pickle, then load files from path
    if not os.path.exists(
            config.ROOT_DIR + config.STORAGE_PATH + config.TRAIN_SET_NAME):
        train_dataframes = get_frames_from_file_list(filelist=train_files,
                                                     name=config.TRAIN_SET_NAME,
                                                     columns=columns)
    else:
        train_dataframes = load_object(config.TRAIN_SET_NAME)
    if not os.path.exists(config.ROOT_DIR + config.STORAGE_PATH + config.TEST_SET_NAME):
        test_dataframes = get_frames_from_file_list(filelist=test_files,
                                                    name=config.TEST_SET_NAME,
                                                    columns=columns)
    else:
        test_dataframes = load_object(config.TEST_SET_NAME)
        # DEBUG
        # self.test_dataframes = self.train_dataframes
    return train_dataframes, test_dataframes


def get_sample_size_overall(dataframes: List[DataFrame]) -> int:
    """
    Get the overall amount of samples in a list of dataframes
    :param dataframes: List(pandas.DataFrame)
    :return: int(amount_samples)
    """
    size = 0
    for frame in dataframes:
        size += frame['anomaly'].size
    return size


def get_filename_by_index(file_list: List[str], idx: int) -> str:
    """
    Get the Filename in a FileList by index
    :param file_list: List(paths)
    :param idx: int
    :return: str(filename)
    """
    return file_list[idx].rsplit('\\', 1)[-1].split(".")[0].capitalize()


def start_timer() -> float:
    """
    Start a timer object
    :return: floating point start time
    """
    return time.time()


def get_duration(timer: float) -> float:
    """
    Get the duration between a end and start time
    :param timer: floating point time
    :return: floating point time
    """
    return time.time() - timer


def store_object(obj: Any, filename: str) -> None:
    """
    Store a object as pickle
    :param obj: Any Object
    :param filename: to be filename
    :return: None
    """
    file = open(config.ROOT_DIR + config.STORAGE_PATH + filename, "wb")
    pickle.dump(obj, file)
    if config.VERBOSE:
        print(f"Successfully stored >> {filename} << to storage")


def load_object(filename: str) -> Any:
    """
    Load a previous pickled object
    :param filename: to load filename
    :return: Any Object
    """
    file = open(config.ROOT_DIR + config.STORAGE_PATH + filename, "rb")
    obj = pickle.load(file)
    if config.VERBOSE:
        print(f"Successfully loaded >> {filename} << from storage")
    return obj


def fit_min_max_frames(frames: List[DataFrame]) -> List[DataFrame]:
    """
    Performs a Min/Max Scaling on a List of DataFrames (between 0, 1)
    :param frames: List(pandas.DataFrame)
    :return: List(pandas.DataFrame)
    """
    for i, frame in enumerate(frames):
        for column in frame:
            if frame[column].dtype == np.float64 or frame[column].dtype == np.int64:
                scaler.fit(np.array(frame[column]).reshape(-1, 1))
                frame[column] = scaler.fit_transform(frame[[column]])
    return frames


def init_stats(frames: List[DataFrame]) -> List[Stats]:
    """
    Creates a Statistics array with the same indices as the dataframe array
    :param frames: List of Dataframes
    :return: List of Stats Object
    """
    stats = []
    for frame in frames:
        stats.append(Stats(frame))
    return stats
