import pickle
import pandas as pd
import pathlib


def open_data(directory, name, separator=None):
    path = str(pathlib.Path.joinpath(directory, name))
    extension = path[-4:]

    if extension == '.pcl':
        with open(path, 'rb') as handle:
            content = pickle.load(handle)
    elif extension == '.txt':
        content = pd.read_csv(path, sep=separator, header=None)
    elif extension == '.csv':
        content = pd.read_csv(path, sep=separator, header=None)
    else:
        print('Cannot open files with extension: ' + extension)

    return content