import os
import pickle
import json

def load_pkl_data(filePath):
    with open(filePath, 'rb') as fp:
        data_pkl = fp.read()
    print(f'loaded {filePath}')
    return pickle.loads(data_pkl)


def save_pkl_data(data, filePath):

    data_pkl = pickle.dumps(data)
    with open(filePath, 'wb') as fp:
        fp.write(data_pkl)
    print(f'saved {filePath}')