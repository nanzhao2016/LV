import os, sys
from prepare import preprocess

def start(path):
    try:
        path = path
    except IndexError:
        print("Input the local path.")
    else:
        files = os.listdir(path)
        print("Checking training and testing datasets.")
        if 'data_train.pkl' in files:
            print('Datasets preparation is done.')
        else:
            print('Start to prepare datasets.')
            preprocess()
            print('Datasets preparation is done.')
        
