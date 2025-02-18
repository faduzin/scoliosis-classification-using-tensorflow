import pandas as pd


def class_remapping(data, column_name):
    data['class'] = (data['Scolio'] > 10).astype(int)
    return data

def test():
    print('Hello world!')