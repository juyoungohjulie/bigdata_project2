import pandas as pd

def data_loader():
    train_df = pd.read_csv("train.txt", delimiter=';', header=None, names=['sentence','label'])
    val_df = pd.read_csv("val.txt", delimiter=';', header=None, names=['sentence','label'])
    ts_df = pd.read_csv("test.txt", delimiter=';', header=None, names=['sentence','label'])

    return train_df, val_df, ts_df
    