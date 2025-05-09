import pandas as pd

def drop_love_sur_train(df):
    df = df[df['label'] != 'love']
    df = df[df['label'] != 'surprise']
    return df

def drop_love_sur_val(val_df):
    val_df = val_df[val_df['label'] != 'love']
    val_df = val_df[val_df['label'] != 'surprise']
    return val_df

def drop_love_sur_test(ts_df):
    ts_df = ts_df[ts_df['label'] != 'love']
    ts_df = ts_df[ts_df['label'] != 'surprise']
    return ts_df