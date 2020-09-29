# dataset sources:
# https://www.kingjamesbibleonline.org
# http://www.bibledbdata.org/onlinebibles/german_l/43_001.htm

import pandas as pd


def get_data(val_split=.1):
    df_english = pd.read_csv('../data/kjv.txt', sep='  ', index_col=0, header=None, engine='python')
    df_german = pd.read_csv('../data/luther.txt', sep='  ', index_col=0, header=None, engine='python')
    df_data = pd.concat([df_english, df_german], axis=1)
    df_data.columns = ['english', 'german']
    split_threshold = int(val_split * len(df_data))

    # train, val
    return df_data.iloc[:-split_threshold], df_data.iloc[-split_threshold:]
