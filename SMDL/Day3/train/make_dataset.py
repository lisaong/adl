# Dataset:
# source: https://www.kingjamesbibleonline.org
# target: http://www.bibledbdata.org/onlinebibles/german_l/43_001.htm

import pandas as pd


def get_data(data_path, val_split=.1):
    df_source = pd.read_csv(f'{data_path}/kjv.txt', sep='  ', index_col=0, header=None, engine='python')
    df_target = pd.read_csv(f'{data_path}/luther.txt', sep='  ', index_col=0, header=None, engine='python')
    df_data = pd.concat([df_source, df_target], axis=1)
    df_data.columns = ['source', 'target']
    split_threshold = int(val_split * len(df_data))

    # train, val
    return df_data.iloc[:-split_threshold], df_data.iloc[-split_threshold:]
