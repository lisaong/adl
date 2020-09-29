# dataset sources:
# https://www.kingjamesbibleonline.org/john-1-parallel-kjv-greek/
# https://www.kingjamesbibleonline.org/john-2-parallel-kjv-greek/
# https://www.kingjamesbibleonline.org/john-3-parallel-kjv-greek/
# https://www.kingjamesbibleonline.org/john-4-parallel-kjv-greek/

import pandas as pd


def get_data(val_split=.1):
    df_english = pd.read_csv('../data/kjv.txt', sep='  ', index_col=0, header=None, engine='python')
    df_greek = pd.read_csv('../data/greek.txt', sep='  ', index_col=0, header=None, engine='python')
    df_data = pd.concat([df_english, df_greek], axis=1)
    df_data.columns = ['english', 'greek']
    split_threshold = int(val_split * len(df_data))

    # train, val
    return df_data.iloc[:-split_threshold], df_data.iloc[-split_threshold:]
