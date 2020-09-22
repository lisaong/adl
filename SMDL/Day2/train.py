# Day 2 - Text Generation
# https://huggingface.co/nlp/viewer/?dataset=empathetic_dialogues
# https://www.tensorflow.org/tutorials/text/text_generation

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report
from nltk import word_tokenize
import pickle
import os

# globals
MODEL_ARTIFACTS = dict()
MODEL_DIR = os.path.join('app', 'demo', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)


def save_artifacts(key_values: dict, dest='model_artifacts.pkl'):
    MODEL_ARTIFACTS.update(key_values)
    pickle.dump(MODEL_ARTIFACTS, open(os.path.join(MODEL_DIR, dest), 'wb'))


df_train = pd.read_csv('../Day1/empathetic_dialogues_train.csv', index_col=0)
print(df_train.head())
print(df_train.info())

df_val = pd.read_csv('../Day1/empathetic_dialogues_val.csv', index_col=0)
print(df_val.head())
print(df_val.info())

# compute median length of the text
sequence_len = int(df_train['utterance'].apply(lambda x: len(word_tokenize(x))).median())
print(f'Sequence length: {sequence_len}')

# vectorize the text