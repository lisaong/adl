# Day 1 - Emotion Classification
# https://huggingface.co/nlp/viewer/?dataset=empathetic_dialogues

import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.callbacks import ModelCheckpoint

df_train = pd.read_csv('./empathetic_dialogues_train.csv')
print(df_train.head())
df_val = pd.read_csv('./empathetic_dialogues_val.csv')
print(df_val.head())

# vectorize the text
sequence_len = 20 # TODO: this can also be computed based on median length
vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=20)
