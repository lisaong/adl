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

# NLP datasets from https://huggingface.co
from datasets import load_dataset

# this dataset is large, for demonstration purposes,
# we'll just load the first 10% of train and validation sets
train_ds, val_ds = load_dataset('empathetic_dialogues', split=['train[:10%]', 'validation[:10%]'])

print(train_ds, val_ds)

# vectorize the text
sequence_len = 20 # TODO: this can also be computed based on median length
vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=20)


