# Day 1 - Emotion Classification
# https://huggingface.co/nlp/viewer/?dataset=empathetic_dialogues

import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint


