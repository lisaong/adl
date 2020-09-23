# Day 2 - Text Generation
# https://huggingface.co/nlp/viewer/?dataset=empathetic_dialogues
# https://www.tensorflow.org/tutorials/text/text_generation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, GRU, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
import os
import json

# globals
MODEL_ARTIFACTS = dict()
MODEL_DIR = os.path.join('app', 'demo', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)


def save_artifacts(key_values: dict, dest='model_artifacts.pkl'):
    MODEL_ARTIFACTS.update(key_values)
    pickle.dump(MODEL_ARTIFACTS, open(os.path.join(MODEL_DIR, dest), 'wb'))


# we'll just use the training set for this task because we are predicting
# the next word (generating our own target).
df_train = pd.read_csv('../Day1/empathetic_dialogues_train.csv', index_col=0)
print(df_train.head())
print(df_train.info())

# vectorize the text, but without padding
# we will pad later on after we've generated the input and next-word (target) sequences
# this allows us to perform pre-padding for shorter input sequences
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, lower=True)
tokenizer.fit_on_texts(df_train['utterance'].values)

sequences = tokenizer.texts_to_sequences(df_train['utterance'].values)

# compute the median length
sequence_len = int(np.median(np.array([len(s) for s in sequences])))
print(sequence_len)

# pad sequences to 1.25 * sequence_len
# 125% is a heuristic - we don't want too much pre-paddings, but we also want the
# sequences to extend a bit beyond the window size, so that we have enough to predict
# the next word.
padded_len = int(sequence_len*1.25)
print(padded_len)
sequences = pad_sequences(sequences, maxlen=padded_len, padding='pre')

# create our dataset
# X: sequence_len (sliding window)
# y: next word
X = []
y = []
for s in sequences:
    for j in range(len(s) - sequence_len):
        X.append(np.array(s[j:j+sequence_len]))
        y.append(s[j+sequence_len])

y_cat = to_categorical(y)

X_train, X_val, y_train, y_val = train_test_split(np.array(X), np.array(y_cat))
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# save our tokenizer configuration
tokenizer_config = json.loads(tokenizer.to_json())
save_artifacts({'tokenizer_config': tokenizer_config})

# create our model
embedding_len = 100
batch_size = 16
num_outputs = y_cat.shape[1]

model_input = Input(shape=(sequence_len,), dtype='int64')
x = Embedding(vocab_size, embedding_len, input_length=sequence_len)(model_input)
x = GRU(1024, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(num_outputs, activation='softmax')(x)
model = Model(model_input, x)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])

mc = ModelCheckpoint(os.path.join(MODEL_DIR, 'gru.h5'),
                     monitor='val_acc', save_best_only=True)

history = model.fit(X_train, y_train, epochs=5, batch_size=batch_size, shuffle=True,
                    validation_data=(X_val, y_val),
                    callbacks=[mc])

plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='validation')
plt.title('Learning Curve')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

