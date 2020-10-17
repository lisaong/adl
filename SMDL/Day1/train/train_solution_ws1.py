# Day 1 - Emotion Classification
# https://huggingface.co/nlp/viewer/?dataset=empathetic_dialogues

# Instructions:
# 1. Go through the lessons before you start
# 2. Search for the TODOs, replace _ANS_ with your answers so that the code will run
# 3. Submit your completed train.py with learning_curve.png

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from nltk import word_tokenize
import pickle
import os

# globals
MODEL_ARTIFACTS = dict()
MODEL_DIR = os.path.join('..', 'app', 'demo', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)


def save_artifacts(key_values: dict, dest='model_artifacts.pkl'):
    MODEL_ARTIFACTS.update(key_values)
    pickle.dump(MODEL_ARTIFACTS, open(os.path.join(MODEL_DIR, dest), 'wb'))


df_train = pd.read_csv('../data/empathetic_dialogues_train.csv', index_col=0)
print(df_train.head())
print(df_train.info())

df_val = pd.read_csv('../data/empathetic_dialogues_val.csv', index_col=0)
print(df_val.head())
print(df_val.info())

# compute median length of the text
sequence_len = int(df_train['prompt'].apply(lambda x: len(word_tokenize(x))).median())
print(f'Sequence length: {sequence_len}')

# visualise the imbalance
plt.title('Label Counts')
plt.hist(df_train['context'], label='train')
plt.xticks(rotation=45, ha='right')
plt.hist(df_val['context'], label='validation')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

print(df_train['context'].value_counts())
print(df_val['context'].value_counts())

# vectorize the text
# the dataset contains repetition of prompts, that's ok as long as the labels are consistent.
# In NN training, as long as we shuffle the data, we should be fine even if the same data
# is seen multiple times.
vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=sequence_len)
vectorizer.adapt(df_train['prompt'].values)
X_train = vectorizer(df_train['prompt']).numpy()
X_val = vectorizer(df_val['prompt']).numpy()
print(vectorizer.get_config()) # configuration
print(vectorizer.get_vocabulary()[:10]) # first 10 words
save_artifacts({'vectorizer_vocab': vectorizer.get_vocabulary()})
save_artifacts({'vectorizer_config': vectorizer.get_config()})

# encode and convert labels to categorical
le = LabelEncoder()
y_train = to_categorical(le.fit_transform(df_train['context']))
y_val = to_categorical(le.transform(df_val['context']))
print(y_train)
print(y_val)
print(le.classes_)
save_artifacts({'label_encoder': le})

# define our model
vocab_len = len(vectorizer.get_vocabulary())
embedding_len = 75
num_classes = len(le.classes_)
model_input = Input(shape=(sequence_len,), dtype='int64')

# TODO: Replace _ANS_ with your answers
x = Embedding(input_dim=vocab_len, output_dim=embedding_len)(model_input)
x = LSTM(8, activation='tanh')(x)
x = Flatten()(x)
x = Dense(8, activation='relu')(x)
x = Dropout(.5)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(model_input, x)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])

# periodically saves the best model every epoch
mc = ModelCheckpoint(os.path.join(MODEL_DIR, 'lstm.h5'),
                     monitor='val_acc', save_best_only=True)

history = model.fit(X_train, y_train, epochs=20, batch_size=32, shuffle=True,
                    validation_data=(X_val, y_val),
                    callbacks=[mc])

plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='validation')
plt.title('Learning Curve')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('learning_curve.png')
plt.show()

# get metrics
best_model = load_model(os.path.join(MODEL_DIR, 'lstm.h5'))
y_pred_train = best_model.predict(X_train)
y_pred_val = best_model.predict(X_val)
print(classification_report(y_train.argmax(axis=1), y_pred_train.argmax(axis=1)))
print(classification_report(y_val.argmax(axis=1), y_pred_val.argmax(axis=1)))