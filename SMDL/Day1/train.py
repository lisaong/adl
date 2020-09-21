# Day 1 - Emotion Classification
# https://huggingface.co/nlp/viewer/?dataset=empathetic_dialogues

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, GRU, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from nltk import word_tokenize
import pickle

# globals
MODEL_ARTIFACTS = dict()


def save_artifacts(key_values: dict, dest='model_artifacts.pkl'):
    MODEL_ARTIFACTS.update(key_values)
    pickle.dump(MODEL_ARTIFACTS, open(dest, 'wb'))


df_train = pd.read_csv('./empathetic_dialogues_train.csv', index_col=0)
print(df_train.head())
df_val = pd.read_csv('./empathetic_dialogues_val.csv', index_col=0)
print(df_val.head())

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
embedding_len = 50
num_classes = len(le.classes_)
model_input = Input(shape=(sequence_len,), dtype='int64')
x = Embedding(vocab_len, embedding_len, input_length=sequence_len)(model_input)
x = GRU(32, activation='relu', recurrent_dropout=.5, return_sequences=True)(x)
x = GRU(32, activation='relu', recurrent_dropout=.5)(x)
x = Flatten()(x)
x = Dense(num_classes*2, activation='relu')(x)
x = Dropout(.5)(x)
x = Dense(num_classes, activation='softmax')(x)
model = Model(model_input, x)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc', Precision(), Recall()])

mc = ModelCheckpoint('gru.h5', monitor='val_acc', save_best_only=True)

history = model.fit(X_train, y_train, epochs=30, batch_size=32, shuffle=True,
                    validation_data=(X_val, y_val),
                    callbacks=[mc])

plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='validation')
plt.title('Learning Curve')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
