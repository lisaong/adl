# Hello tensorflow
# Toy MLP regressor to predict stock price from past 5 days

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflowjs as tfjs
import os

# remove the $ sign
def clean(text: str):
    return text.replace('$', '')


model_artifacts = dict()


def save(key_values: dict, dest='model_artifacts.json'):
    model_artifacts.update(key_values)
    with open(dest, 'w') as f:
        json.dump(model_artifacts, f)


def convert_to_js_model(model_h5: str, dest='tfjs_artifacts'):
    # convert the model to tensorflow js
    if not os.path.exists(dest):
        os.mkdir(dest)
    best_model = load_model(model_h5)
    tfjs.converters.save_keras_model(best_model, dest)
    print(f'converted {model_h5} to {dest}')


# Import the stock price data
df = pd.read_csv('./HistoricalQuotes_TSLA.csv', sep=', ', index_col=0, parse_dates=True, engine='python')

df['Close/Last'] = pd.to_numeric(df['Close/Last'].apply(clean))
print(df['Close/Last'].head())

# Create windows and target
window_size = 5
df_windowed = pd.concat([df['Close/Last'].shift(-i) for i in range(window_size + 1)], axis=1).dropna()
df_windowed.columns = [f'x[t+{i}]' for i in range(window_size)] + ['y']
print(df_windowed.head())

# Train test split
features = df_windowed.columns[:-1]
target = df_windowed.columns[-1]
X = df_windowed.loc[:, features].values
y = df_windowed.loc[:, target].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

# save for deployment
save({'X_test': X_test.tolist(), 'y_test': y_test.tolist()},
     dest='./hello_tensorflow_app/model_artifacts.json')

# create tensorflow model
model_input = Input(shape=(window_size,))
x = Dense(window_size, activation='relu')(model_input)
x = Dense(1)(x)
model = Model(model_input, x)
print(model.summary())

# compile
model.compile(optimizer='adam', loss='mse')

mc = ModelCheckpoint('hello_mlp.h5', save_best_only=True)
history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[mc])

# plot learning curve
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.legend()
plt.show()

convert_to_js_model('hello_mlp.h5', dest='./hello_tensorflow_app/model')

# now we are ready to deploy to node.js
