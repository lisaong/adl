# Hello tensorflow
# Toy MLP regressor to predict stock price from past 5 days

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pickle
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


def save(key_values: dict):
    model_artifacts.update(key_values)
    pickle.dump(model_artifacts, open('model_artifacts.pkl', 'wb'))


def convert_to_js_model(model_h5: str, dest='tfjs_artifacts'):
    # convert the model to tensorflow js
    if not os.path.exists(dest):
        os.mkdir(dest)
    best_model = load_model(model_h5)
    tfjs.converters.save_keras_model(best_model, dest)
    print(f'converted {model_h5} to {dest}')


# Import the stock price data
df = pd.read_csv('./HistoricalQuotes_TSLA.csv', sep=', ', index_col=0, parse_dates=True)

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
X = df_windowed.loc[:, features]
y = df_windowed.loc[:, target]
X_train, X_test, y_train, y_test = train_test_split(X, y)

# scale
X_min = tf.reduce_min(X_train)
X_max = tf.reduce_max(X_train)

X_train = (X_train.values - X_min) / (X_max - X_min)
X_test = (X_test.values - X_min) / (X_max - X_min)

# save for deployment
save({'X_min': X_min, 'X_max': X_max})

# create tensorflow model
model_input = Input(shape=(window_size,))
x = Dense(window_size, activation='relu')(model_input)
x = Dense(1)(x)
model = Model(model_input, x)
print(model.summary())

# compile
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

mc = ModelCheckpoint('hello_mlp.h5')
history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[mc])

# plot learning curve
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epochs')
plt.ylabel('mean_squared_logarithmic_error')
plt.show()

convert_to_js_model('hello_mlp.h5')

# now we are ready to deploy to node.js
