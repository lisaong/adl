# Returning sequences demonstration
# Note: you can replace LSTM with GRU
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten


def f(t):
    return np.log(t)


my_data = f(np.arange(0.5, 10, .2))

# create windowed sequences
sequence_len = 3
s = pd.Series(my_data)
df_windowed = pd.concat([s.shift(-i) for i in range(sequence_len+1)], axis=1).dropna()
print(df_windowed.head())

# pick all except the last columns as X
# pick the last column as y
X = df_windowed.iloc[:, :-1].values
y = df_windowed.iloc[:, -1].values

# reshape to (batch, timesteps, features)
n_features = 1
X_rnn = X.reshape(X.shape[0], sequence_len, n_features)

# Model 1
model_input = Input(shape=(sequence_len, n_features,))
x = LSTM(8, activation='relu',
         return_sequences=True)(model_input)  # can also use GRU here
x = Flatten()(x)  # flatten the timesteps dimension
x = Dense(1)(x)
model1 = Model(model_input, x)
model1.summary()

# compile, fit, predict
model1.compile(loss='mse', optimizer='adam')
model1.fit(X_rnn, y, epochs=20, batch_size=2)
preds1 = model1.predict(X_rnn)
preds1 = np.concatenate((X[0], preds1.flatten()))

# Model 2
# define the model
model_input = Input(shape=(sequence_len, n_features,))
x = LSTM(8, activation='relu')(model_input)  # can also use GRU here
x = Dense(1)(x)
model2 = Model(model_input, x)
model2.summary()

# compile, fit, predict
model2.compile(loss='mse', optimizer='adam')
model2.fit(X_rnn, y, epochs=20, batch_size=2)
preds2 = model2.predict(X_rnn)
preds2 = np.concatenate((X[0], preds2.flatten()))

plt.title('RNN predicting log(t)')
plt.plot(my_data, label='actual')
plt.plot(preds1, label='return_sequences=True')
plt.plot(preds2, label='return_sequences=False')
plt.xlabel('time t')
plt.legend()
plt.savefig('predictions.png')
plt.show()
