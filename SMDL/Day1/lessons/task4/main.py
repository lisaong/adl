# LSTM demonstration
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# using LSTM to learn a function
def f(x):
   return np.log(x)


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

# RNN needs a 3D tensor
# For time series:
# [batch, sequence_len] -> [batch, sequence_len, features]
# For text, the embedding layer takes care of it:
# [batch, sequence_len, word_vector_size]
n_features = 1
X = X.reshape(X.shape[0], sequence_len, n_features)

# define the model
model_input = Input(shape=(sequence_len, n_features,))
x = LSTM(8)(model_input)
x = Dense(1)(x)
model = Model(model_input, x)
model.summary()

# train
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=10, batch_size=2)

# predictions
preds = model.predict(X)
plt.title('RNN predicting log(x)')
plt.plot(my_data, label='actual')
plt.plot(preds, label='predicted')
plt.xlabel('x')
plt.legend()
plt.savefig('predictions.png')
plt.show()
