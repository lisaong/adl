# pad_sequences demonstration

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

my_sequences = np.array([[4, 1, 2, 3], [3, 1, 2]])

print('Original sequences', my_sequences)

print('=======================')

print('maxlen=5, padding="pre"')

print(pad_sequences(my_sequences, maxlen=5))

print('=======================')

print('maxlen=5, padding="post"')

print(pad_sequences(my_sequences, maxlen=5, padding='post'))

print('=======================')

print('maxlen=3, truncating="pre"')

print(pad_sequences(my_sequences, maxlen=3, truncating='pre'))

print('=======================')

print('maxlen=3, truncating="post"')

print(pad_sequences(my_sequences, maxlen=3, truncating='post'))
