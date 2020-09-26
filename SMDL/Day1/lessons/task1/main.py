# vectorizer demonstration

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np

my_text = np.array(["hello this is a demonstration of text vectorization"])

vectorizer = TextVectorization(max_tokens=100, output_sequence_length=10)
vectorizer.adapt(my_text)

print('=======================')

print('Original Text', my_text)

print('Vectorized sequence', vectorizer(my_text))

print('=======================')

print('Vocabulary', vectorizer.get_vocabulary())
