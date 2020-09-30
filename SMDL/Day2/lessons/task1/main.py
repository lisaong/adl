# Tokenizer demonstration

from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import json

my_text = np.array(["Hello! this is a demonstration of Tokenizer.",
                    "Demonstration, is it really a?"])  # yoda speak

tokenizer = Tokenizer(num_words=5, lower=True)
tokenizer.fit_on_texts(my_text)

sequences = tokenizer.texts_to_sequences(my_text)

print('=======================')

print('Original Text', my_text)

print('Tokenized sequences', sequences)

print('=======================')

print('Vocabulary', tokenizer.word_index)
print('Word counts', tokenizer.word_counts)

print('=======================')

print('Conversion back to texts (sequences only contain the top num_words-1):')

print(tokenizer.sequences_to_texts(sequences))

print('=======================')

print('Configuration as JSON', json.loads(tokenizer.to_json()))
