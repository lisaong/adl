# Tokenizer demonstration

from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
import numpy as np

my_text = np.array(["Hello! this is a demonstration of Tokenizer.",
                    "Demonstration, is it really a?"])  # yoda speak

tokenizer = Tokenizer(num_words=5, lower=True)

# one-time initialization to setup the vocabulary
tokenizer.fit_on_texts(my_text)

# perform the tokenization for any text
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

config = tokenizer.to_json()
print('Configuration as JSON string', config)

print('=======================')

print('Load configuration into another Tokenizer')

another_tokenizer = tokenizer_from_json(config)

print(another_tokenizer.sequences_to_texts(sequences))
