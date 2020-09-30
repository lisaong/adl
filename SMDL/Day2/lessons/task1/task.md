## Tokenizer

Vectorize a text corpus, by turning each text into a sequence of integers (each integer being the index of a token in a dictionary). 

This is the predecessor for `tf.keras.layers.experimental.preprocessing.TextVectorization`. Unlike `TextVectorization`, this only performs tokenization.

Note: besides a sequence of integers, other representations include binary, word count, and tfidf.

The following settings are most commonly used:
- num_words: if set, only returns the most common (`num_words-1`) words for each sequence. "Most common" is determined globally.
- lower: whether to convert to lower-case

These are the default settings:
```
tf.keras.preprocessing.text.Tokenizer(
    num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
    split=' ', char_level=False, oov_token=None, document_count=0, **kwargs)
```
### Usage Notes
- By default, all punctuation is removed (see `filters`), turning the texts into space-separated sequences of words (see `split=' '`). These sequences are then split into lists of tokens. They will then be indexed or vectorized.
- 0 is a reserved index that won't be assigned to any word. This is usually assigned to out-of-vocab words.
- `TextVectorization` (refer to Day1) is a high-level API to provide a quick way to convert text to sequences as input to the neural network (it does everything including padding). If you need to convert back from numeric sequences to text, the `Tokenizer` is recommended because it can do the conversion in both directions (at the expense of having to do padding yourself using `tensorflow.keras.preprocessing.sequence.pad_sequences`). The next lesson will cover `pad_sequences`.

[Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)