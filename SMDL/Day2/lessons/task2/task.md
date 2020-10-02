## pad_sequences

Pads sequences to the same length. This is often done after tokenization (see previous task on `tf.keras.preprocessing.text.Tokenizer`) to ensure that all sequences are of the same length before passing it to an Embedding layer.

Note: if using `tf.keras.layers.experimental.preprocessing.TextVectorization`, rightmost padding is performed, so no separate padding step is needed.

The following settings are most commonly used:
- maxlen: if set, limits the maximum length of all sequences. If `None`, sequences will be padded to the length of the longest individual sequence.
- padding: 'pre' or 'post' - whether to pad before or after each sequence, for sequences shorter than `maxlen`.
- truncating: pre' or 'post' - whether to remove values at the beginning or at the end of the sequences, for sequences longer than `maxlen`.

These are the default settings:
```
tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre',
    value=0.0)
```

[Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences)
