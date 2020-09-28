# Toy RNN Encoder-Decoder
# https://www.tensorflow.org/tutorials/text/nmt_with_attention

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

english_text = ['Ask, and it will be given to you', 'seek, and you will find', 'knock, and it will be opened to you.']
spanish_text = ['Pidan, y se les dará', 'busquen, y encontrarán', 'llamen, y se les abrirá.']
BATCH_SIZE=3

START_TOKEN = '<start>'
END_TOKEN = '<end>'

# append start and end tokens, this will indicate when translation should start & stop
src_text = [f'{START_TOKEN} {t} {END_TOKEN}' for t in english_text]
target_text = [f'{START_TOKEN} {t} {END_TOKEN}' for t in spanish_text]

src_vectorizer = TextVectorization(output_sequence_length=10)
src_vectorizer.adapt(src_text)
src_sequences = src_vectorizer(src_text)
src_vocab_size = len(src_vectorizer.get_vocabulary())

target_vectorizer = TextVectorization(output_sequence_length=10)
target_vectorizer.adapt(target_text)
target_sequences = target_vectorizer(target_text)
target_vocab_size = len(target_vectorizer.get_vocabulary())

# Encoder
# In the next task we'll look at adding Attention layers. To enable an easier comparison
# we'll wrap the Encoder and Decoder models into classes

class MyEncoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(MyEncoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.enc_units, return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class MyDecoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(MyDecoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.dec_units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # enc_output shape == (batch_size, encoding_dim)
        # tf.expand_dims(enc_output, 1) == (batch_size, 1, encoding_dim)
        # x shape after concatenation == (batch_size, 1, embedding_dim + encoding_dim)
        x = tf.concat([tf.expand_dims(enc_output, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state


# test
BOTTLENECK_UNITS = 5
encoder = MyEncoder(src_vocab_size, embedding_dim=2, enc_units=BOTTLENECK_UNITS, batch_size=BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(src_sequences, sample_hidden)
print(f'Encoder output shape: (batch size, sequence length, units) {sample_output.shape}')

decoder = MyDecoder(target_vocab_size, embedding_dim=2, dec_units=BOTTLENECK_UNITS, batch_size=BATCH_SIZE)
sample_decoder_output, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                   sample_hidden, sample_output)
print(f'Decoder output shape: (batch_size, vocab size) {sample_decoder_output.shape}')

optimizer = tf.keras.optimizers.Adam()

# Custom losses for encoder-decoder for translation
#
# SparseCategorialCrossentropy: labels will be provided as integers
# (compare to CategoricalCrossentropy: labels provided as one-hot)
# using 'none' reduction type (i.e. don't sum across batch so that we can do reduce_mean later)
# from_logits means we are not using probability
loss_equation = SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(truth, pred):
    # if truth != 0
    mask = tf.math.logical_not(tf.math.equal(truth, 0))
    loss_value = loss_equation(truth, pred)

    # convert mask to the same datatype as loss
    mask = tf.cast(mask, dtype=loss_value.dtype)

    # multiply-accumulate the losses
    loss_value *= mask

    # compute the average across the batch
    return tf.reduce_mean(loss_value)

