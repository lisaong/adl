# Demo of Attention Layer

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, GRU, Dense, Attention


BATCH_SIZE = 3
EMBEDDING_SIZE = 4
BOTTLENECK_UNITS = 2

START_TOKEN = 'aaaaa'
END_TOKEN = 'zzzzz'


# Encoder from Day3
class MyEncoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(MyEncoder, self).__init__()
        self.vocab_size = vocab_size
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

    def get_config(self):
        # to enable model saving as HDF5 format
        return {'batch_size': self.batch_size,
                'enc_units': self.enc_units}

# Decoder from Day3, modified with Attention


# Decoder
class MyDecoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(MyDecoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.dec_units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size)

        # NEW: attention
        self.attention = Attention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # pass the target through the decoder
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # NEW: get the context vector (i.e. weighted encoded input) by applying attention
        # query: target, value: enc_output, key: enc_output
        weighted_encoded_input = self.attention([x, enc_output])

        # NEW: concat the context vector with the intermediate data
        x = tf.concat([tf.expand_dims(weighted_encoded_input, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        # (same as Flatten)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state

    def get_config(self):
        # to enable model saving as HDF5 format
        return {'batch_size': self.batch_size,
                'dec_units': self.dec_units}


