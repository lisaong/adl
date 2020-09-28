# Toy RNN Encoder-Decoder (Part 2: Decoder)
# https://www.tensorflow.org/tutorials/text/nmt_with_attention

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, GRU, Dense
import numpy as np

BATCH_SIZE = 3
EMBEDDING_SIZE = 4
BOTTLENECK_UNITS = 2

START_TOKEN = 'aaaaa'
END_TOKEN = 'zzzzz'


# Decoder
# In the next task we'll look at adding Attention layers. To enable an easier comparison
# we'll wrap the Decoder models into classes

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
        # tf.expand_dims(enc_output, 1) shape == (batch_size, 1, encoding_dim)
        # x shape after concatenation == (batch_size, 1, embedding_dim + encoding_dim)
        x = tf.concat([tf.expand_dims(enc_output, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state

    def get_config(self):
        return {'batch_size': self.batch_size,
                'dec_units': self.dec_units}


# test
if __name__ == '__main__':
    # target text that we will eventually want to decode to
    # see previous task for the source text
    spanish_text = ['Pidan, y se les dará',
                    'busquen, y encontrarán',
                    'llamen, y se les abrirá.']

    texts_delimited = [f'{START_TOKEN} {t} {END_TOKEN}' for t in spanish_text]
    vectorizer = TextVectorization()
    vectorizer.adapt(texts_delimited)
    print('Vocabulary', vectorizer.get_vocabulary())
    vocab_size = len(vectorizer.get_vocabulary())

    sample_encoder_output = np.array([[-0.00256194], [-0.00898881], [-0.00391034]], dtype=np.float32)
    sample_encoder_hidden = np.array([[-0.00156194], [0.00020050], [-0.00095034]], dtype=np.float32)

    decoder = MyDecoder(vocab_size, embedding_dim=2,
                        dec_units=BOTTLENECK_UNITS,
                        batch_size=BATCH_SIZE)
    sample_decoder_output, sample_decoder_hidden = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                                           sample_encoder_hidden, sample_encoder_output)
    print(f'Decoder output shape: (batch_size, vocab size) {sample_decoder_output.shape}')

    print('========================')
    print('Decoder output')
    print(sample_decoder_output)

    print('========================')
    print('Decoder hidden')
    print(sample_decoder_hidden) # this will be passed back into the next call to the decoder
