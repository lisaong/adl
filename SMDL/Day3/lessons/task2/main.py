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
        # to enable model saving as HDF5 format
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
    vocab = vectorizer.get_vocabulary()
    print('Vocabulary', vocab)
    print('Vocabulary size', len(vocab))

    print('========================')
    print('Vectorized texts')
    sequences = vectorizer(texts_delimited)
    print(sequences)

    sample_encoder_output = np.array([[-0.00256194], [-0.00898881], [-0.00391034]], dtype=np.float32)
    sample_encoder_hidden = np.array([[-0.00156194], [0.00020050], [-0.00095034]], dtype=np.float32)

    decoder = MyDecoder(len(vocab), embedding_dim=EMBEDDING_SIZE,
                        dec_units=BOTTLENECK_UNITS,
                        batch_size=BATCH_SIZE)

    dec_hidden = sample_encoder_hidden

    for t in range(0, sequences.shape[1]):
        # sequences[:, t] shape == (batch_size, 1)
        # tf.expand_dims(sequences[:, t], 1) shape == (batch_size, 1, 1)
        dec_input = tf.expand_dims(sequences[:, t], 1)

        print('========================')
        print(f'{t}: Decoder input: {dec_input}')

        dec_output, dec_hidden = decoder(dec_input, dec_hidden, sample_encoder_output)
        print(f'{t}: Decoder output: {dec_output}')

        decoder_output_id = tf.argmax(dec_output[0]).numpy()
        print(f'{t}: Decoder output (id): {decoder_output_id}')

        decoded_token = vectorizer.get_vocabulary()[decoder_output_id]
        print(f'{t}: Decoder output (text): {decoded_token}')

        print(f'{t}: Decoder hidden: {dec_hidden}')  # passed back into the next call to the decoder
