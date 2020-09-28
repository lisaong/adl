# Toy RNN Encoder-Decoder (Part 1: Encoder)
# https://www.tensorflow.org/tutorials/text/nmt_with_attention

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, GRU


BATCH_SIZE = 3
EMBEDDING_SIZE = 4
BOTTLENECK_UNITS = 2

START_TOKEN = 'aaaaa'
END_TOKEN = 'zzzzz'


# Encoder
# In the final task we'll look at adding Attention layers. To enable an easier comparison
# we'll wrap the Encoder models into classes

class MyEncoder(Model):
    def __init__(self, vectorizer, embedding_dim, enc_units, batch_size):
        super(MyEncoder, self).__init__()
        self.vectorizer = vectorizer
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = Embedding(len(vectorizer.get_vocabulary()), embedding_dim)
        self.gru = GRU(self.enc_units, return_state=True)

    def call(self, x, hidden):
        x = self.vectorizer(x)
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

    def get_config(self):
        return {'batch_size': self.batch_size,
                'enc_units': self.enc_units}


# test
if __name__ == '__main__':  # so that we can import this file without running the test code

    # source text that we want to encode
    # see next task for the target text
    english_text = ['Ask, and it will be given to you',
                    'seek, and you will find',
                    'knock, and it will be opened to you.']
    texts_delimited = [f'{START_TOKEN} {t} {END_TOKEN}' for t in english_text]
    vectorizer = TextVectorization()
    vectorizer.adapt(texts_delimited)
    print('Vocabulary', vectorizer.get_vocabulary())

    encoder = MyEncoder(vectorizer, embedding_dim=EMBEDDING_SIZE,
                        enc_units=BOTTLENECK_UNITS,
                        batch_size=BATCH_SIZE)
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(english_text, sample_hidden)
    print(f'Encoder output shape: (batch size, sequence length, units) {sample_output.shape}')

    print('========================')
    print('Encoder output')
    print(sample_output)

    print('========================')
    print('Encoder hidden')
    print(sample_hidden)  # this will be passed to the next call to the encoder