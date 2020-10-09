# Transformer, in parts

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
import numpy as np


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


if __name__ == "__main__":
    # create a TokenAndPositionEmbedding layer, and play with it
    emb = TokenAndPositionEmbedding(maxlen=20, vocab_size=10, embed_dim=2)

    # pass some test sequence into it
    test = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 4], dtype='float32')

    # use the debugger to step into this line, see how it works
    output = emb(test)

    print('Input:', test)
    print('Output:', output)
