# Transformer, in parts
# Part 1: Positional Encoding

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
import numpy as np


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim,
                                 weights=[self.init_sinusoid_table(maxlen, embed_dim)])

    def init_sinusoid_table(self, n_positions, embed_dim):
        # Initializes the sinusoid position encoding table
        # Formula:
        #   sin(t/10000^(2k/embed_dim)) when i=2k (even)
        #   cos(t/10000^(2k/embed_dim)) when i=2k+1 (odd)
        #
        # t = position being encoded
        # i = dimension index (from 0 up to embed_dim
        # k = quotient when i is divided by 2
        position_enc = np.array([
            [t / np.power(10000, 2 * (i // 2) / embed_dim) for i in range(embed_dim)]
            for t in range(n_positions)])

        # apply sine on even embedding dimensions
        #  [1:,  means start with token index 1 (token index 0 is reserved for the padding token)
        #  ,0::2] means skip every 2 starting from 0th embedding dim (i.e. even embedding dim)
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])

        # apply cosine on odd embedding dimensions
        #  [1:,  means start with token index 1 (token index 0 is reserved for the padding token)
        #  ,1::2] means skip every 2 starting from 1st embedding dim (i.e. odd embedding dim)
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])

        return position_enc

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
