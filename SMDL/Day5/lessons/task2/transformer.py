# Transformer, in parts
# Part 3: Transformer

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
import numpy as np

from multiheaded_self_attention import MultiHeadSelfAttention


class TransformerBlock(Layer):
    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.dense1(out1)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)

        self.dense1 = Dense(ff_dim, activation='relu')
        self.dense2 = Dense(embed_dim)

        # https://www.tensorflow.org/addons/tutorials/layers_normalizations
        # LayerNormalization normalizes across a sample (e.g. along the features axes)
        # so that the output values are zero-centered with unit variance
        # (this is to speed up training by keeping the outputs scaled down)
        # (epsilon is just a small number to avoid dividing by zero if variance is too small)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)


if __name__ == "__main__":
    # create a TransformerBlock, and play with it
    tb = TransformerBlock(embed_dim=4, num_heads=2, ff_dim=6)

    # pass some test input into it
    # shape = [batch_size, seq_len, embedding_dim]
    test = np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]], dtype='float32')
    print(test.shape)

    # use the debugger to step into this line, see how it works
    output = tb(test)

    print('Input:', test)
    print('Output:', output)
