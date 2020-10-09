# Transformer, in parts
# Part 2: Multi-headed Self-attention

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import numpy as np

# You don't have to implement this from scratch, you can also use
# keras.layers.MultiHeadAttention from Tensorflow Add-ons
# https://www.tensorflow.org/addons
# https://www.tensorflow.org/addons/api_docs/python/tfa/layers/MultiHeadAttention


class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )

        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.output_dense = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        out = tf.matmul(weights, value)
        return out, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]

        # Query
        # Dense layer to project and split embed_dim to num_heads*projection_dim
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)

        # Key
        # Dense layer to project and split embed_dim to num_heads*projection_dim
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)

        # Value
        # Dense layer to project and split embed_dim to num_heads*projection_dim
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)

        # Attention
        attention, _ = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)

        # Combine back to num_heads*projection_dim
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)

        # Apply a final Dense layer
        out = self.output_dense(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return out


if __name__ == "__main__":
    # create a MultiHeadSelfAttention layer, and play with it
    mhsa = MultiHeadSelfAttention(embed_dim=4, num_heads=2)

    # pass some test input into it
    # shape = [batch_size, seq_len, embedding_dim]
    test = np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]], dtype='float32')
    print(test.shape)

    # use the debugger to step into this line, see how it works
    output = mhsa(test)

    print('Input:', test)
    print('Output:', output)
