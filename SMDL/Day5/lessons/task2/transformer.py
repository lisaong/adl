# Transformer, in parts
# Part 3: Transformer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, \
    LayerNormalization, Dropout, Input, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import LabelEncoder
import numpy as np

from multiheaded_self_attention import MultiHeadSelfAttention
from position_encoding import TokenAndPositionEmbedding


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

        # https://arxiv.org/abs/1607.06450
        # LayerNormalization normalizes across a sample (e.g. along the features axes)
        # so that the output values are zero-centered with unit variance
        # (this is to speed up training by keeping the outputs scaled down)
        # (epsilon is just a small number to avoid dividing by zero if variance is too small)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)


def create_model(sequence_len, vocab_size, embed_dim, num_heads):
    model_input = Input(shape=(sequence_len,))

    # Transformer
    x = TokenAndPositionEmbedding(sequence_len,
                                  vocab_size, embed_dim)(model_input)
    x = TransformerBlock(embed_dim, num_heads, ff_dim=2)(x)

    # MLP
    x = GlobalAveragePooling1D()(x)
    x = Dense(4, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(model_input, x)
    return model


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

    # let's train something
    my_text = np.array(['We, the citizens of Singapore',
                        'pledge ourselves as one united people',
                        'regardless of race, language or religion',
                        'to build a democratic society',
                        'based on justice and equality',
                        'so as to achieve happiness, prosperity',
                        'and progress for our nation.'])

    # create some simple labels, we'll just label every text
    # based on whether it contains a verb (action)
    my_labels = np.array(['None',
                          'Action',  # "pledge"
                          'None',
                          'Action',  # "to build"
                          'None',
                          'Action',  # "to achieve"
                          'None'])

    SEQUENCE_LEN = 5
    VOCAB_SIZE = 100
    EMBEDDING_DIM=8

    vectorizer = TextVectorization(max_tokens=VOCAB_SIZE,
                                   output_sequence_length=SEQUENCE_LEN)
    vectorizer.adapt(my_text)

    X = vectorizer(my_text).numpy()

    le = LabelEncoder()
    y = le.fit_transform(my_labels)

    model = create_model(SEQUENCE_LEN, VOCAB_SIZE,
                         EMBEDDING_DIM, num_heads=4)
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(X, y, epochs=10, batch_size=2)

    pred = model.predict(vectorizer(['achieve my goals']))
    print(pred)
