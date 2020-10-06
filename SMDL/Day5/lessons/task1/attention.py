# Demo of Attention Layer
# A modified version of Day3/task4/training.py to add Attention to the Seq2Seq model

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Embedding, GRU, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import time


BATCH_SIZE = 16
EMBEDDING_SIZE = 4
BOTTLENECK_UNITS = 6

START_TOKEN = 'aaaaa'
END_TOKEN = 'zzzzz'


def get_data(data_path):
    df_source = pd.read_csv(f'{data_path}/kjv.txt', sep='  ', index_col=0, header=None, engine='python')
    df_target = pd.read_csv(f'{data_path}/luther.txt', sep='  ', index_col=0, header=None, engine='python')
    df_data = pd.concat([df_source, df_target], axis=1)
    df_data.columns = ['source', 'target']
    return df_data


def get_delimited_texts(s: pd.Series):
    return s.apply(lambda x: f'{START_TOKEN} {x} {END_TOKEN}').values



# Attention layer
# https://www.tensorflow.org/tutorials/text/nmt_with_attention
class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


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
class MyDecoderWithAttention(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(MyDecoderWithAttention, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.dec_units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size)

        # NEW: attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # NEW: get the context vector (i.e. weighted encoded input) by applying attention
        # query: previous decoder hidden state, value: encoded input sequence
        weighted_encoded_input, attention_weights = self.attention(hidden, enc_output)

        # pass the target through the decoder
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # NEW: concat the context vector with the target
        x = tf.concat([tf.expand_dims(weighted_encoded_input, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        # (same as Flatten)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights

    def get_config(self):
        # to enable model saving as HDF5 format
        return {'batch_size': self.batch_size,
                'dec_units': self.dec_units}


# Loss Function
loss_equation = SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(truth, pred):
    mask = tf.math.logical_not(tf.math.equal(truth, 0))
    loss_value = loss_equation(truth, pred)
    mask = tf.cast(mask, dtype=loss_value.dtype)
    loss_value *= mask
    return tf.reduce_mean(loss_value)


def get_vectorizer(texts):
    vectorizer = TextVectorization()
    vectorizer.adapt(texts)
    return vectorizer


# Vectorize
df = get_data('../../../Day3/data')
src_delimited = get_delimited_texts(df['source'])
tgt_delimited = get_delimited_texts(df['target'])

src_vectorizer = get_vectorizer(src_delimited)
src_vocab = src_vectorizer.get_vocabulary()
print('Source Vocabulary', src_vocab)
src_sequences = src_vectorizer(src_delimited)

tgt_vectorizer = get_vectorizer(tgt_delimited)
tgt_vocab = tgt_vectorizer.get_vocabulary()
print('Target Vocabulary', tgt_vocab)
tgt_sequences = tgt_vectorizer(tgt_delimited)
tgt_start_token_index = tgt_vocab.index(START_TOKEN)


# Model
encoder = MyEncoder(len(src_vocab), embedding_dim=EMBEDDING_SIZE,
                    enc_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)

decoder = MyDecoderWithAttention(len(tgt_vocab), embedding_dim=EMBEDDING_SIZE,
                                 dec_units=BOTTLENECK_UNITS,
                                 batch_size=BATCH_SIZE)


# Train
@tf.function
def train_step(source, target, enc_hidden, optimizer):
    loss = 0

    # enable automatic gradient in the block below
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(source, enc_hidden)
        dec_hidden = enc_hidden

        # set the start token
        dec_input = tf.expand_dims([tgt_vocab.index(START_TOKEN)] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, target.shape[1]):
            # Loop through the target sequence,
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(target[:, t], predictions)

            # Using teacher forcing by setting the decoder input to the next target
            # (we don't use the predictions as input, only for compute the loss)
            # (we are "teaching" the encoder-decoder with the target)

            # target[:, t] shape == (batch_size, 1)
            # tf.expand_dims(target[:, t], 1) shape == (batch_size, 1, 1)
            dec_input = tf.expand_dims(target[:, t], 1)

    # compute the gradient of the loss w.r.t. weights
    # apply gradient descent
    weights = encoder.trainable_weights + decoder.trainable_weights
    gradients = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(gradients, weights))

    batch_loss = (loss / int(target.shape[1]))
    return batch_loss


def train(train_ds, epochs, optimizer):
    # training loop
    hist = []
    for epoch in range(epochs):
        start_time = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        # loop batches per epoch
        batch = 0
        for batch, (src_batch, tgt_batch) in enumerate(train_ds):
            batch_loss = train_step(src_batch, tgt_batch, enc_hidden, optimizer)

            total_loss += batch_loss
            print(f'> {epoch + 1} ({batch + 1}) Loss {batch_loss.numpy():.4f}')

        print(f'>> {epoch + 1} Loss {(total_loss / (batch + 1)):.4f}\n')
        hist.append(total_loss / (batch + 1))

    return hist


def predict(sentence: str):
    result = ''

    sentence = f'{START_TOKEN} {sentence} {END_TOKEN}'
    inputs = src_vectorizer([sentence])
    inputs = tf.convert_to_tensor(inputs)

    hidden = [tf.zeros((1, BOTTLENECK_UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tgt_start_token_index], 0)

    max_target_sequence_length = 10
    for t in range(max_target_sequence_length):
        # get the predicted id for the next word
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[-1]).numpy()
        result += tgt_vocab[predicted_id] + ' '

        # stop when we reach the end token
        if tgt_vocab[predicted_id] == END_TOKEN:
            break

        # the predicted id and decoder hidden state is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


if __name__ == '__main__':
    BATCHES_PER_EPOCH = 5
    dataset = tf.data.Dataset.from_tensor_slices((src_sequences, tgt_sequences))
    dataset = dataset.shuffle(1024) \
        .batch(BATCH_SIZE) \
        .repeat(BATCHES_PER_EPOCH)

    history = train(dataset, epochs=5000, optimizer=Adam())

    plt.plot(history)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('learning curve')
    plt.savefig('learning_curve.png')
    plt.show()

    # get predictions
    for t in english_text:
        print(t, '=>', predict(t))
