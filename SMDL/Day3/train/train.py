# Day 3 - Neural Machine Translation (Enron Emails)

# Instructions:
# 1. Go through the lessons before you start
# 2. Replace _ANS_ with your answers so that the code will run
# 3. Submit your completed train.py with learning_curve.png

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

START_TOKEN = 'aaaaaa'
END_TOKEN = 'zzzzzz'

BATCH_SIZE = 16
EMBEDDING_SIZE = 30
BOTTLENECK_UNITS = 5

df_train = pd.read_csv('../data/aeslc_train.csv', index_col=0)
df_val = pd.read_csv('../data/aeslc_val.csv', index_col=0)


def get_delimited_texts(s: pd.Series):
    return s.apply(lambda x: f'{START_TOKEN} {x} {END_TOKEN}').values


def vectorize(train_texts: list, val_texts: list):
    vectorizer = TextVectorization()
    vectorizer.adapt(train_texts)
    return vectorizer, vectorizer(train_texts), vectorizer(val_texts)


# Part 1a: Vectorize
# delimit with start and end tokens
train_src = get_delimited_texts(df_train['email_body'])
train_tgt = get_delimited_texts(df_train['subject_line'])
val_src = get_delimited_texts(df_val['email_body'])
val_tgt = get_delimited_texts(df_val['subject_line'])

# vectorize
vectorizer_src, seq_train_src, seq_val_src = vectorize(train_src, val_src)
vectorizer_tgt, seq_train_tgt, seq_val_tgt = vectorize(train_tgt, val_tgt)

vocab_src = vectorizer_src.get_vocabulary()
vocab_tgt = vectorizer_tgt.get_vocabulary()

print(f'Source Vocab size: {len(vocab_src)}')
print(f'Target Vocab size: {len(vocab_tgt)}')


# Part 1b: Encoder
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


encoder = MyEncoder(len(vocab_src), embedding_dim=EMBEDDING_SIZE,
                    enc_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)


# Part 2: Decoder
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


decoder = MyDecoder(len(vocab_tgt), embedding_dim=EMBEDDING_SIZE,
                    dec_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)


# Part 3: Loss Function
loss_equation = SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(truth, pred):
    mask = tf.math.logical_not(tf.math.equal(truth, 0))
    loss_value = loss_equation(truth, pred)
    mask = tf.cast(mask, dtype=loss_value.dtype)
    loss_value *= mask
    return tf.reduce_mean(loss_value)


# Part 4: Train

@tf.function
def train_step(source, target, enc_hidden, optimizer):
    loss = 0

    # enable automatic gradient in the block below
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(source, enc_hidden)
        dec_hidden = enc_hidden

        # set the start token
        dec_input = tf.expand_dims([vocab_tgt.index(START_TOKEN)] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, target.shape[1]):
            # Loop through the target sequence,
            # passing enc_output to the decoder
            predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
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


def train(train_ds, val_ds, epochs, optimizer):
    # training loop
    hist = {
        'loss': [],
        'val_loss': []
    }
    for epoch in range(epochs):
        start_time = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        # loop batches per epoch
        for batch, (src_batch, tgt_batch) in enumerate(train_ds):
            batch_loss = train_step(src_batch,
                                    tgt_batch,
                                    enc_hidden, optimizer)
            total_loss += batch_loss
            print(f'> {epoch + 1} ({batch + 1}) Loss {batch_loss.numpy():.4f}')

        val_loss = validate(val_ds)

        print(f'>> {epoch + 1} Loss {(total_loss / (batch+1)):.4f} '
              f'Val Loss {val_loss:.4f} Elapsed {time.time() - start_time:.4f} sec\n')
        hist['loss'].append(total_loss / (batch+1))
        hist['val_loss'].append(val_loss)

    return hist


def validate(dataset):
    total_loss = 0
    for batch, (src_batch, tgt_batch) in enumerate(dataset):
        enc_hidden = [tf.zeros((BATCH_SIZE, BOTTLENECK_UNITS))]
        enc_output, enc_hidden = encoder(src_batch, enc_hidden)

        dec_input = tf.expand_dims([vocab_tgt.index(START_TOKEN)] * BATCH_SIZE, 1)
        dec_hidden = enc_hidden

        loss = 0
        for t in range(1, tgt_batch.shape[1]):
            # Loop through the target sequence,
            # passing enc_output to the decoder
            predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(tgt_batch[:, t], predictions)

            # the predicted id and decoder hidden state is fed back into the model
            predicted_id = tf.argmax(predictions, axis=1)  # shape: (BATCH_SIZE)
            dec_input = tf.expand_dims(predicted_id, 1)  # shape: (BATCH_SIZE, 1)

        total_loss += (loss / int(tgt_batch.shape[1]))
    return total_loss/(batch+1)


# Part 5: Predict
def predict(sentence: str):
    # prepend start and end token
    sentence = f'{START_TOKEN} {sentence} {END_TOKEN}'
    inputs = vectorizer_src([sentence])
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, BOTTLENECK_UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([vectorizer_tgt.index(START_TOKEN)], 0)

    sequence_len_to_try = 10
    for t in range(sequence_len_to_try):
        # get the predicted id for the next word
        predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += vocab_tgt[predicted_id] + ' '

        # stop when we reach the end token
        if vocab_tgt[predicted_id] == END_TOKEN:
            break

        # the predicted id and decoder hidden state is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


if __name__ == '__main__':
    limit = 512  # experiment with smaller dataset sizes
    train_dataset = tf.data.Dataset.from_tensor_slices((seq_train_src[:limit], seq_train_tgt[:limit]))
    train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((seq_val_src[:limit], seq_val_tgt[:limit]))
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

    history = train(train_dataset, val_dataset, epochs=10, optimizer=tf.keras.optimizers.Adam())

    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='validation')
    plt.title('Learning Curve for Email Subject Translation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('learning_curve.png')
    plt.show()
