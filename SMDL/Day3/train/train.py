# Day 3 - Neural Machine Translation

# Instructions:
# 1. Go through the lessons before you start
# 2. Search for the TODOs, replace _ANS_ with your answers so that the code will run
# 3. Submit your completed train.py with learning_curve.png

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import time
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from shutil import copyfile

from make_dataset import get_data
from seq2seq import MyEncoder, MyDecoder

START_TOKEN = 'aaaaaa'
END_TOKEN = 'zzzzzz'

BATCH_SIZE = 16
EMBEDDING_SIZE = 10
BOTTLENECK_UNITS = 8


def get_delimited_texts(s: pd.Series):
    return s.apply(lambda x: f'{START_TOKEN} {x} {END_TOKEN}').values


def vectorize(train_texts: list, val_texts: list):
    vectorizer = TextVectorization()
    vectorizer.adapt(train_texts)
    return vectorizer, vectorizer(train_texts), vectorizer(val_texts)


# Part 1a: Vectorize
df_train, df_val = get_data()

train_src = get_delimited_texts(df_train['english'])
train_tgt = get_delimited_texts(df_train['german'])
val_src = get_delimited_texts(df_val['english'])
val_tgt = get_delimited_texts(df_val['german'])

# vectorize
vectorizer_src, seq_train_src, seq_val_src = vectorize(train_src, val_src)
vectorizer_tgt, seq_train_tgt, seq_val_tgt = vectorize(train_tgt, val_tgt)

vocab_src = vectorizer_src.get_vocabulary()
vocab_tgt = vectorizer_tgt.get_vocabulary()

print(f'Source Vocab size: {len(vocab_src)}')
print(f'Target Vocab size: {len(vocab_tgt)}')


# Part 1b: Encoder
# TODO: Replace _ANS_ with your solution to create the encoder
# encoder = _ANS_
encoder = MyEncoder(len(vocab_src), embedding_dim=EMBEDDING_SIZE,
                    enc_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)


# Part 2: Decoder
# TODO: Replace _ANS_ with your solution to create the decoder
# decoder = _ANS_
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
            # TODO: perform training using the train_step function
            # Replace _ANS_ with your solution
            # batch_loss = _ANS_
            batch_loss = train_step(src_batch, tgt_batch, enc_hidden, optimizer)

            total_loss += batch_loss
            print(f'> {epoch + 1} ({batch + 1}) Loss {batch_loss.numpy():.4f}')

        val_loss = validate(val_ds)

        print(f'>> {epoch + 1} Loss {(total_loss / (batch + 1)):.4f} '
              f'Val Loss {val_loss:.4f} Elapsed {time.time() - start_time:.4f} sec\n')
        hist['loss'].append(total_loss / (batch + 1))
        hist['val_loss'].append(val_loss)

    return hist


def validate(dataset):
    total_loss = 0
    batch = 0
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
    return total_loss / (batch + 1)


if __name__ == '__main__':
    EPOCHS = 30
    MODEL_PATH = '../app/demo/model'

    train_dataset = tf.data.Dataset.from_tensor_slices((seq_train_src, seq_train_tgt))
    train_dataset = train_dataset.shuffle(10*BATCH_SIZE) \
        .repeat(EPOCHS) \
        .batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_tensor_slices((seq_val_src, seq_val_tgt))
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)  # make complete batches

    history = train(train_dataset, val_dataset, epochs=EPOCHS,
                    optimizer=tf.keras.optimizers.Adam())

    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='validation')
    plt.title('Learning Curve for Bible Translation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('learning_curve.png')
    plt.show()

    # Save the model weights and architecture
    encoder.save_weights(f'{MODEL_PATH}/encoder_weights_e{EPOCHS}.h5')
    decoder.save_weights(f'{MODEL_PATH}/decoder_weights_e{EPOCHS}.h5')
    copyfile('seq2seq.py', f'{MODEL_PATH}/seq2seq.py')

    # Save the train source and target for creating the vectorizers
    # (Note: We'll recreate the vectorizers on deployment due to their limitations)
    np.save(f'{MODEL_PATH}/train_src.npy', train_src)
    np.save(f'{MODEL_PATH}/train_tgt.npy', train_tgt)

    # save the artifacts
    artifacts = {'start_token': START_TOKEN,
                 'end_token': END_TOKEN,
                 'embedding_size': EMBEDDING_SIZE,
                 'batch_size': BATCH_SIZE,
                 'bottleneck_units': BOTTLENECK_UNITS,
                 'epochs': EPOCHS}
    pickle.dump(artifacts, open(f'{MODEL_PATH}/model_artifacts.pkl', 'wb'))
