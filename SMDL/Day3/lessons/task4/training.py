# Toy RNN Encoder-Decoder: Part 4 - Training

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.optimizers import Adam
import time
import matplotlib.pyplot as plt
import pickle

# import classes and losses from previous tasks
# (everything else we will re-declare below for clarity)
import sys

sys.path.append('..')
from task1.encoder import MyEncoder
from task2.decoder import MyDecoder
from task3.loss_function import loss_function

# source text
english_text = ['Ask, and it will be given to you',
                'seek, and you will find',
                'knock, and it will be opened to you.',
                'For everyone who asks receives',
                'and he who seeks finds',
                'and to him who knocks it will be opened']

# target text
spanish_text = ['Pidan, y se les dará',
                'busquen, y encontrarán',
                'llamen, y se les abrirá.',
                'Porque todo el que pide, recibe',
                'el que busca, encuentra',
                'y al que llama, se le abre']

BATCH_SIZE = len(english_text)
EMBEDDING_SIZE = 4
BOTTLENECK_UNITS = 6

START_TOKEN = 'aaaa'
END_TOKEN = 'zzzz'


def get_vectorizer(texts):
    vectorizer = TextVectorization()
    vectorizer.adapt(texts)
    return vectorizer


src_delimited = [f'{START_TOKEN} {t} {END_TOKEN}' for t in english_text]
src_vectorizer = get_vectorizer(src_delimited)
src_vocab = src_vectorizer.get_vocabulary()
print('Source Vocabulary', src_vocab)
src_sequences = src_vectorizer(src_delimited)

tgt_delimited = [f'{START_TOKEN} {t} {END_TOKEN}' for t in spanish_text]
tgt_vectorizer = get_vectorizer(tgt_delimited)
tgt_vocab = tgt_vectorizer.get_vocabulary()
print('Target Vocabulary', tgt_vocab)
tgt_sequences = tgt_vectorizer(tgt_delimited)

encoder = MyEncoder(len(src_vocab), embedding_dim=EMBEDDING_SIZE,
                    enc_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)

decoder = MyDecoder(len(tgt_vocab), embedding_dim=EMBEDDING_SIZE,
                    dec_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)


# https://www.tensorflow.org/api_docs/python/tf/function
# Compile the function into a Tensorflow graph
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


def train(train_dataset, epochs, optimizer):
    # training loop

    loss_history = []

    for epoch in range(epochs):
        start_time = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        # https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
        for batch, (src_batch, tgt_batch) in enumerate(train_dataset):
            batch_loss = train_step(src_batch, tgt_batch, enc_hidden, optimizer)
            total_loss += batch_loss
            print(f'> {epoch + 1} ({batch + 1}) Loss {batch_loss.numpy():.4f}')

        print(f'>> {epoch + 1} Loss {(total_loss / (batch + 1)):.4f} Elapsed {time.time() - start_time:.4f} sec')
        loss_history.append(total_loss / (batch + 1))

    return loss_history


if __name__ == '__main__':
    # Create a batched dataset from our sequences
    BATCHES_PER_EPOCH = 5
    dataset = tf.data.Dataset.from_tensor_slices((src_sequences, tgt_sequences))
    dataset = dataset.shuffle(1024)\
        .batch(BATCH_SIZE)\
        .repeat(BATCHES_PER_EPOCH)

    history = train(dataset, epochs=500, optimizer=Adam())

    plt.plot(history)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('learning curve')
    plt.savefig('learning_curve.png')
    plt.show()

    # save the model weights
    encoder.save_weights('encoder_weights.h5')
    decoder.save_weights('decoder_weights.h5')

    # save the artifacts
    artifacts = {'src_vocab': src_vocab,
                 'src_seq_len': src_sequences.numpy().shape[1],
                 'target_vocab': tgt_vocab,
                 'target_seq_len': tgt_sequences.numpy().shape[1],
                 'start_token': START_TOKEN,
                 'end_token': END_TOKEN,
                 'embedding_size': EMBEDDING_SIZE,
                 'batch_size': BATCH_SIZE,
                 'bottleneck_units': BOTTLENECK_UNITS}
    pickle.dump(artifacts, open('model_artifacts.pkl', 'wb'))
