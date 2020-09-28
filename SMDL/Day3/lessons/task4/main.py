# Toy RNN Encoder-Decoder: Part 4 - Training
# https://www.tensorflow.org/tutorials/text/nmt_with_attention

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
from task1.main import MyEncoder
from task2.main import MyDecoder
from task3.main import loss_function

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

src_delimited = [f'{START_TOKEN} {t} {END_TOKEN}' for t in english_text]
src_vectorizer = TextVectorization()
src_vectorizer.adapt(src_delimited)
print('Source Vocabulary', src_vectorizer.get_vocabulary())

target_delimited = [f'{START_TOKEN} {t} {END_TOKEN}' for t in spanish_text]
target_vectorizer = TextVectorization()
target_vectorizer.adapt(target_delimited)
target_vocab = target_vectorizer.get_vocabulary()
print('Target Vocabulary', target_vocab)
target_sequences = target_vectorizer(target_delimited)

encoder = MyEncoder(src_vectorizer, embedding_dim=EMBEDDING_SIZE,
                    enc_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(src_delimited, sample_hidden)
print(f'Encoder output shape: (batch size, sequence length, units) {sample_output.shape}')

decoder = MyDecoder(len(target_vocab), embedding_dim=EMBEDDING_SIZE,
                    dec_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)
sample_decoder_output, sample_decoder_hidden = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                                       sample_hidden, sample_output)
print(f'Decoder output shape: (batch_size, vocab size) {sample_decoder_output.shape}')


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
        dec_input = tf.expand_dims([target_vocab.index(START_TOKEN)] * BATCH_SIZE, 1)

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


def train(epochs, batches_per_epoch, optimizer):
    # training loop
    loss_history = []
    for epoch in range(epochs):
        start_time = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        # loop batches per epoch
        for batch in range(batches_per_epoch):
            # we are repeating the same target and source sequences in this toy example
            batch_loss = train_step(src_delimited,
                                    target_sequences,
                                    enc_hidden, optimizer)
            total_loss += batch_loss
            print(f'Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy():.4f}')

        print(f'Epoch {epoch + 1} Loss {(total_loss / batches_per_epoch):.4f} Elapsed {time.time() - start_time} sec\n')
        loss_history.append(total_loss / batches_per_epoch)

    return loss_history


if __name__ == '__main__':
    loss_history = train(epochs=100, batches_per_epoch=20, optimizer=Adam())

    plt.plot(loss_history)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('learning_curve.png')
    plt.title('learning curve')
    plt.show()

    # save the model weights using the TF format for compatibility
    encoder.save_weights('encoder_weights.tf', save_format='tf')
    decoder.save_weights('decoder_weights.tf', save_format='tf')

    # save the artifacts
    artifacts = {'target_vocab': target_vectorizer.get_vocabulary(),
                 'target_seq_len': target_sequences.numpy().shape[1],
                 'start_token': START_TOKEN,
                 'end_token': END_TOKEN,
                 'embedding_size': EMBEDDING_SIZE,
                 'batch_size': BATCH_SIZE,
                 'bottleneck_units': BOTTLENECK_UNITS}
    pickle.dump(artifacts, open('model_artifacts.pkl', 'wb'))

