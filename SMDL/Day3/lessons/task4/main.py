# Toy RNN Encoder-Decoder: Part 4 - Training
# https://www.tensorflow.org/tutorials/text/nmt_with_attention

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

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
                'knock, and it will be opened to you.']

# target text
spanish_text = ['Pidan, y se les dará',
                'busquen, y encontrarán',
                'llamen, y se les abrirá.']

BATCH_SIZE = 3
EMBEDDING_SIZE = 2
BOTTLENECK_UNITS = 1

START_TOKEN = 'aaaa'
END_TOKEN = 'zzzz'

# append start and end tokens, this will indicate when translation should start & stop
src_text = [f'{START_TOKEN} {t} {END_TOKEN}' for t in english_text]

src_vectorizer = TextVectorization(output_sequence_length=10)
src_vectorizer.adapt(src_text)
src_sequences = src_vectorizer(src_text)
src_vocab_size = len(src_vectorizer.get_vocabulary())

target_text = [f'{START_TOKEN} {t} {END_TOKEN}' for t in spanish_text]

target_vectorizer = TextVectorization(output_sequence_length=10)
target_vectorizer.adapt(target_text)
target_sequences = target_vectorizer(target_text)
target_vocab_size = len(target_vectorizer.get_vocabulary())
target_start_token_index = target_vectorizer.get_vocabulary().index(START_TOKEN)

encoder = MyEncoder(src_vocab_size, embedding_dim=EMBEDDING_SIZE,
                    enc_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(src_sequences, sample_hidden)
print(f'Encoder output shape: (batch size, sequence length, units) {sample_output.shape}')

decoder = MyDecoder(target_vocab_size, embedding_dim=2,
                    dec_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)
sample_decoder_output, sample_decoder_hidden = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                                       sample_hidden, sample_output)
print(f'Decoder output shape: (batch_size, vocab size) {sample_decoder_output.shape}')

# https://www.tensorflow.org/api_docs/python/tf/function
# Compile the function into a Tensorflow graph
optimizer = Adam()


@tf.function
def train_step(source, target, enc_hidden):
    loss = 0

    # enable automatic gradient in the block below
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(source, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([target_start_token_index] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, target.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(target[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, t], 1)

    batch_loss = (loss / int(target.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


if __name__ == '__main__':
    pass
