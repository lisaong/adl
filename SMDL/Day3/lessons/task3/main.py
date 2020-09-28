# Toy RNN Encoder-Decoder: Part 3 - Loss Function
# https://www.tensorflow.org/tutorials/text/nmt_with_attention

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np

spanish_text = ['Pidan, y se les dará',
                'busquen, y encontrarán',
                'llamen, y se les abrirá.']

START_TOKEN = 'aaaaa'
END_TOKEN = 'zzzzz'

# append start and end tokens, this will indicate when translation should start & stop
target_text = [f'{START_TOKEN} {t} {END_TOKEN}' for t in spanish_text]

target_vectorizer = TextVectorization()
target_vectorizer.adapt(target_text)
target_sequences = target_vectorizer(target_text)
target_vocab_size = len(target_vectorizer.get_vocabulary())

# SparseCategorialCrossentropy
# Computes categorical cross entropy between logits (one-hot) predictions vs integer labels
# (Note - CategoricalCrossentropy: predictions and labels are both logits)
# using 'none' reduction type (i.e. don't sum across batch so that we can do reduce_mean later)
#
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
loss_equation = SparseCategoricalCrossentropy(from_logits=True, reduction='none')


# Our loss function needs to be custom so that we don't count padding
# (empty tokens) as loss. Otherwise we penalise short texts for no reason.
def loss_function(truth, pred):
    # https://www.tensorflow.org/api_docs/python/tf/math/logical_not
    # Element-wise logical NOT(truth == 0) (i.e. padding is not counted as loss)
    mask = tf.math.logical_not(tf.math.equal(truth, 0))
    loss_value = loss_equation(truth, pred)

    # https://www.tensorflow.org/api_docs/python/tf/cast
    # Convert the mask to the same datatype as loss (e.g. float32)
    mask = tf.cast(mask, dtype=loss_value.dtype)

    # Don't count padding as loss,
    # else use the loss value "as is"
    loss_value *= mask

    # Compute the average across the batch
    # (remember that tf operations are per batch)
    return tf.reduce_mean(loss_value)


# test
if __name__ == '__main__':
    decoder_output = np.array([[4.3874364e-03, -3.4227686e-03, 3.6993152e-03, 3.3376981e-03,
                                1.4048530e-03, 1.7296056e-03, -1.7817006e-03, -5.1986468e-03,
                                -2.5740519e-05, -5.5509093e-03, 2.1300791e-03, -2.7679512e-03,
                                -1.2714119e-03],
                               [5.6613060e-03, -4.4165519e-03, 4.7733923e-03, 4.3067816e-03,
                                1.8127448e-03, 2.2317877e-03, -2.2990082e-03, -6.7080469e-03,
                                -3.3214146e-05, -7.1625873e-03, 2.7485366e-03, -3.5716116e-03,
                                -1.6405598e-03],
                               [4.6547838e-03, -3.6313338e-03, 3.9247321e-03, 3.5410796e-03,
                                1.4904573e-03, 1.8349986e-03, -1.8902679e-03, -5.5154245e-03,
                                -2.7309010e-05, -5.8891526e-03, 2.2598749e-03, -2.9366156e-03,
                                -1.3488850e-03]], dtype=np.float32)

    # loss is computed per batch
    # as the truth, we will take the first token of the batch (arbitrarily)
    target_token = target_sequences[:, 0]
    print('========================')
    print(f'truth: {target_token}')
    loss = loss_function(truth=target_token, pred=decoder_output)

    print('========================')
    print(f'predictions (as logits): {decoder_output}')
    print(f'predictions (as labels): {decoder_output.argmax(axis=1)}')

    print('========================')
    print(f'loss: {loss}')
