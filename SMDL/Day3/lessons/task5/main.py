# Toy RNN Encoder-Decoder: Part 5 - Predict

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pickle

# import classes and losses from previous tasks
# (everything else we will re-declare below for clarity)
import sys

sys.path.append('..')
from task1.main import MyEncoder
from task2.main import MyDecoder


# source text
english_text = ['Ask, and it will be given to you',
                'seek, and you will find',
                'knock, and it will be opened to you.',
                'For everyone who asks receives',
                'and he who seeks finds',
                'and to him who knocks it will be opened']

model_artifacts = pickle.load(open('../task4/model_artifacts.pkl', 'rb'))

BATCH_SIZE = model_artifacts['batch_size']
EMBEDDING_SIZE = model_artifacts['embedding_size']
BOTTLENECK_UNITS = model_artifacts['bottleneck_units']

START_TOKEN = model_artifacts['start_token']
END_TOKEN = model_artifacts['end_token']


def create_vectorizer(vocab, seq_len):
    vectorizer = TextVectorization(max_tokens=None, output_sequence_length=seq_len)
    vectorizer.set_vocabulary(vocab)
    return vectorizer


src_vectorizer = create_vectorizer(model_artifacts['src_vocab'],
                                   model_artifacts['src_seq_len'])

# hack - somehow the vectorizer cannot be properly instantiated
# if used for vectorization
src_vectorizer.adapt([f'{START_TOKEN} {t} {END_TOKEN}' for t in english_text])
src_vocab_size = len(src_vectorizer.get_vocabulary())
assert src_vectorizer.get_vocabulary() == model_artifacts['src_vocab'], \
    'vocabs must be equal'
# end hack

target_vectorizer = create_vectorizer(model_artifacts['target_vocab'],
                                      model_artifacts['target_seq_len'])
target_vocab = target_vectorizer.get_vocabulary()
target_vocab_size = len(target_vocab)
target_start_token_index = target_vocab.index(START_TOKEN)

encoder = MyEncoder(src_vocab_size, embedding_dim=EMBEDDING_SIZE,
                    enc_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)
# call the model first to create the variables
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(tf.zeros((BATCH_SIZE, model_artifacts['src_seq_len'])),
                                       sample_hidden)
encoder.load_weights('../task4/encoder_weights.h5')
print(encoder.summary())

decoder = MyDecoder(target_vocab_size, embedding_dim=EMBEDDING_SIZE,
                    dec_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)
# call the model first to create the variables
_ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
            sample_hidden, sample_output)
decoder.load_weights('../task4/decoder_weights.h5')
print(decoder.summary())


def predict(sentence: str):
    result = ''

    sentence = f'{START_TOKEN} {sentence} {END_TOKEN}'
    inputs = src_vectorizer([sentence])
    inputs = tf.convert_to_tensor(inputs)

    hidden = [tf.zeros((1, BOTTLENECK_UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_start_token_index], 0)

    sequence_len_to_try = 10
    for t in range(sequence_len_to_try):
        # get the predicted id for the next word
        predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += target_vectorizer.get_vocabulary()[predicted_id] + ' '

        # stop when we reach the end token
        if target_vectorizer.get_vocabulary()[predicted_id] == END_TOKEN:
            break

        # the predicted id and decoder hidden state is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


if __name__ == "__main__":
    for t in english_text:
        print(t, '=>', predict(t))
