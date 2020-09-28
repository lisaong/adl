import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pickle

# import classes and losses from previous tasks
# (everything else we will re-declare below for clarity)
import sys

sys.path.append('..')
from task1.main import MyEncoder
from task2.main import MyDecoder


model_artifacts = pickle.load(open('../task4/model_artifacts.pkl', 'rb'))

BATCH_SIZE = model_artifacts['batch_size']
EMBEDDING_SIZE = model_artifacts['embedding_size']
BOTTLENECK_UNITS = model_artifacts['bottleneck_units']

START_TOKEN = model_artifacts['start_token']
END_TOKEN = model_artifacts['end_token']

# source text
english_text = ['Ask, and it will be given to you',
                'seek, and you will find',
                'knock, and it will be opened to you.',
                'For everyone who asks receives',
                'and he who seeks finds',
                'and to him who knocks it will be opened']


def create_vectorizer(vocab, seq_len):
    vectorizer = TextVectorization()
    vectorizer.set_vocabulary(vocab)
    return vectorizer


src_vectorizer = create_vectorizer(model_artifacts['src_vocab'],
                                   model_artifacts['src_seq_len'])
src_vocab_size = len(src_vectorizer.get_vocabulary())

target_vectorizer = create_vectorizer(model_artifacts['target_vocab'],
                                      model_artifacts['target_seq_len'])
target_vocab_size = len(target_vectorizer.get_vocabulary())
target_start_token_index = target_vectorizer.get_vocabulary().index(START_TOKEN)

encoder = MyEncoder(src_vocab_size, embedding_dim=EMBEDDING_SIZE,
                    enc_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()

# call the model first to create the variables
sample_output, sample_hidden = encoder(tf.zeros((BATCH_SIZE,
                                                 model_artifacts['src_seq_len'])),
                                       sample_hidden)
encoder.load_weights('../task4/encoder_weights.h5')

decoder = MyDecoder(target_vocab_size, embedding_dim=EMBEDDING_SIZE,
                    dec_units=BOTTLENECK_UNITS,
                    batch_size=BATCH_SIZE)
# call the model first to create the variables
_ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
            sample_hidden, sample_output)
decoder.load_weights('../task4/decoder_weights.h5')


def predict(sentence: str):
    # prepend start and end token
    sentence = f'{START_TOKEN} {sentence} {END_TOKEN}'
    inputs = src_vectorizer([sentence])
    inputs = tf.convert_to_tensor(inputs)

    result = ''

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
