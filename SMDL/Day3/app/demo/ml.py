import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# support loading from local and flask app
import imp
import sys

sys.path.append('model')
try:
    imp.find_module('seq2seq')
    found = True
except ImportError:
    found = False

if found: # testing only
    from seq2seq import MyEncoder, MyDecoder
else: # run from flask app
    from demo.model.seq2seq import MyEncoder, MyDecoder


class TFModel:
    def __init__(self, model_dir):
        # load the artifacts
        self.artifacts = pickle.load(open(os.path.join(model_dir, 'model_artifacts.pkl'), 'rb'))

        # create the vectorizers
        train_src = np.load(os.path.join(model_dir, 'train_src.npy'), allow_pickle=True)
        train_tgt = np.load(os.path.join(model_dir, 'train_tgt.npy'), allow_pickle=True)

        vectorizer_src = TextVectorization()
        vectorizer_src.adapt(train_src)
        train_seq = vectorizer_src(train_src)
        self.vectorizer_src = vectorizer_src

        vectorizer_tgt = TextVectorization()
        vectorizer_tgt.adapt(train_tgt)
        self.vectorizer_tgt = vectorizer_tgt

        # load models
        vocab_src = vectorizer_src.get_vocabulary()
        self.encoder = MyEncoder(len(vocab_src), embedding_dim=self.artifacts['embedding_size'],
                                 enc_units=self.artifacts['bottleneck_units'],
                                 batch_size=self.artifacts['batch_size'])

        # call the model first to create the variables
        sample_hidden = self.encoder.initialize_hidden_state()
        sample_output, sample_hidden = self.encoder(tf.zeros((self.artifacts['batch_size'],
                                                              train_seq.numpy().shape[1])),
                                                    sample_hidden)
        self.encoder.load_weights(os.path.join(model_dir,
                                               f'encoder_weights_e{self.artifacts["epochs"]}.h5'))
        print(self.encoder.summary())

        vocab_tgt = vectorizer_tgt.get_vocabulary()
        self.decoder = MyDecoder(len(vocab_tgt), embedding_dim=self.artifacts['embedding_size'],
                                 dec_units=self.artifacts['bottleneck_units'],
                                 batch_size=self.artifacts['batch_size'])

        # call the model first to create the variables
        _ = self.decoder(tf.random.uniform((self.artifacts['batch_size'], 1)),
                         sample_hidden, sample_output)
        self.decoder.load_weights(os.path.join(model_dir,
                                               f'decoder_weights_e{self.artifacts["epochs"]}.h5'))
        print(self.decoder.summary())

    def predict(self, sentence: str, prepend_tokens=True):
        result = ''

        if prepend_tokens:
            # prepend start and end token
            sentence = f'{self.artifacts["start_token"]} {sentence} {self.artifacts["end_token"]}'
        inputs = self.vectorizer_src([sentence])
        inputs = tf.convert_to_tensor(inputs)
        vocab_tgt = self.vectorizer_tgt.get_vocabulary()

        hidden = [tf.zeros((1, self.artifacts['bottleneck_units']))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([vocab_tgt.index(self.artifacts['start_token'])], 0)

        sequence_len_to_try = 10
        for t in range(sequence_len_to_try):
            # get the predicted id for the next word
            predictions, dec_hidden = self.decoder(dec_input, dec_hidden, enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy()

            # stop when we reach the end token
            if vocab_tgt[predicted_id] == self.artifacts['end_token']:
                break

            result += vocab_tgt[predicted_id] + ' '

            # the predicted id and decoder hidden state is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result


# tests
if __name__ == '__main__':
    model = TFModel(model_dir='model')

    train_src = np.load(os.path.join('model', 'train_src.npy'),
                        allow_pickle=True)

    for t in train_src[-10:]:
        print(t, '\n\t\t=>', model.predict(t, prepend_tokens=False))
