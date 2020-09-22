import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import os
import pickle


class TFModel:
    def __init__(self):
        model_dir = os.path.join(os.getcwd(), 'demo', 'model')
        model_path = os.path.join(model_dir, 'lstm.h5')
        artifacts_path = os.path.join(model_dir, 'model_artifacts.pkl')

        self.model = load_model(model_path)

        self.artifacts = pickle.load(open(artifacts_path, 'rb'))
        self.label_encoder = self.artifacts['label_encoder']

        # init vectorizer
        vectorizer_config = self.artifacts['vectorizer_config']
        vectorizer_vocab = self.artifacts['vectorizer_vocab']

        self.vec = TextVectorization(max_tokens=vectorizer_config['max_tokens'],
                                     output_sequence_length=vectorizer_config['output_sequence_length'])
        self.vec.set_vocabulary(vectorizer_vocab)

    def predict(self, data: str):
        x = self.vec(tf.constant([data]))
        y = self.model.predict(x).argmax(axis=1)
        return self.label_encoder.inverse_transform(y)[0]

