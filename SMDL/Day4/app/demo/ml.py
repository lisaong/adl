from tensorflow.keras.models import load_model
import numpy as np
import os
import pickle

# support loading from local and flask app
import imp
import sys

try:
    imp.find_module('make_dataset')
    found = True
except ImportError:
    found = False

if found:  # testing only
    from make_dataset import parse_video_frames
else: # run from flask app
    from demo.make_dataset import parse_video_frames


class TFModel:
    def __init__(self, model_dir):
        model_path = os.path.join(model_dir, 'td_cnn_rnn.h5')
        artifacts_path = os.path.join(model_dir, 'model_artifacts.pkl')

        self.model = load_model(model_path)
        self.artifacts = pickle.load(open(artifacts_path, 'rb'))
        self.label_encoder = self.artifacts['label_encoder']

    def predict(self, filename: str):
        frames, _ = parse_video_frames(filename, self.artifacts['sequence_len'],
                                       self.artifacts['start_offset'],
                                       self.artifacts['step'],
                                       self.artifacts['image_size'])

        if len(frames) >= self.artifacts['sequence_len']:
            input_sequence = np.expand_dims(frames[:self.artifacts['sequence_len']],
                                            axis=0)  # add a batch axis
            probabilities = self.model.predict(input_sequence)
            predicted_id = probabilities.argmax(axis=1)

            prediction = self.label_encoder.inverse_transform(predicted_id)[0]
            probability = probabilities[0][predicted_id[0]]
            return prediction, probability
        else:
            print('Video is too short')
            prediction = 'Video is too short'
            return prediction, None


# tests
if __name__ == '__main__':
    model = TFModel(model_dir='model')
    print(model.predict('../../lessons/task1/video.mov'))
