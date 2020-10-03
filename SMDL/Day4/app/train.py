# Day 4 - Video Classification

# Instructions:
# 1. Go through the lessons before you start
# 2. Search for the TODOs, replace _ANS_ with your answers so that the code will run
# 3. Submit your completed train.py with learning_curve.png

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

import pickle
import os
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from make_dataset import download_dataset
from td_cnn_rnn import create_model

# globals
MODEL_ARTIFACTS = dict()
MODEL_DIR = os.path.join('..', 'app', 'demo', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)


def save_artifacts(key_values: dict, dest='model_artifacts.pkl'):
    MODEL_ARTIFACTS.update(key_values)
    pickle.dump(MODEL_ARTIFACTS, open(os.path.join(MODEL_DIR, dest), 'wb'))


if __name__ == "__main__":
    sequence_len = 15
    start_offset = 10  # frames
    step = 3  # frames
    image_size = (128, 128)  # width, height

    dataset_info = {
        'Archery': {
            'filename': 'Archery.zip',
            'url': 'https://github.com/lisaong/mldds-courseware/raw/master/data/ucf101-5classes/train/Archery.zip'
        },
        'Basketball': {
            'filename': 'Basketball.zip',
            'url': 'https://github.com/lisaong/mldds-courseware/raw/master/data/ucf101-5classes/train/Basketball.zip'
        },
        'CricketBowling': {
            'filename': 'CricketBowling.zip',
            'url': 'https://github.com/lisaong/mldds-courseware/raw/master/data/ucf101-5classes/train/CricketBowling.zip'
        }
    }

    X, y = download_dataset(dataset_info, sequence_len,
                            start_offset, step, image_size)

    # label encode targets
    le = LabelEncoder()
    y_cat = to_categorical(le.fit_transform(y.ravel()))
    num_classes = len(le.classes_)

    # split to train and test
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, stratify=y)

    # TODO: create a batched tf.data.Dataset for training
    batch_size = 32
    batches_per_epoch = 5
    _ANS_

    # TODO: create model, using the create_model helper function (see td_cnn_rnn.py)
    height, width, channels = X.shape[2], X.shape[3], X.shape[4]
    _ANS_

    # TODO: train model and plot learning curve
    model_path = os.path.join(MODEL_DIR, 'td_cnn_rnn.h5')
    mc = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')

    _ANS_


    # save artifacts
    save_artifacts({'label_encoder': le,
                    'sequence_len': sequence_len,
                    'start_offset': start_offset,
                    'step': step,
                    'image_size': image_size})

    # copy dataset pre-processing code
    copyfile('make_dataset.py', os.path.join(model_path, 'make_dataset.py'))

    # metrics
    best_model = load_model(model_path)
    y_pred = best_model.predict(X_val)
    print(classification_report(y_val.argmax(axis=1), y_pred.argmax(axis=1)))
