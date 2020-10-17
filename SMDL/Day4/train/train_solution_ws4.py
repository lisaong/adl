# Day 4 - Video Classification

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
    # create batched training dataset
    batch_size = 32
    batches_per_epoch = 10
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)). \
        batch(batch_size).repeat(batches_per_epoch)

    # TODO: create model, using the create_model helper function (see td_cnn_rnn.py)
    # create model
    height, width, channels = X.shape[2], X.shape[3], X.shape[4]
    model = create_model(height, width, channels, 8, num_classes)

    # TODO: train model and plot learning curve
    # train model
    model_path = os.path.join(MODEL_DIR, 'td_cnn_rnn.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    mc = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    history = model.fit(train_ds, epochs=3,
                        validation_data=(X_val, y_val),
                        callbacks=[mc])

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Learning Curve')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('learning_curve.png')
    plt.show()

    save_artifacts({'label_encoder': le,
                    'sequence_len': sequence_len,
                    'start_offset': start_offset,
                    'step': step,
                    'image_size': image_size})

    # copy dataset pre-processing code
    copyfile('make_dataset.py', os.path.join(MODEL_DIR, 'make_dataset.py'))

    # metrics
    best_model = load_model(model_path)
    y_pred = best_model.predict(X_val)
    print(classification_report(y_val.argmax(axis=1), y_pred.argmax(axis=1)))
