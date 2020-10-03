# Demo of time-distributed CNN for binary classification

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, \
    TimeDistributed, GRU, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

BATCH_SIZE = 32
BATCHES_PER_EPOCH = 5


def create_model(h, w, c):
    # load pre-trained model, set to non-trainable
    base_model = MobileNetV2(input_shape=(h, w, c), include_top=False)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)  # compute batch average
    cnn_model = Model(base_model.input, x)

    # sequential model that wraps the cnn submodel
    # input shape == (sequence, height, width, channels)
    model_input = Input(shape=(None, height, width, channels))
    x = TimeDistributed(cnn_model)(model_input)

    # pass the output to our small RNN
    x = GRU(8, activation='tanh')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)  # binary classifier

    rnn_model = Model(model_input, x)
    rnn_model.summary()
    return rnn_model


if __name__ == "__main__":
    # load dataset saved from previous task
    X = np.load('../task2/X.npy', allow_pickle=True)
    y = np.load('../task2/y.npy', allow_pickle=True)

    # encode y from labels to binary
    le = LabelEncoder()
    y_enc = le.fit_transform(y.ravel())  # convert y from 2D to 1D for encoding

    # split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, stratify=y_enc)

    height, width, channels = X_train.shape[2], X_train.shape[3], X_train.shape[4]

    model = create_model(height, width, channels)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    # create batched training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)). \
        batch(BATCH_SIZE).repeat(BATCHES_PER_EPOCH)

    mc = ModelCheckpoint('td_cnn_rnn.h5', save_best_only=True, monitor='val_loss')

    model.fit(train_ds, epochs=3,
              validation_data=(X_val, y_val),
              callbacks=[mc])

    # metrics
    best_model = load_model('td_cnn_rnn.h5')
    preds = best_model.predict(X_val) >= 0.5
    print(classification_report(y_val, preds))
