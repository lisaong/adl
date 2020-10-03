import tensorflow as tf
from sklearn.model_selection import train_test_split

import numpy as np

BATCH_SIZE = 16
BATCHES_PER_EPOCH = 5

if __name__ == "__main__":
    # load dataset saved from previous task
    X = np.load('../task2/X.npy', allow_pickle=True)
    y = np.load('../task2/y.npy', allow_pickle=True)

    # split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y)

    # create batched training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)). \
        batch(BATCH_SIZE).repeat(BATCHES_PER_EPOCH)

    # create model
