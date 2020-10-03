import tensorflow as tf
import os
import sys
import glob
import numpy as np

sys.path.append('..')

from task1.frame_extractor import extract_frames, plot_images

SEQUENCE_LEN = 20
START_OFFSET = 10  # frames
STEP = 3  # frames
IMAGE_SIZE = (128, 128)  # width, height


def parse_video_frames(filename):
    label_ = os.path.basename(os.path.split(filename)[0])  # parent folder
    frames_, _ = extract_frames(filename, SEQUENCE_LEN,
                                start_offset=START_OFFSET, step=STEP,
                                image_size=IMAGE_SIZE)
    return frames_, label_


if __name__ == "__main__":
    dataset_info = {
        'Archery': {
            'filename': 'Archery.zip',
            'url': 'https://github.com/lisaong/mldds-courseware/raw/master/data/ucf101-5classes/train/Archery.zip'
        },
        'Basketball': {
            'filename': 'Basketball.zip',
            'url': 'https://github.com/lisaong/mldds-courseware/raw/master/data/ucf101-5classes/train/Basketball.zip'
        }
    }

    dataset_dir = ''
    for info in dataset_info.values():
        # download the dataset if not cached at ~/.keras/datasets
        dataset_path = tf.keras.utils.get_file(fname=info['filename'],
                                               origin=info['url'],
                                               extract=True,
                                               cache_subdir='ucf101-5classes')
        dataset_dir = os.path.dirname(dataset_path)

    # get all the video files
    filenames = glob.glob(f'{dataset_dir}/*/*')
    print(filenames)

    frames_data = []
    labels = []
    for f in filenames:
        frames, label = parse_video_frames(f)

        # only take videos that have sufficient frames extracted
        if len(frames) == SEQUENCE_LEN:
            frames_data.append(frames)
            labels.append(label)

    X = np.stack(frames_data, axis=0)
    y = np.vstack(labels)
    print(X.shape, y.shape)

    print('========================================')
    print('Saving dataset')
    np.save('X.npy', X)
    np.save('y.npy', y)
