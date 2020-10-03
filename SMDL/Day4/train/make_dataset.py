import cv2
import os
import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def extract_frames(video_path, sequence_len, start_offset=0, step=1,
                   image_size=(128, 128)):
    imgs = []
    ids = []

    count = 0
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened() and len(imgs) < sequence_len:
        ret, frame = cap.read()
        if not ret:
            print(f'No more frames from cap.read')
            break

        # after start_offset, take every step frame
        if count > start_offset and count % step == 0:
            # convert BGR format to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resize with linear interpolation
            frame = cv2.resize(frame, image_size,
                               interpolation=cv2.INTER_LINEAR)
            imgs.append([preprocess_input(frame)])
            ids.append(count)
        count += 1

    # cleanup opencv stuff
    cap.release()
    cv2.destroyAllWindows()

    # stack the images along the batch dimension (axis=0)
    return np.vstack(imgs), ids


def parse_video_frames(filename, sequence_len, start_offset, step, image_size):
    label_ = os.path.basename(os.path.split(filename)[0])  # parent folder
    frames_, _ = extract_frames(filename, sequence_len,
                                start_offset=start_offset, step=step,
                                image_size=image_size)
    return frames_, label_


def download_dataset(dataset_info, sequence_len, start_offset, step, image_size):

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

    frames_data = []
    labels = []
    for f in filenames:
        frames, label = parse_video_frames(f, sequence_len,
                                           start_offset, step,
                                           image_size)

        # only take videos that have sufficient frames extracted
        if len(frames) == sequence_len:
            frames_data.append(frames)
            labels.append(label)

    X = np.stack(frames_data, axis=0)
    y = np.vstack(labels)
    return X, y
