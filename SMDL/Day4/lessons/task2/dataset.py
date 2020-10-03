import tensorflow as tf
import os
import sys

sys.path.append('..')
from task1.frame_extractor import extract_frames, plot_images


def parse_video(filename):
    # get the label from the parent folder name
    # filename is a Tensor, so we need to use tf.strings.split
    parts = tf.strings.split(filename, os.sep)
    label_ = parts[-3]  # parent folder
    frames, _ = extract_frames(filename, 20,
                               start_offset=10, step=3,
                               image_size=(128, 128))
    return frames, label_


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
    print(dataset_dir)
    list_ds = tf.data.Dataset.list_files(f'{dataset_dir}/*/*.avi')

    # extract the frames from the video files
    frames_ds = list_ds.map(parse_video)

    for images, label in frames_ds.take(2):
        plot_images(images, title=label)
